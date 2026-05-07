import numpy as np

from modeling.utils.random_utils import reseed_distributed
from modeling.Spatial import Building, LabeledRoadLine, RoadLine

def distribution_proportion(arr, max_val=None, min_val=None):
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)

    if max_val - min_val == 0:
        return np.array([1/len(arr)]*len(arr))

    return arr / np.sum(arr)

class SampleLocationPresentationStrategy:
    def __init__(self):
        pass
    def getSampleLocation(self, index):
        raise NotImplementedError("Function getSampleLocation must be implemented by a subclass.")
    def __len__(self):
        raise NotImplementedError("Function __len__ must be implemented by a subclass.")

#Below are the sample location presentation strategies to be used when the sample locations are known ahead of time
class PregeneratedSampleLocationPresentationStrategy(SampleLocationPresentationStrategy):
    def __init__(self):
        self._sample_locations = None
    def initialize_samples(self, sample_locations):
        self._sample_locations = sample_locations
    def getSampleLocation(self, index):
        return self._sample_locations[index]
    def __len__(self):
        return len(self._sample_locations)

class IndexSampleLocationPresentationStrategy(PregeneratedSampleLocationPresentationStrategy):
    pass

class WeightedSampleLocationPresentationStrategy(PregeneratedSampleLocationPresentationStrategy):
    def __init__(self, length, expected_class_balances, sample_selection_smoothing_alpha=0.1, sample_selection_smoothing_beta=2, balance_monitor="label"):
        super().__init__()
        self._length = length

        target_class_norm_denom = sum(expected_class_balances.values())
        self._sample_target_class_proportions = {k:v/target_class_norm_denom for k,v in expected_class_balances.items()}

        self._sample_selection_smoothing_alpha = sample_selection_smoothing_alpha
        self._sample_selection_smoothing_beta = sample_selection_smoothing_beta
        self._sample_class_observation_counts = {k:0 for k in expected_class_balances.keys()}
        self._class_weights_per_sample = {k:[] for k in expected_class_balances.keys()}
        self._balance_monitor = balance_monitor
        self.__random_state = np.random.RandomState()
        self.__samples_presented = 0

    def initialize_samples(self, sample_locations):
        super().initialize_samples(sample_locations)
        self._generate_sample_class_weights()

    def _balance_monitor_func(self, a):
        if self._balance_monitor == "label":
            return 1
        if self._balance_monitor == "pixel":
            #TODO: THIS WILL OVERWEIGHT LARGE OBJECTS THAT SPAN MORE THAN ONE FRAME
            #TODO: Consider how a sample with both building and roadline should be handled...
            if isinstance(a, Building):
                return a.getGeometry("pixel").area
            if isinstance(a, (LabeledRoadLine, RoadLine)):
                return a.getGeometry("pixel").length
            raise ValueError("Error: Unexpected Spatial Object Found. Expected Either RoadLine or Building.")
        raise ValueError("Error: Balance monitor must be one of label or pixel.")

    def  __len__(self):
        return self._length

    def _compute_sample_weights(self, sample, target_class):
        # Get roadlines in samples
        roadlines_in_samples = []
        for roadline in sample.getRoadLines():
            for subline in roadline.get_labeled_sub_lines():
                roadlines_in_samples.append(subline)

        spatial_objects_in_sample = sample.getBuildings() + roadlines_in_samples

        #If there are no spatial objects in this sample, then there is nothing valid in the sample
        if len(spatial_objects_in_sample) == 0:
            return 0

        #If there are spatial objects...
        #We need to get the weighting for the sample. To do this we compute the weight based on the number of
        #spatial objects in the sample that correspond to the target class
        numerator = 0
        for a in spatial_objects_in_sample:
            if a.getLabel() == target_class:
                numerator += self._balance_monitor_func(a)**self._sample_selection_smoothing_beta
            else:
                numerator += self._sample_selection_smoothing_alpha

        #And then we normalize that value based on the total number of spatial objects in the sample
        denominator = sum(self._smooth(a) for a in spatial_objects_in_sample)
        return numerator/denominator

    def _smooth(self, a):
        return self._sample_selection_smoothing_alpha + self._balance_monitor_func(a)**self._sample_selection_smoothing_beta

    def _generate_sample_class_weights(self):
        for target_class in self._class_weights_per_sample:
            for sample in self._sample_locations:
                self._class_weights_per_sample[target_class].append(self._compute_sample_weights(sample, target_class))
            self._class_weights_per_sample[target_class] = distribution_proportion(self._class_weights_per_sample[target_class])

    def _add_sample_class_observation_counts(self, sample_location):
        #First we compute the values of each class that appear in the sample location that was passed

        # Get roadlines in samples
        roadlines_in_sample_location = []
        for roadline in sample_location.getRoadLines():
            for subline in roadline.get_labeled_sub_lines():
                roadlines_in_sample_location.append(subline)

        spatial_objects_in_sample = sample_location.getBuildings() + roadlines_in_sample_location
        labels_in_sample = {a.getLabel():self._balance_monitor_func(a) for a in spatial_objects_in_sample}

        #Then we compute the proportions of the weights that will change based on the data in the sample location passed
        for k, v in labels_in_sample.items():
            self._sample_class_observation_counts[k] += v

    def _get_next_weighted_sample(self):
        #First, we need to sample which type of sample we want
        #To do this we see how far off from our target distribution we are
        observed_sample_proportions = distribution_proportion(list(self._sample_class_observation_counts.values()))
        observed_sample_proportions_dict = dict(zip(self._sample_class_observation_counts.keys(), observed_sample_proportions))

        #Compute how far from the expected class balances we are
        class_balance_target_deltas = {}
        for k in self._sample_target_class_proportions:
            class_balance_target_deltas[k] = observed_sample_proportions_dict[k] - self._sample_target_class_proportions[k]

        #We then normalize this data between 0 and 1
        class_balance_target_delta_max = max(class_balance_target_deltas.values())
        class_balance_target_delta_min = min(class_balance_target_deltas.values())
        class_balance_target_delta_spread = (class_balance_target_delta_max - class_balance_target_delta_min) + 1e-6
        class_balance_target_deltas_normed = {}
        for k, class_balance_target_delta in class_balance_target_deltas.items():
            class_balance_target_deltas_normed[k] = (class_balance_target_delta-class_balance_target_delta_min)/class_balance_target_delta_spread

        #We then smooth this data so the minority class has a chance of getting sampled
        class_balance_target_deltas_smoothed = {}
        for k in class_balance_target_deltas:
            smoothed_value = self._sample_selection_smoothing_alpha + (1 - class_balance_target_deltas_normed[k])**self._sample_selection_smoothing_beta
            class_balance_target_deltas_smoothed[k] = smoothed_value

        #Then we normalize the weights
        class_balance_target_deltas_smoothed_sum = sum(class_balance_target_deltas_smoothed.values())
        inv_class_balance_target_deltas_smoothed_normed = {}
        for k, delta_smoothed_val in class_balance_target_deltas_smoothed.items():
            inv_class_balance_target_deltas_smoothed_normed[k] = delta_smoothed_val/class_balance_target_deltas_smoothed_sum

        #We then sample the label pool we want to pick from
        class_ps = distribution_proportion(list(inv_class_balance_target_deltas_smoothed_normed.values()))
        selected_sample_bucket = np.random.choice(a=list(class_balance_target_deltas_smoothed.keys()), p=class_ps)

        index = np.random.choice(a=np.arange(0, len(self._sample_locations)), p=self._class_weights_per_sample[selected_sample_bucket])
        return self._sample_locations[index]

    def getSampleLocation(self, index):
        reseed_distributed(self.__samples_presented, self.__random_state)
        self.__samples_presented += 1

        selected_sample = self._get_next_weighted_sample()
        self._add_sample_class_observation_counts(selected_sample)
        return selected_sample

#Below are the sample location presentation strategies to be used when the sample locations are not known ahead of time
class RealTimeSampleLocationPresentationStrategy(SampleLocationPresentationStrategy):
    def observeSampleLocation(self, sample_location):
        raise NotImplementedError("Function observeSample must be implemented by a subclass.")
    def getSampleLocation(self, index):
        raise NotImplementedError("Function getSampleLocation must be implemented by a subclass.")

class MostRecentlyObservedSampleLocationPresentationStrategy(RealTimeSampleLocationPresentationStrategy):
    def __init__(self, length):
        super().__init__()
        self._length = length
        self._most_recent_sample = None
    def getSampleLocation(self, index):
        return self._most_recent_sample
    def observeSampleLocation(self, sample_location):
        self._most_recent_sample = sample_location
    def __len__(self):
        return self._length
