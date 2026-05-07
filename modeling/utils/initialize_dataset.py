from modeling.utils.data_augmentations import get_normalize_transform, get_tensor_transform

from modeling.adaptors.WindowedDatasetAdaptor import WindowedDatasetAdaptor
from modeling.datasets.WindowedDataset import WindowedDataset
from modeling.DataMap import Labels2IdxMap

from modeling.Models.model_registry import (
    LOCATIONSTRATEGY2MODULEMAPPING,
    MASKINGSTRATEGY2MODULEMAPPING,
    KEYPOINTSTRATEGY2MODULEMAPPING,
    SAMPLEANNOTATORTRATEGY2MODULEMPAPPING,
    PRESENTATIONTRATEGY2MODULEMAPPING,
)

def initialize_windowed_dataset(orthomosaics, channel_parameters, model_hyperparameters, datagen_hyperparameters, augmentation_transform):

    # Parse the dataset label map from the channel parameters
    input_dataset_label_map = Labels2IdxMap(
        channel_parameters["channel_maps"]["input_dataset_class_2_idx_map"],
        channel_parameters["channel_maps"]["background_class_idx"],
    )

    # Parse and initialize the masking strategy that should be used by the dataset
    print("\tInitializing Masking Strategy")
    masking_strategy_args = {}
    if "masking_strategy_parameters" in datagen_hyperparameters.keys():
        masking_strategy_args = datagen_hyperparameters["masking_strategy_parameters"]
    masking_strat = MASKINGSTRATEGY2MODULEMAPPING[model_hyperparameters["task"]](**masking_strategy_args)

    # Parse and initialize the keypoint augmentation strategy that should be used by the dataset
    print("\tInitializing Keypoint Strategy")
    keypoint_strat = KEYPOINTSTRATEGY2MODULEMAPPING[model_hyperparameters["task"]]()

    # Initialize the dataset based on task...
    print("\tInitializing Sample Location Generation Strategy")
    presentation_strategy_args = {}
    if "presentation_strategy_parameters" in datagen_hyperparameters.keys():
        presentation_strategy_args.update(datagen_hyperparameters["presentation_strategy_parameters"])

    # Initialize the location selection strategy that should be used to select image frames
    loc_pres_strat = PRESENTATIONTRATEGY2MODULEMAPPING[datagen_hyperparameters["presentation_strategy"]](**presentation_strategy_args)
    annotator = SAMPLEANNOTATORTRATEGY2MODULEMPAPPING[model_hyperparameters["task"]](**datagen_hyperparameters["annotator_parameters"])
    sample_location_strategy_args = {
        "annotator":annotator,
        "sample_location_presentation_strategy":loc_pres_strat,
        "orthomosaics":orthomosaics
    }
    if "location_parameters" in datagen_hyperparameters.keys():
        sample_location_strategy_args.update(datagen_hyperparameters["location_parameters"])
    location_strat = LOCATIONSTRATEGY2MODULEMAPPING[datagen_hyperparameters["location_strategy"]](**sample_location_strategy_args)

    # Initialize the adaptor that will consume all of the strategies we have initialized
    print("\tInitializing Dataset Adaptor. This may take a moment as it can involve generating samples to send to the model...")
    dataset_adaptor_args = {
        "orthomosaics": orthomosaics,
        "label_map": input_dataset_label_map,
        "sample_location_generation_strategy": location_strat,
        "keypoint_conversion_strategy": keypoint_strat,
    }
    dataset_adaptor_args.update(datagen_hyperparameters["dataset_adaptor_parameters"])
    adaptor = WindowedDatasetAdaptor(**dataset_adaptor_args)

    # Initialize the dataset with all the transforms that we care about
    print("\tInitializing Dataset")
    dataset = WindowedDataset(adaptor,
                              masking_strat,
                              augmentation_transform,
                              get_normalize_transform(),
                              get_tensor_transform(),
                              model_hyperparameters["input"]["normalized_inputs"])

    print("\tDone")

    return dataset
