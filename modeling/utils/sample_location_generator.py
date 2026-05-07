import time
from multiprocessing import Pool

import numpy as np

from modeling.utils.sample_generator_utils import generate_sample_point, get_valid_lines, get_valid_buildings
from modeling.utils.building_frame_generation import get_candidate_samples_center
from modeling.utils.sample_presentation import RealTimeSampleLocationPresentationStrategy, PregeneratedSampleLocationPresentationStrategy
from modeling.utils.random_utils import reseed_distributed
from modeling.utils.gsd_utils import get_gsd_steps
from modeling.Spatial import MultiLabeledRoadLineFactory

# This is a class that implements a strategy for generating sample locations in an orthomosaic
# The expectation is that if you want to get different behavior from your data generator, you
# Can pass these different strategies to your code and it it will utilize the different strategy
# as needed. This is the entry point into the class hierarchy.
class SampleLocationGenerationStrategy:
    def __init__(self,
                 strategy_name,
                 sample_location_presentation_strategy,
                 annotator,
                 tile_x,
                 tile_y,
                 scale_strategy="pixel",
                 min_gsd_x=0.0,
                 min_gsd_y=0.0,
                 max_gsd_x=1e10,
                 max_gsd_y=1e10,
                 multiscale_steps=1,
                 multiscale_step_interval="linspace"):
        self._strategy_name = strategy_name
        self._sample_location_presentation_strategy = sample_location_presentation_strategy
        self._annotator = annotator
        self._xdim = tile_x
        self._ydim = tile_x
        self._x_sample_rate = None
        self._y_sample_rate = None
        self._scale_strategy = scale_strategy
        self._multiscale_steps_gsd_x, self._multiscale_steps_gsd_y = get_gsd_steps(min_gsd_x,
                                                                                   min_gsd_y,
                                                                                   max_gsd_x,
                                                                                   max_gsd_y,
                                                                                   multiscale_steps,
                                                                                   multiscale_step_interval)
        assert(len(self._multiscale_steps_gsd_x) == len(self._multiscale_steps_gsd_y))
        self._multiscale_steps_count = len(self._multiscale_steps_gsd_x)
        self._initializeLocationGenerationStrategy(scale_strategy)

    def _initializeLocationGenerationStrategy(self, scale):
        if scale == "pixel":
            self._xdim = int(self._xdim)
            self._ydim = int(self._ydim)
        elif scale == "spatial":
            self._xdim = float(self._xdim)
            self._ydim = float(self._ydim)
        elif scale == "spatial_px":
            self._xdim = int(self._xdim)
            self._ydim = int(self._ydim)
        else:
            raise ValueError("Unknown dimension scale passed " + str(scale) + ". Options are " + str(["pixel", "spatial", "spatial_px"]))
        self._scale_strategy = scale
    def _getXDim_scaled(self):
        return int(np.around(self._xdim / self._x_sample_rate))
    def _getYDim_scaled(self):
        return int(np.around(self._ydim / self._y_sample_rate))
    def _getXDim(self):
        return self._xdim
    def _getYDim(self):
        return self._ydim
    
    def _prime_scale_strategy(self, ortho, gsd_override=None):
        if self._scale_strategy == "pixel":
            self._x_sample_rate = 1
            self._y_sample_rate = 1
        elif self._scale_strategy == "spatial":
            self._x_sample_rate, self._y_sample_rate = ortho.get_gsd()[:2]
        elif self._scale_strategy == "spatial_px":
            if gsd_override == None:
                raise ValueError("When operating in dim_scale mode \"spatial_px\" you must pass a gsd_override.")
            #Compute the number of pixels that we want to sample so that when resized to the pixel xdim, will be a target gsd
            gsd_x, gsd_y = ortho.get_gsd()[:2]
            self._x_sample_rate = gsd_x / gsd_override[0]
            self._y_sample_rate = gsd_y / gsd_override[1]
    def getStrategyName(self):
        return self._strategy_name
    def getAnnotator(self):
        return self._annotator
    def getSampleLocation(self, index):
        raise NotImplementedError("getSampleLocations must be implemented by a subclass")
    def __len__(self):
        return len(self._sample_location_presentation_strategy)

# This is a class that is specifically for sample generation strategies that involve generating
# Samples in real time when the function getSampleLocation is called.
class RealTimeSampleLocationGenerationStrategy(SampleLocationGenerationStrategy):
    def __init__(self, *args, orthomosaics, **kwargs):
        self._orthomosaics = orthomosaics
        super().__init__(*args, **kwargs)

        if not isinstance(sample_location_presentation_strategy, RealTimeSampleLocationPresentationStrategy):
            raise ValueError("sample_location_presentation_strategy must be an instance of ",
                             RealTimeSampleLocationPresentationStrategy,
                             "instead found",
                             type(sample_location_presentation_strategy))

    def getSampleLocation(self, index):
        raise NotImplementedError("getSampleLocations must be implemented by a subclass")

# This is a class that iteratively generates a random sample location until it finds a valid one
class RandomSampleLocationGenerationStrategy(RealTimeSampleLocationGenerationStrategy):
    def __init__(self, *args, **kwargs):
        kwargs["strategy_name"]="Random"
        self._sample_acceptance_persistence = int(sample_acceptance_persistence)
        self.__rs = np.random.RandomState()
        self.__seed_range = int(float(seed_range))
        self._samples_generated = 0
        super().__init__(*args, **kwargs)

    def getSampleLocation(self, index):
        reseed_distributed(self._samples_generated, self.__rs, self.__seed_range)
        self._samples_generated += 1

        orthomosaic_idx = self.__rs.randint(0, len(self._orthomosaics))
        ortho = self._orthomosaics[orthomosaic_idx]

        attempts = 0
        t_sample_generation = 0
        t_sample_annotation = 0
        t_sample_validation = 0
        accepted = False
        exceptions = {}
        while((not accepted) and attempts < self._sample_acceptance_persistence):
            t_0 = time.time()
            x_p, y_p = generate_sample_point(ortho, self.__rs)
            t_1 = time.time()

            gsd_target = np.random.choice(zip(self._multiscale_steps_gsd_x, self._multiscale_steps_gsd_y))
            self._prime_scale_strategy(ortho, gsd_override=gsd_target)
            validation_call_args = self._annotator.make_sample_annotation_call_args(x_p, y_p, self._getXDim_scaled(), self._getYDim_scaled(), ortho, orthomosaic_idx, None)
            sample_candidate = self._annotator.annotate_sample(*validation_call_args)

            #Combine the exceptions with what we have so far
            exceptions = {e:cur_exceptions + sample_candidate.getGenerationMetadata().getExceptions()[e] for e, cur_exceptions in exceptions.items()}

            t_2 = time.time()
            if len(sample_candidate.getBuildings()) > 0 or len(sample_candidate.getRoadLines()) > 0:
                accepted = True
            t_3 = time.time()

            t_sample_generation += t_1-t_0
            t_sample_annotation += t_2-t_1
            t_sample_validation += t_3-t_2

            attempts += 1

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(attempts, t_sample_generation, t_sample_annotation, t_sample_validation, exceptions)

        # Return the valid road lines that were generated for the sample.
        result = SampleLocation(x=x_p,
                                y=y_p,
                                x_dim=self._getXDim_scaled(),
                                y_dim=self._getYDim_scaled(),
                                resize_x_target=self._getXDim(),
                                resize_y_target=self._getYDim(),
                                gsd_target=gsd_target,
                                buildings=sample_candidate.getBuildings(),
                                roadlines=sample_candidate.getRoadLines(),
                                orthomosaic_idx=orthomosaic_idx,
                                generation_meta=generation_meta)

        self._sample_location_presentation_strategy.observeSampleLocation(result)
        return self._sample_location_presentation_strategy.getSampleLocation(index)


# This is a subclass that generates sample locations all at once and validates them for the user
# so that when the user calls the getSampleLocation function, there is a valid sample waiting for
# them to pass to a model.
class PregeneratedSampleLocationGenerationStrategy(SampleLocationGenerationStrategy):
    def __init__(self, orthomosaics, *args, sample_generator_process_pool_size=6, **kwargs):
        self._sample_generator_process_pool_size = sample_generator_process_pool_size
        self._samples = []
        self._orthomosaics = orthomosaics
        super().__init__(*args, **kwargs)

        if not isinstance(self._sample_location_presentation_strategy, PregeneratedSampleLocationPresentationStrategy):
            raise ValueError("sample_location_presentation_strategy must be an instance of ",
                             PregeneratedSampleLocationPresentationStrategy,
                             "instead found",
                             type(kwargs["sample_location_presentation_strategy"]))

    def _initializeLocationGenerationStrategy(self, scale="pixel"):
        super()._initializeLocationGenerationStrategy(scale)
        self._samples = self._pregenerate_sample_locations(self._orthomosaics)
        self._sample_location_presentation_strategy.initialize_samples(self._samples)

    def getSampleLocation(self, index):
        return self._sample_location_presentation_strategy.getSampleLocation(index)

    def _generate_sample_location_meta(self, gsd_x, gsd_y):
        return {"gsd_x": gsd_x, "gsd_y": gsd_y}

    def _pregenerate_sample_locations(self, orthomosaics):
        testing_locations = []
        testing_location_metas = []
        for i, orthomosaic in enumerate(orthomosaics):
            locs = self._get_sample_locations_to_validate(orthomosaic, i)
            testing_locations.extend(locs[0])
            testing_location_metas.extend(locs[1])

        result = []
        with Pool(processes=self._sample_generator_process_pool_size) as pool:
            candidate_samples = pool.starmap(self._annotator.annotate_sample, testing_locations)

        for candidate_sample, metas in zip(candidate_samples, testing_location_metas):
            candidate_sample.setGSDTarget(metas["gsd_x"], metas["gsd_y"])
            candidate_sample.setResizeTarget(self._getXDim(), self._getYDim())
            if len(candidate_sample.getBuildings()) > 0 or len(candidate_sample.getRoadLines()) > 0:
                result.append(candidate_sample)
        return result

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        raise NotImplementedError("_get_sample_locations_to_validate must be implemented by a subclass")

# This is a subclass that generates samples for building damage assessment and alignment training that
# Attempts to include as many buildings in a frame as possible while keeping them within the range
# of adjustment_buffer_distance_px pixels from the edge of the frame
class CenteredBuildingSampleStrategy(PregeneratedSampleLocationGenerationStrategy):
    def __init__(self, adjustment_buffer_distance_px, *args, **kwargs):
        kwargs["strategy_name"]="Centered"
        
        self._adjustment_buffer_distance_px = int(adjustment_buffer_distance_px)
        super().__init__(*args, **kwargs)

        if not isinstance(self._annotator, BDASampleAnnotator):
            raise ValueError("CenteredSampleStrategy is only defined for samples that can be validated using the BDASampleAnnotator.")

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        sample_validation_calls = []
        sample_validation_call_metas = []
        frames_of_buildings = []
        for gsd_x, gsd_y in zip(self._multiscale_steps_gsd_x, self._multiscale_steps_gsd_y):
            self._prime_scale_strategy(orthomosaic, gsd_override=(gsd_x, gsd_y))
            frames_of_buildings = get_candidate_samples_center(orthomosaic,
                                                               self._getXDim_scaled(),
                                                               self._getYDim_scaled(),
                                                               self._adjustment_buffer_distance_px,
                                                               adjusted=self._annotator.generatesAdjustedSamples())
            for frame_of_buildings in frames_of_buildings:        
                x = frame_of_buildings[0].centroid.x - self._getXDim_scaled()/2
                y = frame_of_buildings[0].centroid.y - self._getYDim_scaled()/2
                building_ids = frame_of_buildings[1]
                sample_validation_calls.append(self._annotator.make_sample_annotation_call_args(x,
                                                                                                y,
                                                                                                self._getXDim_scaled(),
                                                                                                self._getYDim_scaled(),
                                                                                                orthomosaic,
                                                                                                orthomosaic_idx,
                                                                                                building_ids))
                sample_validation_call_metas.append(self._generate_sample_location_meta(gsd_x, gsd_y))
        return sample_validation_calls, sample_validation_call_metas

# This is a subclass that generates sample for building and road damage assessment by uniformly tiling
# the image into a grid.
class GridSampleStrategy(PregeneratedSampleLocationGenerationStrategy):
    def __init__(self, adjustment_buffer_distance_px, *args, **kwargs):
        kwargs["strategy_name"]="Grid"
        super().__init__(*args, **kwargs)

        self._adjustment_buffer_distance_px = adjustment_buffer_distance_px

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        # Create a list to store the candidate samples
        sample_validation_calls = []
        sample_validation_call_metas = []
        #For every gsd that we are going to consider. This will be a single entry when operating in spatial and pixel modes
        for gsd_x, gsd_y in zip(self._multiscale_steps_gsd_x, self._multiscale_steps_gsd_y):
            self._prime_scale_strategy(orthomosaic, gsd_override=(gsd_x, gsd_y))
            # Iterate over the orthomosaic looking for buildings that need to be labeled
            for x in np.arange(0-self._adjustment_buffer_distance_px, orthomosaic.get_width(), self._getXDim_scaled()-2*self._adjustment_buffer_distance_px):
                for y in np.arange(0-self._adjustment_buffer_distance_px, orthomosaic.get_height(), self._getYDim_scaled()-2*self._adjustment_buffer_distance_px):
                    call = self._annotator.make_sample_annotation_call_args(x, y, self._getXDim_scaled(), self._getYDim_scaled(), orthomosaic, orthomosaic_idx, None)
                    sample_validation_calls.append(call)
                    sample_validation_call_metas.append(self._generate_sample_location_meta(gsd_x, gsd_y))
        return sample_validation_calls, sample_validation_call_metas

class SampleLocation:
    def __init__(self, x, y, x_dim, y_dim, buildings, roadlines, orthomosaic_idx, generation_meta=None, resize_x_target=None, resize_y_target=None, gsd_target=None):
        self._x = x
        self._y = y
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._resize_x_target = resize_x_target
        self._resize_y_target = resize_y_target
        self._gsd_target = gsd_target
        self._buildings = buildings
        self._roadlines = roadlines
        self._orthomosaic_idx = orthomosaic_idx
        self._generation_meta = generation_meta
        if self._generation_meta is None:
            self._generation_meta = SampleLocationGenerationMetadata()

    def getX(self):
        return self._x
    def getY(self):
        return self._y
    def getXDim(self):
        return self._x_dim
    def getYDim(self):
        return self._y_dim
    def getResizeXTarget(self):
        return self._resize_x_target
    def getResizeYTarget(self):
        return self._resize_y_target
    def getGSDTarget(self):
        return self._gsd_target
    def getBuildings(self):
        return self._buildings
    def getRoadLines(self):
        return self._roadlines
    def getOrthomosaicIdx(self):
        return self._orthomosaic_idx
    def getGenerationMetadata(self):
        return self._generation_meta
    def setResizeTarget(self, resize_x_target, resize_y_target):
        self._resize_x_target = resize_x_target
        self._resize_y_target = resize_y_target
    def setGSDTarget(self, gsd_x, gsd_y):
        self._gsd_target = (gsd_x, gsd_y)

class SampleLocationGenerationMetadata:
    def __init__(self, attempts=1, generation_sec=0.0, annotation_sec=0.0, validation_sec=0.0, exceptions=None):
        self._exceptions = exceptions
        if self._exceptions is None:
            self._exceptions = {}
        self._attempts = attempts
        self._generation_sec = generation_sec
        self._annotation_sec = annotation_sec
        self._validation_sec = validation_sec
    def getExceptions(self):
        return self._exceptions
    def getAttempts(self):
        return self._attempts
    def getGenerationSec(self):
        return self._generation_sec
    def getAnnotationSec(self):
        return self._annotation_sec
    def getValidationSec(self):
        return self._validation_sec

class SamplePregenerationAnnotator:
    def __init__(self, generate_adjusted_sample_locations, center_xy=False):
        self._generate_adjusted_sample_locations = generate_adjusted_sample_locations
        self._center_xy = center_xy
    def expectsCenteredXY(self):
        return self._center_xy
    def generatesAdjustedSamples(self):
        return self._generate_adjusted_sample_locations

class BDASampleAnnotator(SamplePregenerationAnnotator):
    def __init__(self, generate_adjusted_sample_locations, center_xy=False, building_intersection_proportion_threshold=0.0):
        super().__init__(generate_adjusted_sample_locations, center_xy)
        self._building_intersection_proportion_threshold = float(building_intersection_proportion_threshold)
    def make_sample_annotation_call_args(self, x, y, x_dim, y_dim, orthomosaic, orthomosaic_idx, ids=None):
        return [x,
                y,
                x_dim,
                y_dim,
                orthomosaic.get_buildings(adjusted=self._generate_adjusted_sample_locations, ids=ids),
                orthomosaic_idx]
    def annotate_sample(self, x, y, x_dim, y_dim, buildings, orthomosaic_idx):
        t0 = time.time()
        # Get the valid polygons for this window
        exceptions = {}
        valid_buildings, exceptions = get_valid_buildings(
            x=x,
            y=y,
            buildings=buildings,
            x_dim=x_dim,
            y_dim=y_dim,
            building_intersection_proportion_threshold=self._building_intersection_proportion_threshold,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        t1 = time.time()

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, exceptions)

        # Return the valid buildings that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=valid_buildings,
                              roadlines=[],
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)

class RDASampleAnnotator(SamplePregenerationAnnotator):
    def make_sample_annotation_call_args(self, x, y, x_dim, y_dim, orthomosaic, orthomosaic_idx, _):
        return [x,
                y,
                x_dim,
                y_dim,
                orthomosaic.get_road_lines(adjusted=self._generate_adjusted_sample_locations),
                orthomosaic.get_road_line_annotation_polygons(),
                orthomosaic_idx]

    def annotate_sample(self, x, y, x_dim, y_dim, roadlines, annotation_polygons, orthomosaic_idx):
        t0 = time.time()
        exceptions = {}
        valid_road_lines, exceptions = get_valid_lines(
            x,
            y,
            roadlines,
            x_dim=x_dim,
            y_dim=y_dim,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        labeled_road_lines = MultiLabeledRoadLineFactory(valid_road_lines, annotation_polygons)
        t1 = time.time()

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, exceptions)

        # Return the valid road lines that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=[],
                              roadlines=labeled_road_lines,
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)
