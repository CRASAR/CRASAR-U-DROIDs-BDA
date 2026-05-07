# The Dataset Pipeline
This portion of the codebase contains following...
1) How to setup your environment to run everything
2) Logic for generating prelabeled tiles that will be passed to labelbox.com for annotation
3) Logic for coalescing the labelbox annotations into a dataset for ML model training

Each will be discussed in detail in the following sections. Each section contains a walk through for how to run each section of the code.

## Setup
This section describes how to set up your python environment in order to run this code.
Note: It is reccomended that you use [anaconda](https://www.anaconda.com/download) to manage your python enviornments.

Create a new anaconda environment with python 3.9 and install the [requirements.txt file](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/main/requirements.txt).
```
conda create --name suas_annotate python=3.9 -y
conda activate suas_annotate
pip install -r requirements.txt
```

Next, run the folder workspace generation script...
   * This script can be found [here](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/ScratchModel/scripts/bat_scripts/folder_setup.bat).
   * This script, will generate a number of folders in the targeted directory which will then house the inputs, intermediate steps, and outputs of the datapiline.

## Tiling for LabelBox.com 
In order to generate the tiles that were passed to labelbox the following steps need to be performed
1) Download and save the orthomosaics that you want to tile...

   * The data can be downloaded from [here](https://drive.google.com/drive/folders/17QrPmpxWICq-_fD3hVo1g_EPV6qgUeSk?usp=sharing).
   * It is suggested that you save this data in the `imagery/` folder that was generated during the setup.

2) Download the building polygons that you want to use...
   * This work leverages the Microsoft US Building Footprints data that is located [here](https://github.com/microsoft/USBuildingFootprints). Simply pick the state that you need to generate tiles for.
   * It is suggested that you save this data in the `building_polygons/Microsoft/` folder that was generated during the setup.

3) **Don't** Download the road data that you want to use...
   * This work leverages open street map (OSM) to generate its road lines. The downloading process is handled by the existing python logic. At this time there is no logic to allow you to pass your own roadlines.

4) Run the tiling script...
   * The script can be found [here](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/main/scripts/bat_scripts/image_tile_generate.bat) has been written as a Windows Bat File, however, the logic remains the same should you want to run the code on another OS.
   * Set the arguments in the script to refer to your local machine paths, and the parameters that you care about.
   * It is suggested that you set the `out_folder` argument to be the `workspace/` folder that was generated during the setup.
   * The script takes as arguments the following items. Additional arguments are specified in [tile.py](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/main/src/dataset/generators/tile.py)...
```
--tile_width THE DIMENSION IN PIXELS OF THE WIDTH OF THE TILE YOU WANT TO GENERATE 
--tile_height THE DIMENSION IN PIXELS OF THE HEIGHT OF THE TILE YOU WANT TO GENERATE 
--input_geotif THE SPACE SEPARATED LIST OF ORTHOMOSAIC PATHS TO BE PROCESSED
--out_folder THE FOLDER WHERE THE OUTPUT TILES WILL BE SAVED
--building_polygon_source THE SOURCE OF THE BUILDING POLYGONS: GEOJSON, OSM. IF "OSM" THEN THE BUILDING POLYGONS ARE DOWNLOADED FROM OSM. IF GEOJSON, THE BUILDING POLYGONS ARE SOURCED FROM THE PASSED BUILDING POLYGON FILE BELOW.
--geojson_building_polygon_path THE FILE CONTAINING THE BUILDING GEOJSONS DOWNLOADED IN STEP #2 
--generate_debug_images A FLAG DENOTING IF DEBUG IMAGES (TILES SHOWING BUILDINGS AND ROADS OVERLAYED) SHOULD BE GENERATED
--swap_ortho_xy A FLAG DENOTING IF THE CRS OF THE ORTHO HAS THE X AND Y AXIS SWAPPED.
```

5) Inspect your outputs...
   * This call script will populate your selected out_folder with several folders corresponding to image tiles, road masks, building masks, and their asssociated priortization (if prioritzation parameters are selected).

## Fuse And Clean LabelBox.com Annotations
This section describes how to take the orthomosaics that are being used in the project and align, merge, and clean those annotations into a cohesive dataset that can be used for machine learning workflows.
1) Download and save the orthomosaics that you want to tile...

   * The data can be downloaded from [here](https://drive.google.com/drive/folders/17QrPmpxWICq-_fD3hVo1g_EPV6qgUeSk?usp=sharing).
   * It is suggested that you save this data in the `imagery/` folder that was generated during the setup.

2) Move the imagery/data you care about into the appropriate imagery folders that were created in the step above.
   * Simply copy and past the orthomosaic imagery that you are interested in working with into the `imagery/` folder.
   * Copy and past the annotations downloaded from Labelbox.com into the appropriate folders inside `annotations/adj/raw/`, `annotations/bda/raw/`, and `annotations/rda/raw/`.

3) Run the data pipeline script of your choosing...
   * Either BDA [here](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/ScratchModel/scripts/bat_scripts/reconstruct_bda_annotations.bat) or RDA [here](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/ScratchModel/scripts/bat_scripts/reconstruct_rda_annotations.bat). 
   * At the time of writing the BDA script represents the most up mature version of the data pipeline, as a result the BDA pipeline will be the subject of the remainder of this section. It should be noted that the RDA pipeline follows a similar structure.

### Discussion of the BDA Data Pipeline Script

At the time of writing, the BDA Data Pipeline proceeds in 7 phases resulting in 12 calls to python scripts to generate intermediate and terminal data products. A diagram of the data pipeline is shown below. You are encouraged to follow along with in the diagram as we progress in this section.

![BDA Data Pipeline](https://github.tamu.edu/hrail/SUASDataAnnotation/blob/ScratchModel/src/dataset/BDA_data_pipeline.png)

The data flow begins in the top left of the image with the "Orthomosaics" oval.

1) First we must process the orthomosaics and collect two data products.
   * `build_ortho_to_path_map.py` - First, we must construct a mapping from orthomosaic titles to orthomosaic paths. We will use this so called "path map" to lookup the complete orthomosaic from its title. This map is specific to the disk/location where the data is saved.
   * `get_mask_from_orthos.py` - Second, we must construct a boundary file. This is a file that contains the bounds of the orthomosaic in terms of its geographic coordinates. This is so we can lookup the areas where we have imagery in the future.
2) Then we calculate the centroids of the orthomosaic boundaries so we have them as reference in the future.
   * `get_centroids_from_mask.py` - To do this we simply calculate the centroids of all of these boundaries. These centroids are saved as a csv.
3) Next, we combine the data from these orthomosaics with the data from labelbox. This is done first for the adjustment (ADJ) task, and then again for the BDA task. Each of these calls will write annotations to a folder, and generate an annotation path map which maps the orthomosaic name to the annotations that will be extracted from the Labelbox data.
   * `reconstruct_from_ADJ_annotations.py` - This first call takes in the adjustments from the `ADJ` task in labelbox, it consumes the raw annotations that have been exported from labelbox for the `ADJ` task, the projectID from labelbox, and the geotif pathmap we constructed above.
   * `reconstruct_from_ADJ_annotations.py` - This second call takes in the adjustments from the `ADJ Two` task in labelbox, it consumes the raw annotations that have been exported from labelbox, the projectID from labelbox for the `ADJ Two` task, and the geotif pathmap we constructed above.
   * `reconstruct_from_BDA_annotations.py` - This first call takes in the adjustments from the `BDA` task in labelbox, it consumes the raw annotations that have been exported from labelbox, the projectID from labelbox for the `BDA` task, and the geotif pathmap we constructed above.
   * `reconstruct_from_BDA_annotations.py` - This second call takes in the adjustments from the `BDA Bulk` task in labelbox, it consumes the raw annotations that have been exported from labelbox, the projectID from labelbox for the `BDA Bulk` task, and the geotif pathmap we constructed above. This call is notably different because it is performing bulk annotations. This means that the data was collected by annotating everything other than the bulk class. Simply put, this will convert all default labels ("unclassified") to the bulk label ("no-damage") unless labeled otherwise.
4) After this, we must fuse the annotation path maps that were generated by these reconstruction scripts called above. This is done by two calls, again for the `ADJ` data and the `BDA` data.
   * `fuse_path_maps` - This first call takes in the two path maps that were generated above for the two `ADJ` tasks, and fuses them into a single path map. At this point in the data pipeline, the adjustment annotations are done and we will not modify them further.
   * `fuse_path_maps` - This second call takes in the two path maps that were generated above for the two `BDA` tasks, and fuses them into a single path map.
5) Now, we must trim the building polygons that were outside the bounds of the imagery that we have. Ideally, this would have been done before the data was uploaded to LabelBox.com but this was not considered at the time.
   * `trim_BDA_polygons_by_boundaries.py` - This call takes in the BDA annotation path map, and the boundaries files we generated in step 1, and intersects the two. It then removes all polyongs that are not completely OR partially inside the boundary polygon area. Note that adjustments are not considered in this call. Polygons that could be adjusted into the boundary area are dropped. It should be noted that at this module, we also encounter our first "implicit pass" this means that the "Fused BDA Annotations" are not passed explicitly as arguments, instead, the files in that folder are looked up and accessed using the passed path map.
6) Next, we must update the BDA annotations with so called "spot checks" where we, as the developers of the dataset, make updates to the labeles because we believe they represent errors in the labeling and review process.
   * `integrate_BDA_spot_checks.py` - This call consumes the updated annotations ("spot checks") and the annotation path maps, it looks up the annotations in the path map, and then updates them based on the spot checks. This creates the final version of the BDA labels which we will not update at this point.
7) Finally, with the dataset completed, we compute statistics that we will use to characterize the dataset.
   * `compute_BDA_dataset_stats.py` - This call consumes the dataset annotations, and returns a csv that contains the count of each class grouped by orthomosaic name.

