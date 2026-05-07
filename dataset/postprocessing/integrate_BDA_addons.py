import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from constants import LAT_LON_CRS, NO_DAMAGE, MINOR_DAMAGE, MAJOR_DAMAGE, DESTROYED, UNCLASSIFIED
from dataset.utils.polygon_utils import get_polygon_id
from shapely.geometry import Polygon, MultiPolygon
from rasterio.transform import from_origin
from pyproj import Transformer

import argparse
import json
import geopandas as gpd
import rasterio
import random

def get_canonical_label(raw_label):
    clean_label = raw_label.lower()
    if("major" in clean_label):
        return MAJOR_DAMAGE
    elif("minor" in clean_label):
        return MINOR_DAMAGE
    elif("troyed" in clean_label):
        return DESTROYED
    elif("classified" in clean_label):
        return UNCLASSIFIED
    elif("no" in clean_label):
        return NO_DAMAGE
    else:
        raise Exception("Unknown class label: " + str(raw_label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='integrate_BDA_addons', description='This program combines the addons that were manually entered and udpates the annotations file.')
    parser.add_argument('--addons_folder', type=str, help="The path to the addons folder.")
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--geotif_path_map', type=str, help="The path to the geotiff file path map.")
    parser.add_argument('--addons_path_map', type=str, help="The path to the output addon annotations file path map.")
    parser.add_argument('--output_annotations_folder', type=str, help="The patht to the output annotations folder with the addons applied.")
    args = parser.parse_args()

    if(args.annotations_path_map):
        annotations_path_map = json.load(open(args.annotations_path_map))
    else:
        annotations_path_map = {}
    geotif_path_map = json.load(open(args.geotif_path_map))

    for (root, dir, files) in os.walk(args.addons_folder, topdown=True):
        for file in files:
            
            addons_path = os.path.join(root, file)
            
            target_geotif = file.replace("-BDA_CUSTOM_LABELS.geojson", "")
            print("Found Addon Annotations for " + target_geotif + " ...")

            geotif_path = geotif_path_map[target_geotif]

            try:
                annotation_path = annotations_path_map[geotif_path]

                # Load Annotations
                print("Loading the BDA annotations from:", annotation_path)
                f = open(annotation_path, "r")
                annotations_data = json.loads(f.read())
                f.close()

            except KeyError:
                print("No annotations for " + str(target_geotif))
                print("Using empty set of annotation for " + str(target_geotif))
                annotations_data = []

            # Load Addons
            print("Loading Addon Annotations from:", addons_path)
            addons_dataframe = gpd.read_file(addons_path)

            # Loading Geotif
            print("Loading Ortho at " + geotif_path)
            input_data = rasterio.open(geotif_path, "r")
            
            transform = input_data.transform

            count = 0
            for index, addon in addons_dataframe.iterrows():
                
                polygons = []
                
                if(type(addon["geometry"]) == MultiPolygon):
                    polygons = list(addon["geometry"].geoms)
                elif(type(addon["geometry"]) == Polygon):
                     polygons = [addon["geometry"]]
                
                
                for polygon in polygons:
                    coords = list(zip(*polygon.exterior.coords.xy))
                    labeled_coords = []
                    pixels_coords = []
                    for x,y in coords:
                        labeled_coords.append({"lat":y, "lon":x})

                        # Transform from lat/lon to pixel coordinates
                        coord_transformer = Transformer.from_crs(LAT_LON_CRS, input_data.crs.to_string())
                        x_t, y_t = coord_transformer.transform(y, x)
                        
                        pixel_x, pixel_y = input_data.index(x_t, y_t)
                        pixels_coords.append({"x": pixel_y, "y": pixel_x})
                    

                    label = get_canonical_label(addon["name"])

                    annotations_data.append({"source": "custom",
                        "id":get_polygon_id(),
                        "label":label, 
			           "pixels": pixels_coords,
			           LAT_LON_CRS:labeled_coords})
                    print("Adding Building Polyon with label - ", addon["name"])
                    count += 1

            print("Adding " + str(count) + " Building Polygons.")
            output_path = os.path.join(args.output_annotations_folder, target_geotif + ".json")
            print("Saving addon annotations to:", output_path)
            f = open(output_path, "w")
            f.write(json.dumps(annotations_data))
            f.close()
            annotations_path_map[geotif_path] = output_path
        
            print("Done.")
    
    print("Writing output addons annotations path map ...")
    f = open(args.addons_path_map, "w")
    f.write(json.dumps(annotations_path_map))
    f.close()
    print("Done.")