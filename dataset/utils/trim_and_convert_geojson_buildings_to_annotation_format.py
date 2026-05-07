import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from rasterio import CRS
from shapely.geometry import Polygon, shape, MultiPolygon
from pyproj import Transformer
from collections import defaultdict

from dataset.constants import UNCLASSIFIED

import json
import numpy as np
import rasterio
import os
import shapely
import pandas as pd
import argparse

from dataset.utils.polygon_utils import get_polygon_id

LAT_LON_CRS = "EPSG:4326"

def find_geotif_file_prefix_match(target_geotiff, candidate_files):
    a_c = target_geotiff.split(".geo")[0]
    for b in candidate_files:
        if(a_c.lower() == b.split(".geo")[0].lower()):
            return b
    return None

def convert_labeled_polygons_to_json_xy_and_lat_lon(polygon, geotiff_data, source, default_epsg_int=None):
	result = []
	if(geotiff_data.crs):
		coord_transformer_to_latlon = Transformer.from_crs(geotiff_data.crs.to_string(), LAT_LON_CRS)
		coord_transformer_to_crs = Transformer.from_crs(LAT_LON_CRS, geotiff_data.crs.to_string())
	else:
		coord_transformer_to_latlon = Transformer.from_crs("EPSG:"+str(default_epsg_int), LAT_LON_CRS)
		coord_transformer_to_crs = Transformer.from_crs(LAT_LON_CRS, "EPSG:"+str(default_epsg_int))
	
	individual_polys = []
	if polygon.geom_type == "MultiPolygon":
		individual_polys = list(polygon.geoms)
	else:
		individual_polys = [polygon]
	
	for p in individual_polys:	
		pixel_coords_polygon = []
		target_crs_polygon = []
		
		for lon, lat in list(zip(*p.exterior.coords.xy)):
			x_source, y_source = coord_transformer_to_crs.transform(lat, lon)

			row, col = rasterio.transform.rowcol(geotiff_data.transform, x_source, y_source)
			pixel_coords_polygon.append({"x": col, "y": row})

			x_t, y_t = coord_transformer_to_latlon.transform(x_source, y_source)
			target_crs_polygon.append({"lat": lat, "lon": lon})

	result.append({"source": source, 
				   "label": UNCLASSIFIED, 
				   "pixels":pixel_coords_polygon,
				   "id": get_polygon_id(),
				   LAT_LON_CRS:target_crs_polygon})

	return result

def get_bounding_polygon(geotiff_data, default_epsg_int=None):
	geotif_bounds = geotiff_data.bounds
	geotif_bounds_transformed_coords = []
	if(geotiff_data.crs):
		coord_transformer = Transformer.from_crs(geotiff_data.crs.to_string(), LAT_LON_CRS)
	else:
		coord_transformer = Transformer.from_crs("EPSG:"+str(default_epsg_int), LAT_LON_CRS)
	geotif_bounds_transformed_coords.append(coord_transformer.transform(geotif_bounds.left, geotif_bounds.top))
	geotif_bounds_transformed_coords.append(coord_transformer.transform(geotif_bounds.right, geotif_bounds.top))
	geotif_bounds_transformed_coords.append(coord_transformer.transform(geotif_bounds.right, geotif_bounds.bottom))
	geotif_bounds_transformed_coords.append(coord_transformer.transform(geotif_bounds.left, geotif_bounds.bottom))
	geotif_bounds_transformed_coords.append(coord_transformer.transform(geotif_bounds.left, geotif_bounds.top))
	return  Polygon([[y,x] for x, y in geotif_bounds_transformed_coords])

def load_boundary_polygon(path_to_boundary_polygon):
	f = open(path_to_boundary_polygon, "r")
	boundary_data = json.load(f)
	f.close()
	polygon_boundaries = []
	for i in range(0, len(boundary_data)):
	    polygon_boundaries.append(shape(boundary_data[i]["geometry"]))
	return shapely.MultiPolygon(polygon_boundaries)

def convert_geojson_buildings_in_geotiff_area(geojson_polygons, geotiff_data, boundary, source="Loaded", default_epsg_int=None):
	converted_buildings = []
	for feat in geojson_polygons["features"]:
		for building in feat['geometry']['coordinates']:
			building_polygon = Polygon(building).buffer(0)
			if(shapely.intersection(boundary, building_polygon).area > 0):
				converted_buildings.extend(convert_labeled_polygons_to_json_xy_and_lat_lon(building_polygon, geotiff_data, source, default_epsg_int))

	return converted_buildings

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--building_polygons_geojson_folder', type=str, help="The path to the folder containing the building polygons in the geojson format.")
	parser.add_argument('--geotif_path_map_file', type=str, help='The input geotif path map file')
	parser.add_argument('--output_annotation_folder', type=str, help='The path to the folder that will contain the converted building polygons.')
	parser.add_argument('--boundaries_folder', type=str, help='The path to the folder that will contain orthomosaic_boundaries.')
	parser.add_argument('--default_epsg_int', type=int, help="The integer used to identify the CRS of orthosmosaics loaded when their transforms are stored in an external file.", default=None)
	args = parser.parse_args()

	print("Parsing building polygons...")
	geojson_features = {}
	for file in os.listdir(args.building_polygons_geojson_folder):
		if(file.endswith("json")):
			f = open(os.path.join(args.building_polygons_geojson_folder, file), "r")
			geojson_features[file] = json.load(f)
			f.close()
	print("Done...")

	print("Loading Geotif File Path Mapping...")
	f = open(args.geotif_path_map_file, "r")
	geotif_path_map = json.loads(f.read())
	f.close()

	print("Loading boundary polygons...")
	geotif_to_boundaries = {}
	boundary_folder_contents = os.listdir(args.boundaries_folder)
	for geotif_name in geotif_path_map.keys():
		geotif_to_boundaries[geotif_name] = find_geotif_file_prefix_match(geotif_name, boundary_folder_contents)

	for prefix, geotif_path in geotif_path_map.items():
		geotif_file = os.path.split(geotif_path)[-1]
		converted_buildings = []
		for geojson_file, features in geojson_features.items():
			print("Loading external base geotif from:", geotif_path)
			input_data = rasterio.open(geotif_path, "r")

			if(geotif_to_boundaries[geotif_file]):
				print("Found boundary polygon...")
				boundary = load_boundary_polygon(os.path.join(args.boundaries_folder, geotif_to_boundaries[geotif_file]))
			else:
				print("Did not find boundary polygon, using geotif bounds as boundary...")
				boundary = get_bounding_polygon(input_data, args.default_epsg_int)

			print("Converting building polygons...")
			converted_buildings.extend(convert_geojson_buildings_in_geotiff_area(features, input_data, boundary, source=geojson_file, default_epsg_int=args.default_epsg_int))

		print("Found", len(converted_buildings), "building polygons in the bound of this orthomosaic.")
		print("Writing buidling polygons...")
		outpath = os.path.join(args.output_annotation_folder, geotif_file + ".json")
		f = open(outpath, "w")
		f.write(json.dumps(converted_buildings))
		f.close()

	print("Done...")




