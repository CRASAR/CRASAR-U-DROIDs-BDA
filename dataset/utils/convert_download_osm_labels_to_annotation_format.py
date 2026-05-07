import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.constants import LAT_LON_CRS
from dataset.utils.ortho_utils import transform_bounds
from dataset.generators.osm_data_generator import get_osm_roads_buildings_and_nodes_for_bounding_box
from dataset.utils.draw_utils import get_road_polylines_from_osm_data

from pyproj import Transformer

import rasterio
import argparse
import json

def download_and_convert_road_lines(geotif_path, output_folder):
	cropped_data = rasterio.open(geotif_path, "r")

	# Initialize a transform to convert from lat lon to the target crs of the ortho
	lat_lon_to_crs_transformer = Transformer.from_crs(
	    LAT_LON_CRS, cropped_data.crs.to_string())
	
	crs_to_lat_lon_crs_transformer = Transformer.from_crs(
	    cropped_data.crs.to_string(), LAT_LON_CRS)

	ortho_lat_lon_bounds = transform_bounds(cropped_data, LAT_LON_CRS)
	osm_bbox = [ortho_lat_lon_bounds.bottom, ortho_lat_lon_bounds.left,
	            ortho_lat_lon_bounds.top, ortho_lat_lon_bounds.right]

	road_and_building_data = get_osm_roads_buildings_and_nodes_for_bounding_box(
	    osm_bbox)
	
	road_polylines = get_road_polylines_from_osm_data(road_and_building_data, cropped_data, lat_lon_to_crs_transformer, False)

	output = {"polygons":[], "road_lines":[]}
	for line in road_polylines:
		geo_line = []
		for x, y in line:
			crs_x, crs_y = cropped_data.xy(y, x)
			lat, lon = crs_to_lat_lon_crs_transformer.transform(crs_x, crs_y)
			geo_line.append([lat, lon]) 
		output["road_lines"].append({
			"source":"OSM",
			"label": "Road Line",
			"pixels": [{"x":x, "y":y} for x, y in line],
			"EPSG:4326": [{"lat":lat, "lon":lon} for lat, lon in geo_line]
		})
	output_path = os.path.join(output_folder, os.path.split(geotif_path)[-1]+".json")
	f = open(output_path, "w")
	f.write(json.dumps(output))
	f.close()

	return output_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='convert_download_osm_labels__to_annotation_format',
	                                 description='This program downloads and then converts the roadlines from OSM into the format that is useful for the dataset.')
	parser.add_argument('--geotif_path_map_file', type=str, help='The path to the path map containing the input geotifs to be processed.')
	parser.add_argument('--output_folder', type=str, help='The path where the output data should be saved.')
	parser.add_argument('--output_path_map', type=str, help='The path to the pathmap which will be used to store the annotations')
	args = parser.parse_args()
	
	imagery_path_map = json.load(open(args.geotif_path_map_file))

	path_map = {}
	for name, full_path in imagery_path_map.items():
		road_lines_path = download_and_convert_road_lines(full_path, args.output_folder)
		path_map[full_path] = road_lines_path

	f = open(args.output_path_map, "w")
	f.write(json.dumps(path_map))
	f.close()
	
