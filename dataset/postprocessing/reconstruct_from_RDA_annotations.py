import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.constants import LABEL_SCORE_PRIORITY_MAP, LABELBOX_DATASET_TO_ORTHO_TITLE, NO_DAMAGE, UNCLASSIFIED
from constants import SWAP_COORDS
from dataset.utils.adjustment_utils import apply_adjustments
from dataset.postprocessing.reconstruct_from_BDA_annotations import get_annotations_by_ortho, count_labels, convert_labeled_polygons_to_json_xy_and_lat_lon

from PIL import Image, ImageDraw
from rasterio import CRS
from shapely.geometry import Polygon, LineString
from pyproj import Transformer
from alive_progress import alive_bar
from collections import defaultdict

import json
import numpy as np
import rasterio
import os
import pandas as pd
import argparse

def parse_labeled_road_lines_from_annotations(annotations, project_key, swap_xy):
	labeled_polygons = []
	road_lines = []
	with alive_bar(total=len(annotations)) as bar:
		for annotation in annotations:
			file_name = annotation["data_row"]["global_key"]
			coords = file_name.split("(")[-1].split(")")[0].split(",")
			x = int(coords[0])
			y = int(coords[1])

			labels = annotation["projects"][project_key]["labels"]
			for label in labels:
				annotated_objects = label["annotations"]["objects"]
				for annotated_object in annotated_objects:
					name = annotated_object["name"]
					annotation_type = annotated_object["annotation_kind"]
					
					if annotation_type == "ImagePolyline":
						polyline = []
						for point in annotated_object["line"]:
							if (swap_xy):
								polyline.append([y + point["y"], x + point["x"]])
							else:
								polyline.append([x + point["x"], y + point["y"]])
						road_lines.append([name, LineString(polyline)])
					elif annotation_type == "ImagePolygon":
						polygon = []
						for point in annotated_object["polygon"]:
							polygon.append([x + point["x"], y + point["y"]])
						labeled_polygons.append([name, Polygon(polygon).buffer(0)])
			bar()
	return road_lines, labeled_polygons

def convert_road_lines_to_json_xy_and_lat_lon(road_lines, geotiff_data, adjustments, swap_xy):
	LAT_LON_CRS = "EPSG:4326"
	result = []
	coord_transformer = Transformer.from_crs(geotiff_data.crs.to_string(), LAT_LON_CRS)
	for label, road_line in road_lines:
		pixel_coord_line = []
		target_crs_line = []
		for x, y in list(zip(*road_line.coords.xy)):

			x_adj, y_adj = apply_adjustments(adjustments, x, y)

			pixel_coord_line.append({"x": x_adj, "y": y_adj})

			#Flip the y and x axis to align the data correctly in the coordinate space
			x_source, y_source = rasterio.transform.xy(geotiff_data.transform, y_adj, x_adj)
			if(swap_xy):
				x_t, y_t = coord_transformer.transform(y_source, x_source)
			else:
				x_t, y_t = coord_transformer.transform(x_source, y_source)
			target_crs_line.append({"lat": x_t, "lon": y_t})

		
		result.append({"source": "OSM",
					"label":label, 
			        "pixels":pixel_coord_line,
			        LAT_LON_CRS:target_crs_line})


	return result

if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='reconstruct_RDA', description='This program reconstructs the road lines and polygons from the annotations.')
	parser.add_argument('--annotations_file', type=str, help="The path to the annotations file.")
	parser.add_argument('--adjustments_file', type=str, help="The path to the adjustments file.", default=None)
	parser.add_argument('--project_key', type=str,  help='The labelbox project key.', default="clo6l72hz08ci073hcurw8js1")
	parser.add_argument('--geotif_path_map_file', type=str, help='The input file that maps from geotifs titles to their full path.')
	parser.add_argument('--geotif_annotation_map_file', type=str, help='The output file that maps from the geotif full paths to their annotation full paths.')
	parser.add_argument('--output_json_folder', type=str, help="The path to the file that will contain the output building polygons.")
	args = parser.parse_args()

	geotif_annotation_map = {}

	print("Loading Geotif File Path Mapping...")
	f = open(args.geotif_path_map_file, "r")
	geotif_path_map = json.loads(f.read())
	f.close()

	print("\n")
	print("Parsing annotations...")
	f = open(args.annotations_file, "r")
	annotations = f.readlines()
	f.close()

	print("\n")
	adjustments = None
	if(not args.adjustments_file is None):
		print("Parsing adjustments...")
		f = open(args.adjustments_file, "r")
		adjustments = json.loads(f.read())
		f.close()

	parsed_annotations = [json.loads(a) for a in annotations]

	ortho_to_annotations = get_annotations_by_ortho(parsed_annotations)

	ortho_label_counts = {}

	for ortho in ortho_to_annotations.keys():

		#Get the path to the ortho
		valid_ortho = False
		try:

			# Handling special case with Champlain Towers Ortho ... TODO: Need to test this when adj is implemented
			if ortho == "20210703-Champlain-Towers-South.geo.tif":
				ortho_tmp = "20210703-Champlain-Towers -South.geo.tif"
				ortho_local_title = LABELBOX_DATASET_TO_ORTHO_TITLE[ortho_tmp]
				geotif_path = geotif_path_map[ortho_local_title]
			else:
				ortho_local_title = LABELBOX_DATASET_TO_ORTHO_TITLE[ortho]
				geotif_path = geotif_path_map[ortho_local_title]
			valid_ortho = True
		except KeyError:
			print("Skipping annotations for orthomosaic:", ortho)
			print("Was unable to find an orthomosaic with that title.")

		if(valid_ortho):
			#Load the ortho
			print("Loading external base geotif from:", geotif_path)
			input_data = rasterio.open(geotif_path, "r")
			print("Done...")
		
			if ortho in SWAP_COORDS:
				road_lines, labeled_polygons = parse_labeled_road_lines_from_annotations(ortho_to_annotations[ortho], args.project_key, swap_xy=True)
			else:
				road_lines, labeled_polygons = parse_labeled_road_lines_from_annotations(ortho_to_annotations[ortho], args.project_key, swap_xy=False)
			print("Parsed annotations from", len(parsed_annotations), "tiles and found", len(labeled_polygons), "polygons, and", len(road_lines), "road lines.")

			adj = []
			if(adjustments):
				try:
					adj = adjustments[ortho]
					print("Found adjustments for this orthomosaic...")
				except KeyError as e:
					pass

			json_polygons = convert_labeled_polygons_to_json_xy_and_lat_lon(labeled_polygons, input_data, [])

			if ortho in SWAP_COORDS:
				print("Swapping Coordinates for roadlines ...")
				json_lines = convert_road_lines_to_json_xy_and_lat_lon(road_lines, input_data, adj, swap_xy=True)
			else:
				json_lines = convert_road_lines_to_json_xy_and_lat_lon(road_lines, input_data, adj, swap_xy=False)

			json_output = {"road_lines": json_lines, "polygons": json_polygons}

			print("\n")
			print("Writing polygons to json file...")
			out_file = ortho_local_title + ".json"
			out_path = os.path.join(args.output_json_folder, out_file)
			f = open(out_path, "w")
			f.write(json.dumps(json_output, indent=4, sort_keys=True))
			f.close()
			print("Polygons saved at", out_path)
			geotif_annotation_map[geotif_path] = out_path

	print("Geotif Annotation Path Map...")
	f = open(args.geotif_annotation_map_file, "w")
	f.write(json.dumps(geotif_annotation_map))
	f.close()
	print("Done...")