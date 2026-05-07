import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from shapely.geometry import Polygon, shape, LineString
from pyproj import Transformer

from dataset.utils.adjustment_utils import match_vertex_to_adjustment, apply_adjustments
from constants import ORTHO_GSD

import json
import argparse
import shapely
import rasterio

BOUNDARY_POLYGON_BUFFER = 100
BUFFER_DISTANCE = 305

if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='trim_RDA_polygons_roadlines_by_boundaries', description='This program removes all roadlines and polyongs that are entirely outside the bounds of the orthomosaic polygon.')
	parser.add_argument('--geotif_annotation_map_file', type=str, help='The file that maps from the geotif full paths to their annotation full paths.')
	parser.add_argument('--adjustments_path_map', type=str, help="The path to the adjustments file path map.", default=None)
	parser.add_argument('--boundaries_folder_path', type=str, help='The folder that contains all of the boundary polygons for the different orthos.')
	parser.add_argument('--output_json_folder', type=str, help="The path to the file that will contain the output building polygons.")
	parser.add_argument('--trimmed_path_map', type=str, help="The path to the output trimmed annotations file path map.")
	args = parser.parse_args()

	try:
		os.makedirs(args.output_json_folder)
	except FileExistsError as e:
		pass

	annotations_path_map = json.load(open(args.geotif_annotation_map_file))
	adjustments_path_map = json.load(open(args.adjustments_path_map))
	trimmed_annotations_path_map = {}

	for geotif_path, annotation_path in annotations_path_map.items():

		target_geotif = os.path.split(geotif_path)[1]

		#Load the annotations
		print("Loading the RDA annotations from:", annotation_path)
		with open(annotation_path, "r") as f:
			annotations_data = json.loads(f.read())

		#Load the ADJ annotations
		adjustments_path = None
		try:
			adjustments_path = adjustments_path_map[geotif_path]
			print("Loading the ADJ RDA annotations from:", adjustments_path)
		except KeyError:
			print("Could not find adjustments for", target_geotif)
			print("Generating unadjusted debug RDA geotiff")
		adjustments = {}
		if(not adjustments_path is None):
			print("Parsing adjustments...")
			f = open(adjustments_path, "r")
			adjustments = json.loads(f.read())
			f.close()

		#Load the polygon boundary
		boundary_path = os.path.join(args.boundaries_folder_path, target_geotif + ".json")
		print("Loading the boundary polygon from:", boundary_path)
		with open(boundary_path, "r") as f:
			boundary_data = json.load(f)

		polygon_boundaries = [shape(boundary_data[i]["geometry"]) for i in range(0, len(boundary_data))]

		print("Loading the target geotiff metadata from:", geotif_path)
		input_geotiff_data = rasterio.open(geotif_path, "r")

		LAT_LON_CRS = "EPSG:4326"
		transform = input_geotiff_data.transform
		coord_transformer = Transformer.from_crs(LAT_LON_CRS, input_geotiff_data.crs.to_string())
		coord_transformer_to_latlon = Transformer.from_crs(input_geotiff_data.crs.to_string(), LAT_LON_CRS, always_xy=True)
		
		valid_buildings = []
		for building in annotations_data["polygons"]:
			coords = [(p["lon"], p["lat"]) for p in building["EPSG:4326"]]
			building_polygon = Polygon(coords)
			valid = False
			for polygon_boundary in polygon_boundaries:
				if(building_polygon.intersection(polygon_boundary).area > 0):
					valid = True
			if(valid):
				valid_buildings.append(building)
		print("Selected", len(valid_buildings), "from a set of", len(annotations_data["polygons"]), "polygons")

		buffer_distance = BUFFER_DISTANCE / ORTHO_GSD[target_geotif]

		valid_roadlines = []
		for road in annotations_data["road_lines"]:
			verts = []
			valid = False
			for point in road["pixels"]:
				# Let's Adjust the roads first ...
				best_adjustment = match_vertex_to_adjustment(adjustments, point)
				if best_adjustment is None:
					best_adjustment_options = []
				else:
					best_adjustment_options = [best_adjustment]
			
				x_adj, y_adj = apply_adjustments(best_adjustment_options, point["x"], point["y"])
				x, y = rasterio.transform.xy(transform, y_adj, x_adj)
				lon, lat = coord_transformer_to_latlon.transform(x, y)

				verts.append([lon,lat])
			adjusted_road = LineString(verts)

			# Determine if the adjusted road is within the the polyogn boundary
			valid = False
			for polygon_boundary in polygon_boundaries:
				# Intersect with adjusted roadline
				intersected_roadline = shapely.intersection(adjusted_road, polygon_boundary)
				
				if(intersected_roadline.length > 0):
					valid = True
			if(valid):
					valid_roadlines.append(road)
					
					
						
		print("Selected", len(valid_roadlines), "from a set of", len(annotations_data["road_lines"]), "roadlines")
		json_output = {"road_lines": valid_roadlines, "polygons": valid_buildings}
		
		out_file = target_geotif + ".json"
		out_path = os.path.join(args.output_json_folder, out_file)
		f = open(out_path, "w")
		f.write(json.dumps(json_output, indent=4, sort_keys=True))
		f.close()

		trimmed_annotations_path_map[geotif_path] = out_path

	print("Generating Annotation Path Map...")
	f = open(args.trimmed_path_map, "w")
	f.write(json.dumps(trimmed_annotations_path_map))
	f.close()
	print("Done...")