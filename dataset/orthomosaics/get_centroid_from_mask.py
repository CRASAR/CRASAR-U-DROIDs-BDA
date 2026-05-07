import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
orthomosaics_path = os.path.dirname(current)
dataset_path = os.path.dirname(orthomosaics_path)
sys.path.append(orthomosaics_path)
sys.path.append(dataset_path)

from shapely.geometry import shape

import pandas
import argparse
import json
import shapely
import pandas as pd

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_folder', type=str, help='The folder of input geotifs to be processed.')
	parser.add_argument('--output_file', type=str, help='The output file.')
	args = parser.parse_args()

	centroids = {}

	for (root,dirs,files) in os.walk(args.input_folder, topdown=True): 
		for file in files:
			if file.endswith(".geo.tif.json"):
				boundary_file = os.path.join(root, file)
				print("Getting centroid from:", boundary_file)
				f = open(boundary_file, "r")
				boundary_data = json.load(f)
				f.close()
				polygon_boundaries = []
				for i in range(0, len(boundary_data)):
					polygon_boundaries.append(shape(boundary_data[i]["geometry"]))
				polygon_boundary = shapely.MultiPolygon(polygon_boundaries)
				centroid = polygon_boundary.centroid
				centroids[file] = {"lon":centroid.x, "lat":centroid.y}

	d = pd.DataFrame(centroids).transpose()
	d.to_csv(args.output_file)
				
