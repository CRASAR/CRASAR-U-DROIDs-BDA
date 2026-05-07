import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from shapely.geometry import Polygon, shape
from alive_progress import alive_bar

import json
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='fuse_path_maps', description='This This is a utility script that combines path maps.')
	parser.add_argument('--path_map_1', type=str, help='The first path map to be fused')
	parser.add_argument('--path_map_2', type=str, help='The second path map to be fused')
	parser.add_argument('--output_path_map', type=str, help="The path to the output path map")
	args = parser.parse_args()

	print("Loading path map data...")
	path_map_1 = json.load(open(args.path_map_1))
	path_map_2 = json.load(open(args.path_map_2))

	print("Fusing path maps...")
	result = {}
	for k,v in path_map_1.items():
		result[k] = v
	for k,v in path_map_2.items():
		result[k] = v

	print("Writing Output Path Map...")
	f = open(args.output_path_map, "w")
	f.write(json.dumps(result))
	f.close()
	print("Done...")