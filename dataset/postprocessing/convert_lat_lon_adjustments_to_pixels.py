import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.constants import BDA_DAMAGE_CLASSES, LAT_LON_CRS
from dataset.utils.adjustment_utils import apply_adjustments, match_polygon_to_adjustment

from PIL import Image, ImageDraw
from rasterio import CRS
from shapely.geometry import Polygon, Point
from shapely import distance
from pyproj import Transformer
from alive_progress import alive_bar
from collections import defaultdict

import json
import numpy as np
import rasterio
import os
import pandas as pd
import argparse
import math

if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='convert_lat_lon_adjustments_to_pixels', description='This program converts adjustments lines that have been generated in lat/lon to xy pixels coords..')
	parser.add_argument('--input_lat_lon_adj_file', type=str, help="The path to the file containing the lat lon adjustments.")
	parser.add_argument('--input_orthomosaic', type=str, help="The path to the orthomosaic who's pixles will be used to compute the pixel coordinates.")
	parser.add_argument('--output_xy_adj_file', type=str, help="The path to the output pixel adjustment file.")
	args = parser.parse_args()

	f = open(args.input_lat_lon_adj_file, "r")
	lat_lon_adjs = json.loads(f.read())
	f.close()

	print("Loading the ortho " + args.input_orthomosaic + " ...")
	input_geotiff_data = rasterio.open(args.input_orthomosaic, "r")

	result = []

	transformer = Transformer.from_crs(LAT_LON_CRS, input_geotiff_data.crs.to_string())

	print("Found crs", input_geotiff_data.crs.to_string(), "in orthomosaic", args.input_orthomosaic)

	for adj in lat_lon_adjs:
		result.append([])
		for lon, lat in adj:
			x_source, y_source = transformer.transform(lat, lon)
			x, y = rasterio.transform.rowcol(input_geotiff_data.transform, x_source, y_source)
			result[-1].append([x, y])
			print("lat", lat, "lon", lon, "x_source", x_source, "y_source", y_source, "x/y", x, y)

	f = open(args.output_xy_adj_file, "w")
	f.write(json.dumps(result))
	f.close()