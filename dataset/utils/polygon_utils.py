import cv2
import random
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from dataset.constants import BDA_DAMAGE_CLASSES, LAT_LON_CRS

def get_polygons_from_mask(mask):
	imgray = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
	contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	polygons = []

	for object in contours:
		coords = []
		
		for point in object:
			coords.append((int(point[0][0]), int(point[0][1])))

		polygons.append(coords)
	return polygons

def get_mask_from_polygons(im_draw, polygons, color="white"):
	for shape in polygons:
		if(len(shape) > 1):
			im_draw.polygon(shape, fill=color)

def get_polygon_id():
	return "%032x" % random.getrandbits(128)

