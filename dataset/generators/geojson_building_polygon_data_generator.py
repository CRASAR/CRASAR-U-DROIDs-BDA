import geopandas as gpd
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon

def load_buildings_from_geojson(path):
	return gpd.read_file(path)

def get_geojson_buildings_for_bounding_box(bbox, building_polygons_gpd_dataframe):
	polygons = building_polygons_gpd_dataframe.clip(bbox)

	buildings = []
	for index, building in polygons.iterrows():

		polygons = []

		if(type(building["geometry"]) == MultiPolygon):
			polygons = list(building["geometry"].geoms)
		elif(type(building["geometry"]) == Polygon):
			polygons = [building["geometry"]]

		for polygon in polygons:
			coords = list(zip(*polygon.exterior.coords.xy))
			labeled_cords = [{"lat":x, "lon":y} for y, x in coords]
			buildings.append({"geometry":labeled_cords})

	return {"buildings":buildings}