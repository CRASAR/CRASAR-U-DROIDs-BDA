import geopandas as gpd
import os
import json
import argparse

from shapely.geometry import shape

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_folder', type=str, help='The folder of input geojsons to be processed.')
	parser.add_argument('--output_folder', type=str, help='The output folder where shapefiles will be stored.')
	args = parser.parse_args()

	for (root,dirs,files) in os.walk(args.input_folder, topdown=True): 
		for file in files:
			if file.endswith(".geo.tif.json"):
				print("Working file:", file)
				out_file = file.replace(".json", ".shp")
				out_path = os.path.join(args.output_folder, out_file)

				f = open(os.path.join(root, file), "r")
				data = json.loads(f.read())

				geom = [shape(i["geometry"]) for i in data]
				gdf = gpd.GeoDataFrame({'geometry':geom})
				gdf.to_file(out_path)
				print("Writing converted file to:", out_path)
				print("Done\n")