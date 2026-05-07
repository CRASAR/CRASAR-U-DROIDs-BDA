import geopandas as gpd
import json
import pandas as pd
from geopandas import GeoDataFrame
import argparse
from alive_progress import alive_bar


def filter_geojson(input_file, bbox_file, output_file):
    print("Loading data from", input_file)
    with open(input_file, 'r') as read_file:
        all_features = json.load(read_file)

    file = open(bbox_file).readline()
    bbox = json.loads(file)

    print("Trimming...")
    trimmed_data = {"type": "FeatureCollection", "features": []}
    with alive_bar(total=len(all_features['features'])) as bar:
        for feature in all_features['features']:
            coordinates = feature['geometry']['coordinates'][0]
            for lon, lat in coordinates:
                valid_lat = False
                valid_lon = False
                if float(bbox[1]) <= float(lon) <= float(bbox[3]):
                    valid_lon = True
                if float(bbox[0]) <= float(lat) <= float(bbox[2]):
                    valid_lat = True
                if valid_lon and valid_lat:
                    trimmed_data['features'].append(feature)
            bar()

    trimmed_gdf = GeoDataFrame.from_features(trimmed_data)
    trimmed_gdf.to_file(output_file, "GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_geojson', type=str, help='name or path to input geojson')
    parser.add_argument('--bbox_file_path', type=str, help='name or path of file of bbox')
    parser.add_argument('--output_file_path', type=str, help='name or path of the output file containing the trimmed geojson data')
    args = parser.parse_args()
    filter_geojson(args.input_geojson, args.bbox_file_path, args.output_file_path)
