import json
import argparse

from multiprocessing import Pool

import shapely

from shapely import LineString
from geopandas import GeoDataFrame
from geojson_length import calculate_distance, Unit

def trim_task(feature, lat, lon, radius_miles):
    lon_coords, lat_coords = feature['geometry']['coordinates'][0][0]

    ls_poly_r_geojson={"geometry":json.loads(shapely.to_geojson(LineString([[lon, lat], [lon_coords, lat_coords]])))}
    dist = calculate_distance(ls_poly_r_geojson, Unit.miles)

    if dist < radius_miles:
        return feature
    return None

def filter_geojson(input_file, lat, lon, radius_miles, output_file, processes=6):
    print("Loading data from", input_file)
    with open(input_file, 'r') as read_file:
        all_features = json.load(read_file)

    print("Initializing process pool")
    with Pool(processes) as p:

        print("Trimming...")
        trimmed_data = {"type": "FeatureCollection", "features": []}
        tasks = []
        for feat in all_features["features"]:
            tasks.append([feat, lat, lon, radius_miles])
        results = p.starmap(trim_task, tasks)
        trimmed_data["features"] = [i for i in results if not i is None]

        print("Included", len(trimmed_data['features']), "building polygons")
        print("Excluded", len(all_features['features'])-len(trimmed_data['features']), "building polygons")

        print("Formatting output file...")
        trimmed_gdf = GeoDataFrame.from_features(trimmed_data)

        if len(trimmed_gdf) > 0:
            print("Writing output file...")
            trimmed_gdf.to_file(output_file, "GeoJSON")
        else:
            print("Found No Building Polygons within Radius. No output will be saved.")

        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_geojson', type=str, help='name or path to input geojson')
    parser.add_argument('--lat', type=str, help='The latitude of the inclusion circle radius.')
    parser.add_argument('--lon', type=str, help='The longitude of the inclusion circle radius.')
    parser.add_argument('--radius_miles', type=float, help='The inclusion radius in miles.')
    parser.add_argument('--output_file_path', type=str, help='name or path of the output file containing the trimmed geojson data')
    parser.add_argument('--processes', type=int, help='The number of processes used.', default=6)
    args = parser.parse_args()

    filter_geojson(args.input_geojson, args.lat, args.lon, args.radius_miles, args.output_file_path, args.processes)
