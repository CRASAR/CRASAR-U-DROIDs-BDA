import os
import json
import argparse

import rasterio
import geopandas as gpd
from pyproj import Transformer

from dataset.constants import LAT_LON_CRS
from dataset.utils.make_empty_adjustment import generate_empty_adjustments_data

def convert_geojson_adjustment_to_pixels(geojson_adjustments, geotiff_data):
    result = []
    coord_transformer_to_crs = Transformer.from_crs(LAT_LON_CRS, geotiff_data.crs.to_string())
    for geojson_adjustment in geojson_adjustments:
        pixel_coord_line = []
        adj_coords = list(zip(*geojson_adjustment.coords.xy))

        for lat, lon in [adj_coords[0], adj_coords[1]]:

            x_source, y_source = coord_transformer_to_crs.transform(lon, lat)

            x_p, y_p = rasterio.transform.rowcol(geotiff_data.transform, x_source, y_source, op=lambda x:x)
            pixel_coord_line.append([x_p, y_p])

        result.append(pixel_coord_line)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='geojson_lines_to_adjustments', description='This program generates adjustments from a geojson file..')
    parser.add_argument('--geotif_folder', type=str, help="The geotif for which we want to generate a bank adjustments file")
    parser.add_argument('--geojson_adjustments_folder', type=str, help="The geojson for which we want to generate an adjustments file")
    parser.add_argument('--output_adjustments_folder', type=str, help='The path where the adjustments should be written.')
    parser.add_argument('--generate_empty_adjustment_on_failed_match', action="store_true", help="When set, an empty adjustment will be generated if \
        a match cannot be found with a geojson adjustments file.")
    args = parser.parse_args()

    adjustments_data = {}
    for filename in os.listdir(args.geojson_adjustments_folder):
        file_path = os.path.join(args.geojson_adjustments_folder, filename)
        print("\t", file_path)
        if filename.endswith(".geojson"):
            imagery_filename = filename.replace(".geojson", "")
            with open(file_path, "r") as f:
                data = gpd.read_file(f.read())
                adjustments_data[imagery_filename] = []
                for _, row in data.iterrows():
                    adjustments_data[imagery_filename].append(row.geometry)

    print("Loading the target geotiffs from:", args.geotif_folder)

    for filename in os.listdir(args.geotif_folder):
        if filename.lower().endswith("tif") or filename.lower().endswith("tiff"):
            output_path = os.path.join(args.output_adjustments_folder, str(filename) + ".json")

            if filename in adjustments_data:
                print("\tMatched", filename, "with corresponding adjustments data")

                geotif_file_path = os.path.join(args.geotif_folder, filename)

                input_geotiff_data = rasterio.open(geotif_file_path, "r")

                adjustments = convert_geojson_adjustment_to_pixels(adjustments_data[filename], input_geotiff_data)

                with open(output_path, "w") as adjustments_out:
                    adjustments_out.write(json.dumps(adjustments))

                print("\t\tFound", len(adjustments), "adjustments")
            else:
                print("\tFailed to match", filename, "with corresponding adjustments data")
                if args.generate_empty_adjustment_on_failed_match:
                    print("\t\tGenerating blank adjustments file")
                    with open(output_path, "w") as adjustments_out:
                        adjustments_out.write(json.dumps(generate_empty_adjustments_data()))

                else:
                    print("\t\tSkipping adjustments data for this file")
