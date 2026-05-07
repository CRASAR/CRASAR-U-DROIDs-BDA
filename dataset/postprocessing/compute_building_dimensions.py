import os
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
from shapely import Polygon

from dataset.constants import ORTHO_GSD

HEIGHT = "Height"
WIDTH = "Width"
AREA = "Area"

AVERAGE_HEIGHT = "Avg. Height"
AVERAGE_WIDTH = "Avg. Width"
AVERAGE_AREA = "Avg. Area"
AVERAGE_HEIGHT_SPATIAL = "Avg. Height (cm)"
AVERAGE_WIDTH_SPATIAL = "Avg. Width (cm)"
AVERAGE_AREA_SPATIAL = "Avg. Area (cm)"


def compute_dimensions(data, ortho_title):
    ortho_stats = {HEIGHT: [], WIDTH: [], AREA: []}
    for p in data:
        pixel_coords = [(point["x"], point["y"]) for point in p["pixels"]]

        pixel_polygon = Polygon(pixel_coords)

        pixel_bounds = pixel_polygon.bounds
        ortho_stats[WIDTH].append(pixel_bounds[2] - pixel_bounds[0])
        ortho_stats[HEIGHT].append(pixel_bounds[3] - pixel_bounds[1])
        ortho_stats[AREA].append(pixel_polygon.area)

    avg_height = np.average(ortho_stats[HEIGHT])
    avg_width = np.average(ortho_stats[WIDTH])
    avg_area = np.average(ortho_stats[AREA])

    return {
        AVERAGE_HEIGHT: avg_height,
        AVERAGE_WIDTH: avg_width,
        AVERAGE_AREA: avg_area,
        AVERAGE_HEIGHT_SPATIAL: avg_height * ORTHO_GSD[ortho_title],
        AVERAGE_WIDTH_SPATIAL: avg_width * ORTHO_GSD[ortho_title],
        AVERAGE_AREA_SPATIAL: avg_area * ORTHO_GSD[ortho_title],
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="compute_building_dimensions",
        description="This program computes the building dimensions per ortho.",
    )
    parser.add_argument(
        "--annotations_path_map",
        type=str,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--output_stats_file_path",
        type=str,
        help="The path to the output statistics file.",
    )
    args = parser.parse_args()

    with open(args.annotations_path_map, "r") as f:
        annotations_path_map = json.load(f)

    ortho_label_stats = {}

    for geotif_path, annotation_path in annotations_path_map.items():

        ortho_local_title = os.path.split(geotif_path)[1]

        # Load the annotations
        print("Loading the BDA annotations from:", annotation_path)
        with open(annotation_path, "r") as f:
            annotations_data = json.load(f)

        with rasterio.open(geotif_path) as geotiff:
            transform = geotiff.transform

        building_stats = compute_dimensions(annotations_data, ortho_local_title)
        ortho_label_stats[ortho_local_title] = building_stats

    stats = pd.DataFrame(ortho_label_stats)
    stats.transpose().to_csv(args.output_stats_file_path)

    print(f"Building dimension stats saved to {args.output_stats_file_path}")
