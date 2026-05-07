import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.constants import BDA_DAMAGE_CLASSES, LAT_LON_CRS
from dataset.utils.adjustment_utils import (
    apply_adjustments,
    match_polygon_to_adjustment,
)

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

    parser = argparse.ArgumentParser(
        prog="integrate_BDA_spot_checks",
        description="This program combines the spot checks that were manually entered and udpates the annotations file.",
    )
    parser.add_argument(
        "--spot_checks_folder", type=str, help="The path to the spot checks folder."
    )
    parser.add_argument(
        "--annotations_path_map",
        type=str,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--adjustments_path_map",
        type=str,
        help="The path to the adjustments file path map.",
    )
    parser.add_argument(
        "--geotif_path_map", type=str, help="The path to the imagery file path map."
    )
    parser.add_argument(
        "--spot_check_path_map",
        type=str,
        help="The path to the output spot checked annotations file path map.",
    )
    parser.add_argument(
        "--output_annotations_folder",
        type=str,
        help="The patht to the output annotations folder with the spot checks applied.",
    )
    parser.add_argument(
        "--suas",
        help="If source is drone, naming convention is different.",
        action="store_true",
    )
    args = parser.parse_args()

    annotations_path_map = json.load(open(args.annotations_path_map))
    adjustments_path_map = json.load(open(args.adjustments_path_map))
    geotif_path_map = json.load(open(args.geotif_path_map))

    matched_count = 0
    failed_count = 0

    os.makedirs(args.output_annotations_folder, exist_ok=True)

    for geotif, annotation_path in annotations_path_map.items():
        geotif_path = geotif
        target_geotif = geotif
        if not args.suas:
            geotif_path = geotif_path_map[geotif]
        else:
            target_geotif = os.path.split(geotif)[1]

        if geotif_path in adjustments_path_map.keys():
            adjustments_path = adjustments_path_map[geotif_path]
        else:
            adjustments_path = adjustments_path_map[target_geotif]

        # Load the annotations
        print("Loading the BDA annotations from:", annotation_path)
        f = open(annotation_path, "r")
        annotations_data = json.loads(f.read())
        f.close()

        print("Loading the ADJ annotations from:", adjustments_path)
        f = open(adjustments_path, "r")
        adjustments_data = json.loads(f.read())
        f.close()

        print("Loading the target geotiff metadata from:", geotif_path)
        input_geotiff_data = rasterio.open(geotif_path, "r")

        coord_system = str(input_geotiff_data.crs)
        print("Generating transformer between:", coord_system, "and", LAT_LON_CRS)
        coord_transformer = Transformer.from_crs(coord_system, LAT_LON_CRS)

        # Load spot checks
        spot_checks_path = os.path.join(args.spot_checks_folder, target_geotif + ".txt")
        if os.path.exists(spot_checks_path):
            print("Loading the BDA spotchecks from:", spot_checks_path)
            f = open(spot_checks_path, "r")
            d = f.read()
            f.close()

            print("Applying spot checks")
            # Integrate the spot checks...
            spot_check_data = []
            for line in d.replace("\r", "").split("\n"):
                try:
                    coord, label = line.split("->")
                    label = label.lstrip(" ").rstrip(" ").lower()

                    lat, lon = coord.replace(" ", "").split(",")
                    target_point = Point(lat, lon)
                    adjusted_distances = []
                    unadjusted_distances = []
                    if label in BDA_DAMAGE_CLASSES:
                        matched = False
                        for i in range(0, len(annotations_data)):

                            # Find the best adjustment for the polygon we are looking at...
                            best_adjustment = match_polygon_to_adjustment(
                                adjustments_data, annotations_data[i]["pixels"]
                            )

                            if best_adjustment is None:
                                best_adjustment_options = []
                            else:
                                best_adjustment_options = [best_adjustment]

                            # Adjust the points...
                            adjusted_points = []
                            for point in annotations_data[i]["pixels"]:
                                x_adj, y_adj = apply_adjustments(
                                    best_adjustment_options, point["x"], point["y"]
                                )
                                # Flip the y and x axis to align the data correctly in the coordinate space
                                x_source, y_source = rasterio.transform.xy(
                                    input_geotiff_data.transform, y_adj, x_adj
                                )
                                lat, lon = coord_transformer.transform(
                                    x_source, y_source
                                )
                                adjusted_points.append((lat, lon))

                            # Generate the polygon, based on the adjusted coords...
                            p_adjusted = Polygon(adjusted_points)
                            p_unadjusted = Polygon(
                                [
                                    (c["lat"], c["lon"])
                                    for c in annotations_data[i][LAT_LON_CRS]
                                ]
                            )
                            adjusted_distances.append(
                                distance(p_adjusted, target_point)
                            )
                            unadjusted_distances.append(
                                distance(p_unadjusted, target_point)
                            )

                            if p_adjusted.contains(
                                target_point
                            ) and p_unadjusted.contains(target_point):
                                annotations_data[i]["label"] = label
                                matched = True
                                matched_count += 1

                        if not matched:
                            adj_hit = False
                            unadj_hit = False
                            min_unadj = np.argmin(unadjusted_distances)
                            min_adj = np.argmin(adjusted_distances)
                            if min_adj == min_unadj and (
                                adjusted_distances[min_adj] < 10e-6
                                or unadjusted_distances[min_unadj] < 10e-6
                            ):
                                annotations_data[min_adj]["label"] = label
                                matched_count += 1
                                matched = True
                            elif min_adj != min_unadj:
                                if adjusted_distances[min_adj] == 0:
                                    adj_hit = True
                                if unadjusted_distances[min_unadj] == 0:
                                    unadj_hit = True
                                if adj_hit and not unadj_hit:
                                    annotations_data[min_adj]["label"] = label
                                    matched_count += 1
                                    matched = True
                                elif unadj_hit and not adj_hit:
                                    annotations_data[min_unadj]["label"] = label
                                    matched_count += 1
                                    matched = True

                            if not matched:
                                print(
                                    "WARNING: Was not able to match",
                                    line,
                                    "with polygon...",
                                )
                                print(
                                    "\t\tPolygon with the closest distance was: (adj |",
                                    adjusted_distances[min_adj],
                                    "), (unadj |",
                                    unadjusted_distances[min_unadj],
                                    ") deg away.",
                                )

                                print(annotations_data[min_unadj][LAT_LON_CRS])
                                print(annotations_data[min_adj][LAT_LON_CRS])

                                if adj_hit == True:
                                    print("\t\tMatching to Adjusted")
                                    annotations_data[min_adj]["label"] = label
                                    matched_count += 1
                                    matched = True
                                else:
                                    print("\t\tMatching Failed")
                                    failed_count += 1

                    else:
                        print(
                            "Skipping",
                            line,
                            "because",
                            label,
                            "isn't in",
                            BDA_DAMAGE_CLASSES,
                        )
                except ValueError as e:
                    print(
                        'Could not parse line"',
                        line,
                        '"in',
                        spot_checks_path,
                        " got error",
                        e,
                    )

            output_path = os.path.join(
                args.output_annotations_folder, target_geotif + ".json"
            )
            print("Saving spot checked annotations to:", output_path)
            f = open(output_path, "w")
            f.write(json.dumps(annotations_data))
            f.close()
            annotations_path_map[geotif] = output_path
        else:
            print("Skipping", spot_checks_path, "because it does not exist")

    print("Writing output spot checked annotations path map")
    f = open(args.spot_check_path_map, "w")
    f.write(json.dumps(annotations_path_map))
    f.close()
    print("Done")

    print(
        "Matched",
        matched_count,
        "Failed",
        failed_count,
        "Ratio",
        matched_count / (failed_count + matched_count),
    )
