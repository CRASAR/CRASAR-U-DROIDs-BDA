import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from utils.adjustment_utils import match_vertex_to_adjustment, apply_adjustments
from dataset.constants import ORTHO_GSD

from shapely import Point

import argparse
import json
import pandas as pd
import rasterio
import math
import scipy

NUMBER_ADJ = "Annotations"
ROADS_VERTEX = "Road Vertices"

AVG_ANGLE = "Avg Angle"
AVG_DISTANCE = "Avg Distance"
AVG_GSD_DISTANCE = "Avg GSD Distance"
VAR_ANGLE = "Var Angle"
VAR_DISTANCE = "Var Distance"
VAR_GSD_DISTANCE = "Var GSD Distance"

TOTAL_DISTANCE = "Total Distance"
TOTAL_GSD_DISTANCE = "Total GSD Distance"
TOTAL_ANGLE = "Total Angle"

ANGLES = "Angles"
PIXEL_DIST = "Pixel Distance"
GSD_DIST = "GSD Distance"

STAT_FIELDS = [ROADS_VERTEX, NUMBER_ADJ, AVG_ANGLE, AVG_GSD_DISTANCE, AVG_DISTANCE, VAR_ANGLE, VAR_DISTANCE, VAR_GSD_DISTANCE, TOTAL_DISTANCE, TOTAL_ANGLE, TOTAL_GSD_DISTANCE]

def compute_adj_stats(geotif_name, geotif, adj, rda_data): 

    ortho_metrics= {NUMBER_ADJ: 0, ROADS_VERTEX: 0, AVG_ANGLE: 0, AVG_DISTANCE: 0, AVG_GSD_DISTANCE: 0, VAR_ANGLE: 0, VAR_DISTANCE: 0, VAR_GSD_DISTANCE:0, TOTAL_ANGLE: 0, TOTAL_DISTANCE: 0, TOTAL_GSD_DISTANCE:0, ANGLES: [], PIXEL_DIST: [], GSD_DIST: []}      
    
    print("Loading the ortho " + geotif + " ...")
    input_geotiff_data = rasterio.open(geotif, "r")

    ortho_metrics[NUMBER_ADJ] = len(adj)

    for line in rda_data["road_lines"]:
        # verts_conv = []
        old_verts = []
        verts = []
        for point in line["pixels"]:
            ortho_metrics[ROADS_VERTEX] += 1
            best_adjustment = match_vertex_to_adjustment(adj, point)
            if best_adjustment is None:
                best_adjustment_options = []
            else:
                best_adjustment_options = [best_adjustment]

            old_verts.append((point["x"], point["y"]))
            old_point = Point(point["x"], point["y"])
            x_adj, y_adj = apply_adjustments(best_adjustment_options, point["x"], point["y"])
            verts.append((x_adj, y_adj))
            new_point = Point(x_adj, y_adj)

            distance_between_points = math.dist([old_point.x, old_point.y], [new_point.x, new_point.y])
            ortho_metrics[PIXEL_DIST].append(distance_between_points)
            ortho_metrics[GSD_DIST].append(distance_between_points*ORTHO_GSD[geotif_name])

            angle_between_points = (math.degrees(math.atan2(float((new_point.y-old_point.y)), float((new_point.x-old_point.x)))) + 360) % 360 # Ensure that Angle is between 0 - 360
            ortho_metrics[ANGLES].append(angle_between_points)


        ortho_metrics[TOTAL_ANGLE] = sum(ortho_metrics[ANGLES])
        ortho_metrics[TOTAL_DISTANCE] = sum(ortho_metrics[PIXEL_DIST])
        ortho_metrics[TOTAL_GSD_DISTANCE] = sum(ortho_metrics[PIXEL_DIST]) * ORTHO_GSD[geotif_name]

        ortho_metrics[AVG_ANGLE] = scipy.mean(ortho_metrics[ANGLES])
        ortho_metrics[AVG_DISTANCE] = scipy.mean(ortho_metrics[PIXEL_DIST])
        ortho_metrics[AVG_GSD_DISTANCE] = scipy.mean(ortho_metrics[PIXEL_DIST]) * ORTHO_GSD[geotif_name]

        ortho_metrics[VAR_ANGLE] = scipy.stats.variation(ortho_metrics[ANGLES])
        ortho_metrics[VAR_DISTANCE] = scipy.stats.variation(ortho_metrics[PIXEL_DIST])
        ortho_metrics[VAR_GSD_DISTANCE] = scipy.stats.variation(ortho_metrics[PIXEL_DIST]) * ORTHO_GSD[geotif_name]


    return ortho_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compute_ADJ_stats", description="This program computes the adjustments statistics for the dataset.")
    parser.add_argument('--adj_annotations_map', type=str, help="The path to the map for ADJ annotations.")
    parser.add_argument('--rda_annotations_map', type=str, help="The path to the map for RDA annotations.")
    parser.add_argument('--geotif_map', type=str, help="The path to the map for Orthos.")
    parser.add_argument('--output_stats_file_path', type=str, help="The path to the output statistics file.")
    args = parser.parse_args()

    print("Loading Geotif File Path Mapping...")
    f = open(args.geotif_map, "r")
    geotif_path_map = json.loads(f.read())
    f.close()

    print("Loading RDA Annotation File Path Mapping...")
    f = open(args.rda_annotations_map, "r")
    rda_path_map = json.loads(f.read())
    f.close()

    print("Loading ADJ Annotation File Path Mapping...")
    f = open(args.adj_annotations_map, "r")
    adj_path_map = json.loads(f.read())
    f.close()
    
    print("Found " + str(len(geotif_path_map.keys())) + " Orthos...")

    adj_metrics = {}
    total_annotations = 0
    
    for geotif, geotif_path in geotif_path_map.items():
        
        try:
            print("Loading ADJ annotations from " + adj_path_map[geotif_path] + "...")
            f = open(adj_path_map[geotif_path], "r")
            adj_data = json.loads(f.read())
            f.close()

            if(len(adj_data) == 0):
                print("WARNING: No adjustments found!")

            print("Loading RDA annotations from " + rda_path_map[geotif_path] + "...")
            f = open(rda_path_map[geotif_path], "r")
            rda_data = json.loads(f.read())
            f.close()

            print("Computing ADJ stats for ortho...")
            adj_metrics[geotif] = compute_adj_stats(geotif, geotif_path, adj_data, rda_data)
            total_annotations += adj_metrics[geotif][NUMBER_ADJ]
            print("Done.")

        except KeyError:
            print("Could not find annotations for " + geotif)
            print("Skipping ortho... \n")

    # Save the ADJ stats to the path
    print("Saving Annotations Stats to csv file located at: " + args.output_stats_file_path)
    stats_df = pd.DataFrame.from_dict(adj_metrics, orient="index")[STAT_FIELDS]
    stats_df.to_csv(args.output_stats_file_path)
    print("Done.")

    print("Number of total adjustments", total_annotations)