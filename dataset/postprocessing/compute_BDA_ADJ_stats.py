import os
import argparse
import json
import datetime
import math
import rasterio
import pandas as pd
import scipy
import scipy.stats
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from shapely.geometry import Polygon

from dataset.utils.adjustment_utils import (
    apply_adjustments,
    match_polygon_to_adjustment,
)
from dataset.constants import (
    ORTHO_DATETIME,
    ORTHO_GSD,
    EVENTS,
    ORTHO_EVENT,
    HURRICANE_HARVEY,
    HURRICANE_IAN,
    HURRICANE_IDA,
    HURRICANE_IDALIA,
    HURRICANE_LAURA,
    HURRICANE_MICHAEL,
    KILAUEA_VOLCANO,
    MAYFIELD_TORNADO,
    MUSSETT_BAYOU_FIRE,
)

ANGLES_FIELD = "Angles"
DISTANCES_FIELD = "Distances"
GSD_DISTANCES_FIELD = "GSD Distances"
IOUS_FIELD = "IoUs"
BBOX_IOUS_FIELD = "bbox IoUs"
AGGREGATION_FIELDS = [
    ANGLES_FIELD,
    DISTANCES_FIELD,
    GSD_DISTANCES_FIELD,
    IOUS_FIELD,
    BBOX_IOUS_FIELD,
]

BUILDINGS = "Buildings"
ANNOTATIONS = "Annotations"
NUMBER = "Number"
AVG_ANGLE = "Avg Angle"
AVG_DISTANCE = "Avg Distance"
AVG_GSD_DISTANCE = "Avg GSD Distance"
AVG_IOU = "Avg IoU"
AVG_BBOX_IOU = "Avg bbox IoU"
VAR_ANGLE = "Var Angle"
VAR_DISTANCE = "Var Distance"
VAR_GSD_DISTANCE = "Var GSD Distance"
VAR_IOU = "Var IoU"
VAR_BBOX_IOU = "Var bbox IoU"
TOTAL_IOU = "Total IoU"
TOTAL_BBOX_IOU = "Total bbox IoU"
TOTAL_DISTANCE = "Total Distance"
TOTAL_GSD_DISTANCE = "Total GSD Distance"
TOTAL_ANGLE = "Total Angle"
ANGLE_DISTANCE_SPEARMAN_R_STAISTIC = "Angle vs Distance Spearman R - staistic"
ANGLE_DISTANCE_SPEARMAN_R_PVALUE = "Angle vs Distance Spearman R - pvalue"
ANGLE_DISTANCE_PEARSON_R_STAISTIC = "Angle vs Distance Pearson R - staistic"
ANGLE_DISTANCE_PEARSON_R_PVALUE = "Angle vs Distance Pearson R - pvalue"

DX = "dx"
DY = "dy"
MEAN_ANGLE = "Mean Angle - New"
VAR_ANGLE = "Var Angle - New"
ANGLES_RAD = "Angle - Radians"
CIRCULAR_VAR = "cicular variance (ortho)"
ANGLES_MEAN = "angles mean (ortho)"

# This list describes the metrics which we will generate plots for, and the metrics that will appear in the CSV.
STAT_FIELDS = [
    BUILDINGS,
    ANNOTATIONS,
    NUMBER,
    AVG_ANGLE,
    AVG_GSD_DISTANCE,
    AVG_DISTANCE,
    AVG_IOU,
    AVG_BBOX_IOU,
    VAR_ANGLE,
    VAR_DISTANCE,
    VAR_GSD_DISTANCE,
    VAR_IOU,
    VAR_BBOX_IOU,
    TOTAL_IOU,
    TOTAL_BBOX_IOU,
    TOTAL_DISTANCE,
    TOTAL_ANGLE,
    TOTAL_GSD_DISTANCE,
    ANGLE_DISTANCE_SPEARMAN_R_STAISTIC,
    ANGLE_DISTANCE_SPEARMAN_R_PVALUE,
    ANGLE_DISTANCE_PEARSON_R_STAISTIC,
    ANGLE_DISTANCE_PEARSON_R_PVALUE,
    CIRCULAR_VAR,
    ANGLES_MEAN,
]

VIOLIN_GRID_ORTHOS = [
    "20211214-Mayfield.geo.tif",
    "10142018-MexicoBeach.geo.tif",
    "090302-Pecan-Grove-Levee.geo.tif",
    "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif",
    "2018-05-18-X4S-visible-CentralPark.geo.tif",
    "0827-A-01.geo.tif",
    "20230830-SteinhatcheeRiver.geo.tif",
    "20210831-LA-DIV-01.geo.tif",
    "1001-Summerlin-San-Carlos.geo.tif",
]

PLOT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def compute_circular_variance(angles):
    return 1 - np.sqrt(np.sum(np.cos(angles)) ** 2 + np.sum(np.sin(angles)) ** 2) / len(
        angles
    )


def compute_circular_variance(angles):
    return 1 - np.sqrt(np.sum(np.cos(angles)) ** 2 + np.sum(np.sin(angles)) ** 2) / len(
        angles
    )


def get_bbox(polygon):
    # Get Boundning Box around polygon
    return [
        (polygon.bounds[0], polygon.bounds[1]),
        (polygon.bounds[0], polygon.bounds[3]),
        (polygon.bounds[2], polygon.bounds[3]),
        (polygon.bounds[2], polygon.bounds[1]),
    ]


def compute_angle(dx, dy):
    # Compute angle between the two point
    angle = math.degrees(math.atan2(float(dy), float(dx)))
    return (angle + 360) % 360  # Ensure that Angle is between 0 - 360


def compute_iou(polygon1, polygon2):
    # Compute the intersection over union for two polygons
    if polygon1.intersects(polygon2):
        return (polygon1.intersection(polygon2).area) / (polygon1.union(polygon2).area)
    return 0


def compute_bbox_iou(polygon1, polygon2):
    # Get the bounding boxes for each poylgons
    bbox_poly1 = Polygon(get_bbox(polygon1))
    bbox_poly2 = Polygon(get_bbox(polygon2))

    # Compute the intersection over union between the two bounding boxes
    if bbox_poly1.intersects(bbox_poly2):
        return (bbox_poly1.intersection(bbox_poly2).area) / (
            bbox_poly1.union(bbox_poly2).area
        )
    return 0


def plot_distance_cdf(ortho_metrics, path, dist_field=DISTANCES_FIELD):
    all_distances = []
    for ortho_metric in ortho_metrics.values():
        all_distances.extend(ortho_metric[dist_field])

    # Create a figure and axes
    
    _, ax1 = plt.subplots()

    # Plot the histogram
    ax1.hist(
        all_distances,
        bins=30,
        density=False,
        alpha=0.5,
        label="Histogram",
        color="tab:blue",
    )

    ax2 = ax1.twinx()

    # Calculate and plot the CDF
    _, _, patches = ax2.hist(
        all_distances,
        bins=30,
        density=True,
        cumulative=True,
        histtype="step",
        color="tab:orange",
        label="CDF",
    )

    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Add labels and legend
    units = "Pixels" if dist_field == DISTANCES_FIELD else "Centimeters"
    ax1.set_xlabel("Distance in " + units, fontsize=11)
    ax1.set_ylabel("Count of Adjusted Buildings", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=10)
    ax2.set_ylabel("Cumulative Proportion of Adjustment Distances", fontsize=11)
    ax2.grid(alpha=0.2)
    ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=10)
    ax1.set_title(
        "Histogram of Adjustment Distances in "
        + units
        + " (N="
        + str(len(all_distances))
        + ")",
        fontsize=15,
    )

    # Show the plot
    plt.savefig(path)

    print("Distance Quantiles in", units)
    quantiles_of_interest = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    qs = np.quantile(all_distances, quantiles_of_interest)

    print({q: v for q, v in zip(quantiles_of_interest, qs)})


def plot_iou_cdf(ortho_metrics, path):
    all_ious = []
    for ortho_metric in ortho_metrics.values():
        all_ious.extend(ortho_metric[IOUS_FIELD])

    # Create a figure and axes
    _, ax1 = plt.subplots()

    # Plot the histogram
    ax1.hist(
        all_ious, bins=30, density=False, alpha=0.5, label="Histogram", color="tab:blue"
    )

    ax2 = ax1.twinx()

    # Calculate and plot the CDF
    _, _, patches = ax2.hist(
        all_ious,
        bins=30,
        density=True,
        cumulative=True,
        histtype="step",
        color="black",
        label="CDF",
    )

    patches[0].set_xy(patches[0].get_xy()[:-1])

    # Add labels and legend
    ax1.set_xlabel("Building Polygon IoUs", fontsize=11)
    ax1.set_ylabel("Count of Adjusted Buildings", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=10)
    ax2.set_ylabel("Cumulative Proportion of Adjusted Building IoUs", fontsize=11)
    ax2.grid(alpha=0.2)
    ax2.tick_params(axis="y", labelcolor="black", labelsize=10)
    ax1.set_title(
        "Histogram of Adjusted Building IoUs (N=" + str(len(all_ious)) + ")",
        fontsize=15,
    )

    # Show the plot
    plt.savefig(path, bbox_inches="tight")

    print("IoU Quantiles")
    quantiles_of_interest = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    qs = np.quantile(all_ious, quantiles_of_interest)

    print({q: v for q, v in zip(quantiles_of_interest, qs)})


def plot_circular_histograms(ortho_metrics, keys, bins, path):
    # Create a GridSpec with 2 rows and 5 columns
    gs = gridspec.GridSpec(2, 5, figure=fig)

    for i, key in enumerate(keys):
        ortho_metric = ortho_metrics[key]
        angles = np.linspace(0, 2 * np.pi, bins, endpoint=False)
        width = 2 * np.pi / bins
        raw_values = ortho_metric[ANGLES_FIELD]
        values, _ = np.histogram(raw_values, bins, density=True)
        values_normed = [v / sum(values) for v in values]

        # Initialize Figure and Axis
        ax = fig.add_subplot(gs[0, i + 1], subplot_kw={"projection": "polar"})

        # Set limits for radial (y) axis. The negative lower bound creates the whole in the middle.
        ax.set_ylim(-0.1, 0.25)
        ax.set_theta_offset(np.pi / 2)
        # Remove all spines
        ax.set_frame_on(False)

        # Remove grid and tick marks
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        new_ticks = []
        for tick in ax.get_yticks():
            if tick >= 0:
                new_ticks.append(tick)
        ax.set_yticks(new_ticks)

        ax.bar(
            angles,
            values_normed,
            width=width,
            linewidth=2,
            color="#61a4b2",
            edgecolor="white",
        )
        ax.grid(zorder=0)
    plt.savefig(path, bbox_inches="tight")


def compute_adj_stats(name, current_geotif, adj, building):

    print("Loading the ortho " + current_geotif + " ...")
    input_geotiff_data = rasterio.open(current_geotif, "r")

    # Iterate through all buildings
    ortho_metrics = {
        BUILDINGS: len(building),
        ANNOTATIONS: len(adj),
        NUMBER: 0,
        AVG_ANGLE: 0,
        AVG_DISTANCE: 0,
        AVG_GSD_DISTANCE: 0,
        AVG_IOU: 0,
        AVG_BBOX_IOU: 0,
        VAR_ANGLE: 0,
        VAR_DISTANCE: 0,
        VAR_GSD_DISTANCE: 0,
        VAR_IOU: 0,
        VAR_BBOX_IOU: 0,
        TOTAL_IOU: 0,
        TOTAL_BBOX_IOU: 0,
        TOTAL_DISTANCE: 0,
        TOTAL_GSD_DISTANCE: 0,
        TOTAL_ANGLE: 0,
        ANGLE_DISTANCE_SPEARMAN_R_STAISTIC: 0,
        ANGLE_DISTANCE_SPEARMAN_R_PVALUE: 0,
        ANGLE_DISTANCE_PEARSON_R_STAISTIC: 0,
        ANGLE_DISTANCE_PEARSON_R_PVALUE: 0,
        ANGLES_FIELD: [],
        DISTANCES_FIELD: [],
        GSD_DISTANCES_FIELD: [],
        IOUS_FIELD: [],
        BBOX_IOUS_FIELD: [],
        DX: [],
        DY: [],
        ANGLES_RAD: [],
    }

    for polygon in building:
        if polygon["source"] != "custom":
            best_adjustment = match_polygon_to_adjustment(adj, polygon["pixels"])
            if best_adjustment is None:
                best_adjustment_options = []
            else:
                best_adjustment_options = [best_adjustment]
                ortho_metrics[NUMBER] += 1

            verts_conv = []
            old_verts = []
            verts = []
            for point in polygon["pixels"]:
                old_verts.append((point["x"], point["y"]))
                x_adj, y_adj = apply_adjustments(
                    best_adjustment_options, point["x"], point["y"]
                )
                verts.append((x_adj, y_adj))

                # Flip the y and x axis to align the data correctly in the coordinate space
                x_source, y_source = rasterio.transform.xy(
                    input_geotiff_data.transform, y_adj, x_adj
                )
                verts_conv.append((x_source, y_source))

            # Get the centriods of the unadjusted polygon and the adjusted polygon
            old_centroid = Polygon(old_verts).centroid
            new_centroid = Polygon(verts).centroid

            # Compute the intesection over union for the polygons and their bounding boxes ...
            ortho_metrics[IOUS_FIELD].append(
                compute_iou(Polygon(old_verts), Polygon(verts))
            )
            ortho_metrics[BBOX_IOUS_FIELD].append(
                compute_bbox_iou(Polygon(old_verts), Polygon(verts))
            )

            # Compute the angle with adjustments x, y based on the centroid of the building polygon...
            ortho_metrics[ANGLES_FIELD].append(
                compute_angle(
                    (new_centroid.x - old_centroid.x), (new_centroid.y - old_centroid.y)
                )
            )
            ortho_metrics[ANGLES_RAD].append(
                math.radians(
                    compute_angle(
                        (new_centroid.x - old_centroid.x),
                        (new_centroid.y - old_centroid.y),
                    )
                )
            )
            ortho_metrics[DX].append((new_centroid.x - old_centroid.x))
            ortho_metrics[DY].append((new_centroid.y - old_centroid.y))

            # Compute distance - dist between point on building to polygon..
            if best_adjustment is None:
                ortho_metrics[DISTANCES_FIELD].append(0)
                ortho_metrics[GSD_DISTANCES_FIELD].append(0)
            else:
                dist = math.dist(
                    [old_centroid.x, old_centroid.y], [new_centroid.x, new_centroid.y]
                )
                ortho_metrics[DISTANCES_FIELD].append(dist)
                ortho_metrics[GSD_DISTANCES_FIELD].append((dist * ORTHO_GSD[name]))

    # Compute the metrics ..
    if len(building) > 0:

        ortho_metrics[AVG_ANGLE] = np.average(ortho_metrics[ANGLES_FIELD])
        ortho_metrics[AVG_DISTANCE] = np.average(ortho_metrics[DISTANCES_FIELD])
        ortho_metrics[AVG_GSD_DISTANCE] = (
            np.average(ortho_metrics[DISTANCES_FIELD]) * ORTHO_GSD[name]
        )
        ortho_metrics[AVG_IOU] = np.average(ortho_metrics[IOUS_FIELD])
        ortho_metrics[AVG_BBOX_IOU] = np.average(ortho_metrics[BBOX_IOUS_FIELD])

        ortho_metrics[VAR_ANGLE] = scipy.stats.variation(ortho_metrics[ANGLES_FIELD])
        ortho_metrics[VAR_DISTANCE] = scipy.stats.variation(
            ortho_metrics[DISTANCES_FIELD]
        )
        ortho_metrics[VAR_GSD_DISTANCE] = (
            scipy.stats.variation(ortho_metrics[DISTANCES_FIELD]) * ORTHO_GSD[name]
        )
        ortho_metrics[VAR_IOU] = scipy.stats.variation(ortho_metrics[IOUS_FIELD])
        ortho_metrics[VAR_BBOX_IOU] = scipy.stats.variation(
            ortho_metrics[BBOX_IOUS_FIELD]
        )

        ortho_metrics[MEAN_ANGLE] = compute_angle(
            np.average(ortho_metrics[DX]), np.average(ortho_metrics[DY])
        )
        ortho_metrics[VAR_ANGLE] = compute_angle(
            scipy.stats.variation(ortho_metrics[DX]), np.average(ortho_metrics[DY])
        )

        if len(ortho_metrics[ANGLES_FIELD]) >= 2:
            spearmanr_object = scipy.stats.spearmanr(
                ortho_metrics[ANGLES_FIELD], ortho_metrics[DISTANCES_FIELD]
            )
            ortho_metrics[ANGLE_DISTANCE_SPEARMAN_R_STAISTIC] = (
                spearmanr_object.statistic
            )
            ortho_metrics[ANGLE_DISTANCE_SPEARMAN_R_PVALUE] = spearmanr_object.pvalue

            pearsonr_object = scipy.stats.pearsonr(
                ortho_metrics[ANGLES_FIELD], ortho_metrics[DISTANCES_FIELD]
            )
            ortho_metrics[ANGLE_DISTANCE_PEARSON_R_STAISTIC] = pearsonr_object.statistic
            ortho_metrics[ANGLE_DISTANCE_PEARSON_R_PVALUE] = pearsonr_object.pvalue

        ortho_metrics[TOTAL_ANGLE] = sum(ortho_metrics[ANGLES_FIELD])
        ortho_metrics[TOTAL_DISTANCE] = sum(ortho_metrics[DISTANCES_FIELD])
        ortho_metrics[TOTAL_GSD_DISTANCE] = (
            sum(ortho_metrics[DISTANCES_FIELD]) * ORTHO_GSD[name]
        )
        ortho_metrics[TOTAL_IOU] = sum(ortho_metrics[IOUS_FIELD])
        ortho_metrics[TOTAL_BBOX_IOU] = sum(ortho_metrics[BBOX_IOUS_FIELD])

        ortho_metrics[ANGLES_MEAN] = compute_angle(
            np.average(ortho_metrics[DX]), np.average(ortho_metrics[DY])
        )
        ortho_metrics[CIRCULAR_VAR] = compute_circular_variance(
            ortho_metrics[ANGLES_RAD]
        )

    else:
        print("Found Ortho with no building annotations!")

    return ortho_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compute_ADJ_stats",
        description="This program computes the adjustments statistics for the dataset.",
    )
    parser.add_argument(
        "--adj_annotations_map",
        type=str,
        help="The path to the map for ADJ annotations.",
    )
    parser.add_argument(
        "--bda_annotations_map",
        type=str,
        help="The path to the map for BDA annotations.",
    )
    parser.add_argument(
        "--geotif_map", type=str, help="The path to the map for Orthos."
    )
    parser.add_argument(
        "--output_stats_file_path",
        type=str,
        help="The path to the output statistics file.",
    )
    parser.add_argument("--plot_graphs", action="store_true")
    parser.add_argument(
        "--plot_folder",
        type=str,
        help="The directory for where the plots should be saved.",
    )
    args = parser.parse_args()

    print("Loading Geotif File Path Mapping...")

    with open(args.geotif_map, "r") as f:
        geotif_path_map = json.load(f.read())

    print("Loading BDA Annotation File Path Mapping...")
    with open(args.bda_annotations_map, "r") as f:
        bda_path_map = json.load(f.read())

    print("Loading ADJ Annotation File Path Mapping...")
    with open(args.adj_annotations_map, "r") as f:
        adj_path_map = json.load(f.read())

    print("Found " + str(len(geotif_path_map.keys())) + " Orthos...")

    adj_metrics = {}
    disaster_level_metrics = {
        event: {DX: [], DY: [], ANGLES_RAD: []} for event in EVENTS
    }

    for geotif, geotif_path in geotif_path_map.items():

        try:
            print("Loading ADJ annotations from " + adj_path_map[geotif_path] + "...")
            with open(adj_path_map[geotif_path], "r") as f:
                adj_data = json.load(f.read())

            if len(adj_data) == 0:
                print("WARNING: No adjustments found!")

            print("Loading BDA annotations from " + bda_path_map[geotif_path] + "...")
            with open(bda_path_map[geotif_path], "r") as f:
                bda_data = json.load(f.read())

            total_dx = []
            total_dy = []
            all_angles = []

            print("Computing ADJ stats for ortho...")
            adj_metrics[geotif] = compute_adj_stats(
                geotif, geotif_path, adj_data, bda_data
            )
            print("Done.")

            print("ADJ Stats for " + geotif + ":")
            print("\tAverage ADJ Angles: " + str(adj_metrics[geotif][AVG_ANGLE]))
            print("\tAverage ADJ Length: " + str(adj_metrics[geotif][AVG_DISTANCE]))
            print(
                "\tAverage ADJ Distance: " + str(adj_metrics[geotif][AVG_GSD_DISTANCE])
            )
            print("\tAverage ADJ IoU: " + str(adj_metrics[geotif][AVG_IOU]))
            print("\tAverage ADJ bbox IoU: " + str(adj_metrics[geotif][AVG_BBOX_IOU]))

            total_dx.extend(adj_metrics[geotif][DX])
            total_dy.extend(adj_metrics[geotif][DY])
            all_angles.extend(adj_metrics[geotif][ANGLES_RAD])

            disaster_level_metrics[ORTHO_EVENT[geotif]][DX].extend(
                adj_metrics[geotif][DX]
            )
            disaster_level_metrics[ORTHO_EVENT[geotif]][DY].extend(
                adj_metrics[geotif][DY]
            )
            disaster_level_metrics[ORTHO_EVENT[geotif]][ANGLES_RAD].extend(
                adj_metrics[geotif][ANGLES_RAD]
            )

        except KeyError:
            print("Could not find annotations for " + geotif)
            print("Skipping ortho... \n")

    # Save the ADJ stats to the path
    print(
        "Saving Annotations Stats to csv file located at: "
        + args.output_stats_file_path
    )
    stats_df = pd.DataFrame.from_dict(adj_metrics, orient="index")[STAT_FIELDS]
    stats_df.to_csv(args.output_stats_file_path)
    print("Done.")

    print("------------ STATS MEASURES FOR ADJUSTMENTS --------------\n")
    total_average_angle = compute_angle(np.average(total_dx), np.average(total_dy))
    total_variance_angle = compute_circular_variance(all_angles)
    print("Overall Average Angle ", total_average_angle)
    print("Overall Circular Variance Angle ", total_variance_angle)
    print("Hurricane Harvey ---")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[HURRICANE_HARVEY][DX]),
            np.average(disaster_level_metrics[HURRICANE_HARVEY][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[HURRICANE_HARVEY][ANGLES_RAD]),
    )
    print("Hurricane Ian ---")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[HURRICANE_IAN][DX]),
            np.average(disaster_level_metrics[HURRICANE_IAN][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[HURRICANE_IAN][ANGLES_RAD]),
    )
    print("Hurricane Ida ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[HURRICANE_IDA][DX]),
            np.average(disaster_level_metrics[HURRICANE_IDA][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[HURRICANE_IDA][ANGLES_RAD]),
    )
    print("Hurricane Idalia ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[HURRICANE_IDALIA][DX]),
            np.average(disaster_level_metrics[HURRICANE_IDALIA][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[HURRICANE_IDALIA][ANGLES_RAD]),
    )
    print("Hurricane Laura ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[HURRICANE_LAURA][DX]),
            np.average(disaster_level_metrics[HURRICANE_LAURA][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[HURRICANE_LAURA][ANGLES_RAD]),
    )
    print("Hurricane Michael ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[HURRICANE_MICHAEL][DX]),
            np.average(disaster_level_metrics[HURRICANE_MICHAEL][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(
            disaster_level_metrics[HURRICANE_MICHAEL][ANGLES_RAD]
        ),
    )
    print("Kilauea Eruption ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[KILAUEA_VOLCANO][DX]),
            np.average(disaster_level_metrics[KILAUEA_VOLCANO][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[KILAUEA_VOLCANO][ANGLES_RAD]),
    )
    print("Mayfield Tornado ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[MAYFIELD_TORNADO][DX]),
            np.average(disaster_level_metrics[MAYFIELD_TORNADO][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(disaster_level_metrics[MAYFIELD_TORNADO][ANGLES_RAD]),
    )
    print("Musset Bayou Fire ")
    print(
        "\tAverage Angle ",
        compute_angle(
            np.average(disaster_level_metrics[MUSSETT_BAYOU_FIRE][DX]),
            np.average(disaster_level_metrics[MUSSETT_BAYOU_FIRE][DY]),
        ),
    )
    print(
        "\tCircular Variance Angle ",
        compute_circular_variance(
            disaster_level_metrics[MUSSETT_BAYOU_FIRE][ANGLES_RAD]
        ),
    )
    print("------------------------------------------\n")

    if args.plot_graphs:
        print("Plotting Graphs...")
        os.makedirs(args.plot_folder, exist_ok=True)

        plot_distance_cdf(
            adj_metrics,
            os.path.join(args.plot_folder, "adjustment_px_distance_cdf.png"),
            DISTANCES_FIELD,
        )
        plot_distance_cdf(
            adj_metrics,
            os.path.join(args.plot_folder, "adjustment_cm_distance_cdf.png"),
            GSD_DISTANCES_FIELD,
        )
        plot_iou_cdf(
            adj_metrics, os.path.join(args.plot_folder, "adjustment_iou_cdf.png")
        )

        plot_circular_histograms(
            adj_metrics,
            VIOLIN_GRID_ORTHOS,
            45,
            os.path.join(args.plot_folder, "adjustment_angles_multiplot.png"),
        )

        # Plot violin 3x3 grids for angle, distance pixels and cm
        grid_folder = os.path.join(args.plot_folder, "3x3_violin_plots")
        os.makedirs(grid_folder, exist_ok=True)

        for idx, metric in enumerate(AGGREGATION_FIELDS):

            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
            axes = axes.flatten()

            # Get max value for y
            max_disty = 0
            max_gsddisty = 0
            for geotif in VIOLIN_GRID_ORTHOS:
                max_gsddisty = max(
                    max_gsddisty, np.max(adj_metrics[geotif][GSD_DISTANCES_FIELD])
                )
                max_disty = max(max_disty, np.max(adj_metrics[geotif][DISTANCES_FIELD]))

            # Create a violin plot for each ortho in the grid
            for ortho_idx, geotif in enumerate(VIOLIN_GRID_ORTHOS):
                ax = axes[ortho_idx]

                data = adj_metrics[geotif][metric]
                if len(data) > 0:
                    violin_plt = ax.violinplot(data, showmeans=True, showmedians=False)
                else:
                    violin_plt = ax.violinplot([0], showmeans=True, showmedians=False)

                for parts in violin_plt["bodies"]:
                    parts.set_facecolor(PLOT_COLORS[ortho_idx])
                    parts.set_edgecolor("black")

                ax.set_title(
                    f"{metric} of {geotif} (N = {adj_metrics[geotif][BUILDINGS]})",
                    wrap=True,
                )
                ax.set_ylim(bottom=0)

                if metric == ANGLES_FIELD:
                    ax.set_ylabel("Degrees")
                    ax.set_ylim(top=360)
                elif metric == DISTANCES_FIELD:
                    ax.set_ylabel("Pixels")
                    ax.set_ylim(top=max_disty)
                elif metric == GSD_DISTANCES_FIELD:
                    ax.set_ylabel("Centimeters")
                    ax.set_ylim(top=max_gsddisty)

                ax.set_xlabel("Adjustment")

            plt.tight_layout()
            grid_path = os.path.join(grid_folder, f"{metric}3x3_violin.png")
            print(f"Saving 3x3 grid for {metric} at " + grid_path)
            plt.savefig(grid_path)
            plt.clf()
            plt.close()

        # Plot adjacent violin plots
        for idx, metric in enumerate(AGGREGATION_FIELDS):
            fig, ax = plt.subplots(figsize=(30, 15))
            num_building = 0
            labels = []
            for i, geotif in enumerate(VIOLIN_GRID_ORTHOS):
                position = [VIOLIN_GRID_ORTHOS.index(geotif)]
                data = adj_metrics[geotif][metric]
                num_building += adj_metrics[geotif][NUMBER]
                if len(data) > 0:
                    violin_plt = ax.violinplot(
                        data,
                        showmeans=True,
                        showmedians=False,
                        positions=[VIOLIN_GRID_ORTHOS.index(geotif)],
                    )
                else:
                    violin_plt = ax.violinplot(
                        [0],
                        showmeans=True,
                        showmedians=False,
                        positions=[VIOLIN_GRID_ORTHOS.index(geotif)],
                    )

                for parts in violin_plt["bodies"]:
                    parts.set_facecolor(PLOT_COLORS[i])
                    parts.set_edgecolor("black")

                if len(data) > 0:
                    mean = np.mean(data)
                    data_max = np.amax(data)
                    data_min = np.amin(data)
                    ax.text(
                        position[0] + 0.2,
                        mean + 0.2,
                        f"{mean:.2f}",
                        color="black",
                        fontsize=14,
                        va="center",
                    )
                    ax.text(
                        position[0] + 0.2,
                        data_min + 0.3,
                        f"{data_min:.2f}",
                        color="black",
                        fontsize=14,
                        va="center",
                    )
                    ax.text(
                        position[0] + 0.2,
                        data_max + 0.2,
                        f"{data_max:.2f}",
                        color="black",
                        fontsize=14,
                        va="center",
                    )

                geotif_name = geotif[:-8]  # remove ".geo.tif"
                labels.append(geotif_name)

            labels = [
                "20211214-Mayfield",
                "10142018-MexicoBeach",
                "090302-Pecan-\nGrove-Levee",
                "05-08-2020-\nMussettBayouFire\n-SouthOf98-DelbertLn",
                "2018-05-18-X4S-visible\n-CentralPark",
                "0827-A-01",
                "20230830-SteinhatcheeRiver",
                "20210831-LA-DIV-01",
                "1001-Summerlin-San-Carlos",
            ]

            ax.set_title(f"{metric} (N={num_building})", fontsize=24)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="both", which="major", labelsize=18)

            if metric == ANGLES_FIELD:
                ax.set_ylabel("Degrees", fontsize=24)
                ax.set_ylim(top=360)
            elif metric == DISTANCES_FIELD:
                ax.set_ylabel("Pixels", fontsize=24)
            elif metric == GSD_DISTANCES_FIELD:
                ax.set_ylabel("Centimeters", fontsize=24)

            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=17)

            plt.tight_layout()
            grid_path = os.path.join(grid_folder, f"{metric}all_violin.png")
            print(f"Saving all plots for {metric} at " + grid_path)
            plt.savefig(grid_path)
            plt.clf()
            plt.close()

        orthos_sort_datetime = sorted(
            adj_metrics.keys(),
            key=lambda x: datetime.datetime.strptime(ORTHO_DATETIME[x], "%m-%d-%Y"),
        )

        # Plot bar graphs for each metric
        for metric in STAT_FIELDS:
            fig, ax = plt.subplots()
            x = np.arange(len(orthos_sort_datetime))
            y = [adj_metrics[ortho][metric] for ortho in orthos_sort_datetime]

            ax.bar(x, y)
            ax.set_xticks(x)
            ax.set_xticklabels(orthos_sort_datetime, rotation=45)
            ax.set_title(f"{metric} for Orthos")

            plot_path = os.path.join(
                args.plot_folder, f"{metric.lower()}_bar_graph_sorted.png"
            )
            print(f"Saving {metric} bar graph at {plot_path}")
            plt.savefig(plot_path)
            plt.clf()
            plt.close()
