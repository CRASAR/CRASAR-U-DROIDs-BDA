import os
import json
import argparse
import pandas as pd
import shapely
import rasterio
from shapely.geometry import Polygon, LineString
from geojson_length import calculate_distance, Unit
from pyproj import Transformer

from dataset.constants import RDA_DATASET_CLASSES, LAT_LON_CRS, ROAD_LINE
from dataset.utils.adjustment_utils import apply_adjustments, match_vertex_to_adjustment


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="compute_overlapping_RDA_stats",
        description="This program computes the length of overlapping RDA labels across the dataset",
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
        default=None,
    )
    parser.add_argument(
        "--output_stats_file_path",
        type=str,
        help="The path to the output statistics file.",
    )
    parser.add_argument("--adjust", action="store_true")
    args = parser.parse_args()

    with open(args.annotations_path_map, "r") as f:
        annotations_path_map = json.load(f)

    adjustments_path_map = None
    if not args.adjustments_path_map is None:
        with open(args.adjustments_path_map, "r") as f:
            adjustments_path_map = json.load(f)

    ortho_label_stats = {}

    for geotif_path, annotation_path in annotations_path_map.items():

        ortho_local_title = os.path.split(geotif_path)[1]

        # Load the annotations
        print("Loading the RDA annotations from:", annotation_path)
        with open(annotation_path, "r") as f:
            annotations_data = json.load(f)

        # Load Adjustements
        adjustments = {}
        if not adjustments_path_map is None:
            print("Parsing adjustments...")
            try:
                adjustments_path = adjustments_path_map[geotif_path]
                with open(adjustments_path, "r") as f:
                    adjustments = json.load(f)
            except KeyError:
                print("Could not find adjustments for", geotif_path)

        if args.adjust:
            print("Loading the target geotiff metadata from:", geotif_path)
            input_geotiff_data = rasterio.open(geotif_path, "r")
            transform = input_geotiff_data.transform
            coord_transformer_to_latlon = Transformer.from_crs(
                input_geotiff_data.crs.to_string(), LAT_LON_CRS, always_xy=True
            )

        polygons = {r: [] for r in RDA_DATASET_CLASSES + [ROAD_LINE]}
        for polygon in annotations_data["polygons"]:
            p_shape = Polygon(
                [(p["lon"], p["lat"]) for p in polygon["EPSG:4326"]]
            ).buffer(0)
            polygons[polygon["label"]].append(p_shape)

        road_lines_length = 0
        class_lengths = {r:{r2: 0 for r2 in RDA_DATASET_CLASSES} for r in RDA_DATASET_CLASSES}
        for line in annotations_data["road_lines"]:
            ls = None
            if args.adjust:
                verts = []
                for point in line["pixels"]:
                    # Find the best adjustment for the roadline
                    best_adjustment = match_vertex_to_adjustment(adjustments, point)
                    if best_adjustment is None:
                        best_adjustment_options = []
                    else:
                        best_adjustment_options = [best_adjustment]

                    if line["source"] == "custom":
                        best_adjustment_options = []

                    x_adj, y_adj = apply_adjustments(
                        best_adjustment_options, point["x"], point["y"]
                    )
                    x, y = rasterio.transform.xy(
                        input_geotiff_data.transform, y_adj, x_adj
                    )
                    lon, lat = coord_transformer_to_latlon.transform(x, y)

                    verts.append([lon, lat])
                ls = LineString(verts)

            else:
                ls = LineString([(p["lon"], p["lat"]) for p in line["EPSG:4326"]])

            for r in RDA_DATASET_CLASSES:
                for poly_r in polygons[r]:
                    ls_poly_r = shapely.intersection(ls, poly_r)
                    if ls_poly_r.length > 0:
                        for r2 in RDA_DATASET_CLASSES:
                            for polyr2 in polygons[r2]:
                                overlapping_polygon = poly_r.intersection(polyr2)
                                if overlapping_polygon.area > 0:
                                    ls_poly_r2 = shapely.intersection(ls, overlapping_polygon)
                                    if ls_poly_r2.length > 0:
                                        overlap_geojson = {"geometry": json.loads(shapely.to_geojson(ls_poly_r2))}
                                        class_lengths[r][r2] += calculate_distance(overlap_geojson, Unit.meters)

            l_geojson = {"geometry": json.loads(shapely.to_geojson(ls))}
            road_lines_length += calculate_distance(l_geojson, Unit.meters)


        data_dict = {}
        key = "1 - road line - meters"
        data_dict[key] = road_lines_length
        print("\tFound", data_dict[key], "meters of road")
        total_overlapping_road = 0
        for i, label in enumerate(RDA_DATASET_CLASSES):
            for j, label2 in enumerate(RDA_DATASET_CLASSES):
                if label != label2:
                    total_overlapping_road += class_lengths[label][label2]
            for label2, length in class_lengths[label].items():
                if label != label2:
                    overlap_key = f"overlap_{label}_{label2}_meters"
                    data_dict[overlap_key] = length
        print("\tFound", total_overlapping_road, "meters of overlapping road")
        data_dict["total overlap"] = total_overlapping_road

        ortho_label_stats[ortho_local_title] = data_dict

    print("Saving summary statistics file...")
    stats = pd.DataFrame(ortho_label_stats)
    stats.transpose().to_csv(args.output_stats_file_path)
    print("Done...")
