import os
import math
import json
import argparse
import pandas as pd
import shapely
import rasterio
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from geojson_length import calculate_distance, Unit
from pyproj import Transformer

from dataset.constants import RDA_DATASET_CLASSES, LAT_LON_CRS, ROAD_LINE
from dataset.utils.adjustment_utils import apply_adjustments, match_vertex_to_adjustment

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="compute_RDA_dataset_stats",
        description="This program combines the spot checks that were manually entered and udpates the annotations file.",
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

        damaged_polygons = [poly for cls in RDA_DATASET_CLASSES for poly in polygons[cls]]
        if damaged_polygons:
            damaged_union = unary_union(damaged_polygons)
        else:
            damaged_union = None  # No damaged areas

        road_lines_length = 0
        road_line_adjusted_difference = 0
        damaged_road_length = 0
        class_lengths = {r: 0 for r in RDA_DATASET_CLASSES}
        for line in annotations_data["road_lines"]:
            ls_adj = None
            ls = LineString([(p["lon"], p["lat"]) for p in line["EPSG:4326"]])
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
                ls_adj = LineString(verts)
            else:
                ls = ls_adj

            for r in RDA_DATASET_CLASSES:
                for poly_r in polygons[r]:
                    ls_poly_r = shapely.intersection(ls_adj if args.adjust else ls, poly_r)
                    if ls_poly_r.length > 0:
                        ls_poly_r_geojson = {
                            "geometry": json.loads(shapely.to_geojson(ls_poly_r))
                        }
                        class_lengths[r] += calculate_distance(
                            ls_poly_r_geojson, Unit.meters
                        )

            ls_adj_geojson = {"geometry": json.loads(shapely.to_geojson(ls_adj if args.adjust else ls))}
            ls_geojson = {"geometry": json.loads(shapely.to_geojson(ls))}
            road_lines_length += calculate_distance(ls_adj_geojson, Unit.meters)
            road_line_adjusted_difference += math.fabs(calculate_distance(ls_geojson, Unit.meters) - calculate_distance(ls_adj_geojson, Unit.meters))

            # Calculate damaged road length by intersecting with damaged_union
            if damaged_union and ls_adj.intersects(damaged_union):
                damaged_part = ls_adj.intersection(damaged_union)
                if not damaged_part.is_empty:
                    damaged_part_geojson = {
                        "geometry": json.loads(shapely.to_geojson(damaged_part))
                    }
                    damaged_road_length += calculate_distance(
                        damaged_part_geojson, Unit.meters
                    )


        # Compute undamaged road length
        undamaged_road_length = road_lines_length - damaged_road_length

        data_dict = {}
        data_dict["0 - road line - meters"] = road_lines_length
        data_dict["1 - adjustment distance change absolute - meters"] = road_line_adjusted_difference
        for i, label in enumerate(RDA_DATASET_CLASSES):
            key = str(i + 1) + " - " + label + " - meters"
            data_dict[key] = class_lengths[label]

        # Add damaged and undamaged road lengths
        data_dict["Damaged road length - meters"] = damaged_road_length
        data_dict["Undamaged road length - meters"] = undamaged_road_length
        ortho_label_stats[ortho_local_title] = data_dict

    print("Saving summary statistics file...")
    stats = pd.DataFrame(ortho_label_stats)
    stats.transpose().to_csv(args.output_stats_file_path)
    print("Done...")
