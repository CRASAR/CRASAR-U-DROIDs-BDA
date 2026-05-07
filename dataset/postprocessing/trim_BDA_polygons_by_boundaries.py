import os
import rasterio

from shapely.geometry import Polygon, shape
from dataset.utils.adjustment_utils import (
    apply_adjustments,
    match_polygon_to_adjustment,
)
from dataset.constants import UNCLASSIFIED, LAT_LON_CRS
from pyproj import Transformer
from shapely.validation import make_valid
from shapely.ops import transform as shapely_transform
from shapely.errors import GEOSException

import json
import argparse

def swap_xy(geom):
    return shapely_transform(lambda x, y, *args: (y, x), geom)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="trim_BDA_polygons_by_boundaries",
        description="This program removes all building polygons that are entirely outside the bounds of the orthomosaic polygon.",
    )
    parser.add_argument(
        "--geotif_annotation_map_file",
        type=str,
        help="The file that maps from the geotif full paths to their annotation full paths.",
    )
    parser.add_argument(
        "--adjustments_annotation_map_file",
        type=str,
        help="the annotations path map file for adjustments.",
    )
    parser.add_argument("--ortho_dir_path", type=str, help="ortho to path map.")
    parser.add_argument(
        "--boundaries_folder_path",
        type=str,
        help="The folder that contains all of the boundary polygons for the different orthos.",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        help="The path to the file that will contain the output building polygons.",
    )
    parser.add_argument(
        "--trimmed_path_map",
        type=str,
        help="The path to the output trimmed annotations file path map.",
    )
    parser.add_argument(
        "--swap_xy",
        action="store_true",
        help="Swap the xy axes of the building polygons.",
    )
    parser.add_argument("--sat", action="store_true", help="Satellite View.")
    args = parser.parse_args()
    try:
        os.makedirs(args.output_json_folder)
    except FileExistsError as e:
        pass

    f = open(args.adjustments_annotation_map_file, "r")
    adjustments_path_map = json.loads(f.read())
    f.close()

    f = open(args.ortho_dir_path, "r")
    ortho_path_map = json.loads(f.read())
    f.close()

    annotations_path_map = json.load(open(args.geotif_annotation_map_file))
    trimmed_annotations_path_map = {}

    for geotif_path, annotation_path in annotations_path_map.items():

        target_geotif = os.path.split(geotif_path)[1].replace(".json", "")

        # Load the annotations
        print("Loading the BDA annotations from:", annotation_path)
        f = open(annotation_path, "r")
        annotations_data = json.loads(f.read())
        f.close()

        if geotif_path in ortho_path_map.keys():
            ortho_path = ortho_path_map[geotif_path]
        else:
            ortho_path = ortho_path_map[target_geotif]
        print("Loading the target geotiff metadata from:", ortho_path)
        input_geotiff_data = rasterio.open(ortho_path, "r")

        transform = input_geotiff_data.transform
        coord_transformer = Transformer.from_crs(
            LAT_LON_CRS, input_geotiff_data.crs.to_string()
        )
        coord_transformer_to_latlon = Transformer.from_crs(
            input_geotiff_data.crs.to_string(), LAT_LON_CRS, always_xy=True
        )

        # Load the polygon boundary
        boundary_path = os.path.join(
            args.boundaries_folder_path, target_geotif + ".json"
        )
        print("Loading the boundary polygon from:", boundary_path)
        f = open(boundary_path, "r")
        boundary_data = json.load(f)
        f.close()

        polygon_boundaries = []
        for i in range(0, len(boundary_data)):
            polygon_boundaries.append(shape(boundary_data[i]["geometry"]))

        valid_buildings = []
        unclassified_buildings = 0
        for building in annotations_data:
            if (
                building["source"] == "custom"
            ):  # TODO: Consider if this will break with the drone data...
                coords = [(p["lon"], p["lat"]) for p in building["EPSG:4326"]]
                building_polygon = Polygon(coords)
            else:

                adjustments_data = []
                try:
                    if ortho_path in adjustments_path_map.keys():
                        annotation_path = adjustments_path_map[ortho_path]
                    else:
                        annotation_path = adjustments_path_map[target_geotif]
                    with open(annotation_path, "r") as f:
                        adjustments_data = json.loads(f.read())
                except KeyError:
                    print(
                        "KeyError when loading adjustments, proceeding without them...",
                        geotif_path,
                    )

                # Let's Adjust non-custom buildings prior to trimming them....
                best_adjustment = match_polygon_to_adjustment(
                    adjustments_data, building["pixels"]
                )
                verts = []
                if best_adjustment is None:
                    best_adjustment_options = []
                else:
                    best_adjustment_options = [best_adjustment]

                for point in building["pixels"]:
                    x_adj, y_adj = apply_adjustments(
                        best_adjustment_options, point["x"], point["y"]
                    )
                    x, y = rasterio.transform.xy(transform, y_adj, x_adj)
                    lon, lat = coord_transformer_to_latlon.transform(x, y)
                    if args.swap_xy or (
                        args.sat
                        and (
                            "MexicoBeach" in target_geotif
                            or "Pecan-Grove" in target_geotif
                        )
                    ):
                        verts.append([lat, lon])
                    else:
                        verts.append([lon, lat])
                building_polygon = Polygon(verts)

            original_area = building_polygon.area
            intersection_area = 0
            valid = False
            for polygon_boundary in polygon_boundaries:

                try:
                    intersection_polygon = building_polygon.intersection(
                        polygon_boundary
                    )
                except GEOSException:
                    print("Found Error, Attempting to make polygons valid to avoid...")
                    if not polygon_boundary.is_valid:
                        polygon_boundary = make_valid(polygon_boundary)
                    if not building_polygon.is_valid:
                        building_polygon = make_valid(building_polygon)
                    intersection_polygon = building_polygon.intersection(
                        polygon_boundary
                    )

                if intersection_polygon.area > 0:
                    valid = True
                    iou = intersection_polygon.area / original_area

                    if iou > intersection_area:
                        intersection_area = iou
            if valid:
                if intersection_area <= 0.5:
                    building["label"] = UNCLASSIFIED
                    unclassified_buildings += 1
                valid_buildings.append(building)
        print(
            "Selected",
            len(valid_buildings),
            "from a set of",
            len(annotations_data),
            "polygons",
        )
        print("Changed", unclassified_buildings, "building to Un-Classified Label")

        out_file = target_geotif + ".json"
        out_path = os.path.join(args.output_json_folder, out_file)
        f = open(out_path, "w")
        f.write(json.dumps(valid_buildings, indent=4, sort_keys=True))
        f.close()

        trimmed_annotations_path_map[geotif_path] = out_path

    print("Generating Annotation Path Map...")
    f = open(args.trimmed_path_map, "w")
    f.write(json.dumps(trimmed_annotations_path_map))
    f.close()
    print("Done...")
