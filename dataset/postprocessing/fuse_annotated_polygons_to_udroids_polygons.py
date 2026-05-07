import os


from shapely.geometry import Polygon
from pyproj import Transformer
from alive_progress import alive_bar
import argparse
import json
import numpy as np
import rasterio


from dataset.constants import LAT_LON_CRS
from dataset.utils.ortho_utils import transform_bounds

def load_building_geoms(base_builings_path):
    f = open(base_builings_path, "r")
    base_buildings = json.loads(f.read())
    f.close()

    return base_buildings


def rebase_annotated_polygons(
    annotations_path_map,
    geotif_path_map,
    target_geotif,
    base_builings_path,
    output_folder_path,
    output_path_map,
    swap_xy=False
):
    print("Working", target_geotif)
    print("\tLoading geotiff path map")
    input_geotif_path = json.load(open(geotif_path_map))[target_geotif]
    print("\t\tGeotif loading from", input_geotif_path)

    print("\tLoading annotations")
    annotations_path = json.load(open(annotations_path_map))[input_geotif_path]
    f = open(annotations_path, "r")
    annotated_buildings = json.loads(f.read())
    f.close()
    print("\t\tAnnotations loaded from", annotations_path)

    print("\tLoading Base Polygons")
    base_buildings = load_building_geoms(base_builings_path)
    print("\t\tBase Polygons loaded from", base_builings_path)

    print("\tLoading external base geotif from:", input_geotif_path)
    input_geotiff_data = rasterio.open(input_geotif_path, "r")

    coord_system = str(input_geotiff_data.crs)
    print("\tGenerating transformer between:", coord_system, "and", LAT_LON_CRS)
    coord_transformer = Transformer.from_crs(LAT_LON_CRS, coord_system)

    ortho_lat_lon_bounds = transform_bounds(input_geotiff_data, LAT_LON_CRS)
    _ = [
        ortho_lat_lon_bounds.left,
        ortho_lat_lon_bounds.bottom,
        ortho_lat_lon_bounds.right,
        ortho_lat_lon_bounds.top,
    ]

    print("Finding matches between annotated and unannotated polygons")
    matched_base_polygon_indicies = []
    with alive_bar(total=len(annotated_buildings)) as bar:
        for annotated_building in annotated_buildings:
            ab_polygon = Polygon(
                [[p["lon"], p["lat"]] for p in annotated_building["EPSG:4326"]]
            )
            ious = []

            for base_building in base_buildings:
                base_polygon = Polygon(
                    [[p["lon"], p["lat"]] for p in base_building["EPSG:4326"]]
                )
                ious.append(
                    ab_polygon.intersection(base_polygon).area
                    / ab_polygon.union(base_polygon).area
                )

            max_iou = max(ious)

            if max_iou == 0:
                matched_base_polygon_indicies.append(-1)
            else:
                matched_base_polygon_indicies.append(np.argmax(ious))

            if max_iou <= 0.25:
                if max_iou > 0:
                    print("WARNING: Match found, but intersection/union was", max_iou)

                else:
                    print(
                        "WARNING: NO MATCH FOUND! Intersection/union was",
                        max_iou,
                        "| Label:",
                        annotated_building["label"],
                    )

            bar()

    print("Merging down paired annotated polygons")
    rebased_annotation_pairs = []
    for annotation_idx, base_building_idx in enumerate(matched_base_polygon_indicies):
        if base_building_idx != -1:
            rebased_annotation_pairs.append(
                [annotated_buildings[annotation_idx], base_buildings[base_building_idx]]
            )

    print("Rebasing Annotations")
    rebased_annotations = []
    for source, rebased in rebased_annotation_pairs:
        rebased_annotation = {}
        coords = []
        pixels = []
        # Since we are working with lats and lons in the polygon, we need to convert and get pixels
        for p in rebased["EPSG:4326"]:
            coords.append({"lat": p["lat"], "lon": p["lon"]})
            if swap_xy:
                x_source, y_souce = coord_transformer.transform(p["lon"], p["lat"])
            else:
                x_source, y_souce = coord_transformer.transform(p["lat"], p["lon"])
            y_p, x_p = rasterio.transform.rowcol(
                input_geotiff_data.transform, x_source, y_souce
            )
            pixels.append({"x": x_p, "y": y_p})
        rebased_annotation["pixels"] = pixels
        rebased_annotation["EPSG:4326"] = coords
        rebased_annotation["label"] = source["label"]

        rebased_annotation["id"] = rebased["id"]
        rebased_annotation["source"] = rebased["source"]
        rebased_annotations.append(rebased_annotation)

    print("\tWriting rebased annotations...")
    path = os.path.join(output_folder_path, target_geotif + ".json")
    os.makedirs(output_folder_path, exist_ok=True)
    f = open(path, "w")
    f.write(json.dumps(rebased_annotations))
    f.close()
    output_path_map[input_geotif_path] = path
    print("\tDone")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="rebase_annotated_polygons",
        description="This program takes the annotated building polygons and fuses them with the unannotated initial polygons that were used to seed the annotation. As a result, this handles any failed tiles during the initial annotation process.",
    )
    parser.add_argument(
        "--annotations_path_map",
        type=str,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--geotif_path_map", type=str, help="The input geotif path map."
    )
    parser.add_argument(
        "--base_polygons_path_map", type=str, help="The path map to the base polygons"
    )
    parser.add_argument(
        "--output_annotations_folder",
        type=str,
        help="The path to the output annotations folder updated polygons applied.",
    )
    parser.add_argument(
        "--output_annotations_path_map",
        type=str,
        help="The path map for the rebased annotations.",
    )
    parser.add_argument(
        "--swap_xy",
        action="store_true",
        help="Flag to determine if xy axes should be swapped.",
    )
    args = parser.parse_args()

    f = open(args.annotations_path_map, "r")
    annotations_path_map = json.loads(f.read())
    f.close()

    f = open(args.base_polygons_path_map, "r")
    base_polygons_path_map = json.loads(f.read())
    f.close()

    resulting_path_map = {}

    print("\tLoading base polygons")
    for target_geotif_path, annotations_file in annotations_path_map.items():
        swap_xy = False
        if    ("Mexico" in target_geotif_path
            or "Pecan-Grove" in target_geotif_path
            or "DMS-Assessment-Westpark" in target_geotif_path
            or "DMS-Assessment-Sienna" in target_geotif_path
            or "Lancaster-Canyon-Gate" in target_geotif_path
            or args.swap_xy
        ):
            swap_xy = True

        target_geotif_file = os.path.split(target_geotif_path)[-1]

        base_polygons_file = base_polygons_path_map[target_geotif_file]

        rebase_annotated_polygons(
            args.annotations_path_map,
            args.geotif_path_map,
            target_geotif_file,
            base_polygons_file,
            args.output_annotations_folder,
            resulting_path_map,
            swap_xy=swap_xy,
        )

    print("Writing Output Path Map...")
    f = open(args.output_annotations_path_map, "w")
    f.write(json.dumps(resulting_path_map))
    f.close()
