import os
import json
import argparse
import rasterio

from alive_progress import alive_bar
from collections import defaultdict

from shapely.geometry import Polygon

from pyproj import Transformer

from dataset.constants import (
    LABEL_SCORE_PRIORITY_MAP,
    LABELBOX_DATASET_TO_ORTHO_TITLE,
    NO_DAMAGE,
    UNCLASSIFIED,
)
from dataset.utils.adjustment_utils import apply_adjustments
from shapely.validation import make_valid


def get_annotations_by_ortho(annotations, suas):
    ortho_to_annotations = defaultdict(list)
    if suas:
        for a in annotations:
            external_id = a["data_row"]["external_id"]
            ortho_name = external_id.split("_")[0].split("/")[-1].split("\\")[-1]
            ortho_to_annotations[ortho_name].append(a)
    else:
        for a in annotations:
            external_id = a["data_row"]["external_id"]
            name = os.path.split(external_id)[-1]
            name = name.split("\\")[-1]
            ortho_name = ".geo.tif_".join(name.split(".geo.tif_")[:2]) + ".geo.tif"
            ortho_to_annotations[ortho_name].append(a)
    return ortho_to_annotations


def fuse_labels(label_1, label_2):
    if LABEL_SCORE_PRIORITY_MAP[label_1.lower()] > LABEL_SCORE_PRIORITY_MAP[label_2.lower()]:
        return label_1
    return label_2


def cascade_fuse_polygons(polygons_to_fuse):
    working_polygons = polygons_to_fuse[:]
    fused_polygons = []
    match_found = True
    while match_found:
        match_found = False
        with alive_bar(total=len(working_polygons)) as bar:

            i = 0
            while i < len(working_polygons):

                label_i = working_polygons[i][0]
                poly_i = working_polygons[i][1]
                match_found_i = False

                j = i + 1
                while j < len(working_polygons) and (not match_found_i):
                    label_j = working_polygons[j][0]
                    poly_j = working_polygons[j][1]

                    if poly_i.intersects(poly_j):

                        fused_label = fuse_labels(label_i, label_j)

                        # Fixes to avoid self-intersection errors
                        if not poly_i.is_valid:
                            poly_i = make_valid(poly_i)

                        if not poly_j.is_valid:
                            poly_j = make_valid(poly_j)

                        fused_poly = poly_i.union(poly_j)

                        fused_polygons.append([fused_label, fused_poly])

                        match_found = True
                        match_found_i = True

                        working_polygons.pop(j)
                    j += 1

                if not match_found_i:
                    fused_polygons.append([label_i, poly_i])

                i += 1
                bar()
        working_polygons = fused_polygons[:]
        fused_polygons = []
    return working_polygons


def parse_labeled_polygons_from_annotations(annotations, project_key):
    labeled_polygons = []
    with alive_bar(total=len(annotations)) as bar:
        for annotation in annotations:
            file_name = annotation["data_row"]["global_key"]
            coords = file_name.split("(")[-1].split(")")[0].split(",")
            x = int(coords[0])
            y = int(coords[1])

            labels = annotation["projects"][project_key]["labels"]
            for label in labels:
                annotated_objects = label["annotations"]["objects"]
                for annotated_object in annotated_objects:
                    name = annotated_object["name"]
                    polygon = []
                    for point in annotated_object["polygon"]:
                        polygon.append([x + point["x"], y + point["y"]])
                    labeled_polygons.append([name, Polygon(polygon)])
            bar()
    return labeled_polygons


def count_labels(labeled_polygons, label_to_count):
    count = 0
    for label, _ in labeled_polygons:
        if label == label_to_count:
            count += 1
    return count


def convert_labeled_polygons_to_json_xy_and_lat_lon(
    labeled_polygons, geotiff_data, adjustments
):
    LAT_LON_CRS = "EPSG:4326"
    result = []
    coord_transformer = Transformer.from_crs(geotiff_data.crs.to_string(), LAT_LON_CRS, always_xy=True)
    for label, polygon in labeled_polygons:

        individual_polys = []
        if polygon.geom_type == "MultiPolygon":
            individual_polys = list(polygon.geoms)
        else:
            individual_polys = [polygon]

        for p in individual_polys:
            pixel_coords_polygon = []
            target_crs_polygon = []

            for x, y in list(zip(*p.exterior.coords.xy)):

                x_adj, y_adj = apply_adjustments(adjustments, x, y)

                pixel_coords_polygon.append({"x": x_adj, "y": y_adj})

                # Flip the y and x axis to align the data correctly in the coordinate space
                x_source, y_source = rasterio.transform.xy(geotiff_data.transform, y_adj, x_adj)
                x_t, y_t = coord_transformer.transform(x_source, y_source)
                target_crs_polygon.append({"lat": y_t, "lon": x_t})

        result.append(
            {
                "source": "Microsoft",
                "label": label,
                "pixels": pixel_coords_polygon,
                LAT_LON_CRS: target_crs_polygon,
            }
        )

    return result


def bulk_annotate(fused_labels, initial_label, bulk_label):
    for i in range(0, len(fused_labels)):
        if fused_labels[i][0] == initial_label:
            fused_labels[i][0] = bulk_label
    return fused_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="reconstruct_BDA",
        description="This program reconstructs the building mask from the annotations.",
    )
    parser.add_argument(
        "--annotations_file", type=str, help="The path to the annotations file."
    )
    parser.add_argument(
        "--adjustments_file",
        type=str,
        help="The path to the adjustments file.",
        default=None,
    )
    parser.add_argument(
        "--project_key",
        type=str,
        help="The labelbox project key.",
        default="clo6d7nxp0592072p4mmv2t38",
    )
    parser.add_argument(
        "--geotif_path_map_file",
        type=str,
        help="The input file that maps from geotifs titles to their full path.",
    )
    parser.add_argument(
        "--geotif_annotation_map_file",
        type=str,
        help="The output file that maps from the geotif full paths to their annotation full paths.",
    )
    parser.add_argument(
        "--output_json_folder",
        type=str,
        help="The path to the file that will contain the output building polygons.",
    )
    parser.add_argument(
        "--bulk",
        help="If BDA labels are unclassified, should they default to the no damage label.",
        action="store_true",
    )
    parser.add_argument(
        "--suas",
        help="If source is drone, naming convention is different.",
        action="store_true",
    )
    args = parser.parse_args()

    try:
        os.makedirs(args.output_json_folder)
    except FileExistsError as e:
        pass

    geotif_annotation_map = {}

    print("Loading Geotif File Path Mapping...")
    f = open(args.geotif_path_map_file, "r")
    geotif_path_map = json.loads(f.read())
    f.close()

    print("\n")
    print("Parsing annotations...")
    f = open(args.annotations_file, "r")
    annotations = f.readlines()
    f.close()

    print("\n")
    adjustments = None
    if not args.adjustments_file is None:
        print("Parsing adjustments...")
        f = open(args.adjustments_file, "r")
        adjustments = json.loads(f.read())
        f.close()

    parsed_annotations = [json.loads(a) for a in annotations]

    ortho_to_annotations = get_annotations_by_ortho(parsed_annotations, args.suas)
    print(ortho_to_annotations.keys())

    ortho_label_counts = {}

    for ortho in ortho_to_annotations.keys():

        # Get the path to the ortho
        valid_ortho = False
        try:
            if ortho in LABELBOX_DATASET_TO_ORTHO_TITLE.keys():
                ortho_local_title = LABELBOX_DATASET_TO_ORTHO_TITLE[ortho]
            else:
                ortho_local_title = ortho
            geotif_path = geotif_path_map[ortho_local_title]
            valid_ortho = True
        except KeyError:
            print("Skipping annotations for orthomosaic:", ortho)
            print("Was unable to find an orthomosaic with that title.")

        if valid_ortho:
            # Load the ortho
            print("Loading external base geotif from:", geotif_path)
            input_data = rasterio.open(geotif_path, "r")
            print("Done...")

            labeled_polygons = parse_labeled_polygons_from_annotations(
                ortho_to_annotations[ortho], args.project_key
            )
            print(
                "Parsed annotations from",
                len(parsed_annotations),
                "tiles and found",
                len(labeled_polygons),
                "polygons.",
            )
            print("\n")
            print("Merging polygons")
            print("Starting cascade...")
            fused_labeled_polygons = cascade_fuse_polygons(labeled_polygons)

            if args.bulk:
                print("Performing bulk annotation...")
                fused_labeled_polygons = bulk_annotate(
                    fused_labeled_polygons, UNCLASSIFIED, NO_DAMAGE
                )

            print(
                "Fused polygons using intersectionality. Found",
                len(fused_labeled_polygons),
                "non intersecting polygons.",
            )

            ortho_label_counts[ortho_local_title] = {
                "1 - no damage": count_labels(fused_labeled_polygons, "no damage"),
                "2 - minor damage": count_labels(fused_labeled_polygons, "minor damage"),
                "3 - major damage": count_labels(fused_labeled_polygons, "major damage"),
                "4 - destroyed": count_labels(fused_labeled_polygons, "destroyed"),
                "5 - obscured": count_labels(fused_labeled_polygons, "obscured"),
                "6 - un-classified": count_labels(fused_labeled_polygons, "un-classified"),
                "7 - total": len(fused_labeled_polygons),
            }

            print("\tFound", ortho_label_counts[ortho_local_title]["1 - no damage"], "polygons with the no damage label",)
            print("\tFound", ortho_label_counts[ortho_local_title]["2 - minor damage"], "polygons with the minor damage label",)
            print("\tFound", ortho_label_counts[ortho_local_title]["3 - major damage"], "polygons with the major damage label",)
            print("\tFound", ortho_label_counts[ortho_local_title]["4 - destroyed"], "polygons with the destroyed label",)
            print("\tFound", ortho_label_counts[ortho_local_title]["5 - obscured"], "polygons with the obscured label",)
            print("\tFound", ortho_label_counts[ortho_local_title]["6 - un-classified"], "polygons with the un-classified label",)

            adj = []
            if adjustments:
                try:
                    adj = adjustments[ortho]
                    print("Found adjustments for this orthomosaic...")
                except KeyError as e:
                    pass

            json_polygons = convert_labeled_polygons_to_json_xy_and_lat_lon(
                fused_labeled_polygons, input_data, adj
            )

            print("\n")
            print("Writing polygons to json file...")
            out_file = ortho_local_title + ".json"
            out_path = os.path.join(args.output_json_folder, out_file)
            f = open(out_path, "w")
            f.write(json.dumps(json_polygons, indent=4, sort_keys=True))
            f.close()
            print("Polygons saved at", out_path)
            geotif_annotation_map[geotif_path] = out_path

    print("Generating Geotif Annotation Path Map...")
    f = open(args.geotif_annotation_map_file, "w")
    f.write(json.dumps(geotif_annotation_map))
    f.close()
    print("Done...")
