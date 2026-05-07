import os
import json
import math
import argparse

from collections import defaultdict
from alive_progress import alive_bar

from dataset.constants import ADJ_LABELBOX_PREFIX_ORTHO_TITLE


def get_annotations_by_ortho(annotations):
    ortho_to_annotations = defaultdict(list)
    for a in annotations:
        external_id = a["data_row"]["external_id"]
        ortho_name = None
        if "_4250_4250" in external_id:
            ortho_name = external_id.split("\\")[-1].split("_4250_4250")[0]
        elif "_8500_8500" in external_id:
            ortho_name = external_id.split("\\")[-1].split("_8500_8500")[0]
        elif "_tile_" in external_id:
            ortho_name = external_id.split("\\")[-1].split("_tile_")[0]
        if ortho_name:
            ortho_to_annotations[ortho_name].append(a)
    return ortho_to_annotations


def parse_adjustments_from_annotations(annotations, project_key):
    adjustments_aligned = []
    with alive_bar(total=len(annotations)) as bar:
        for annotation in annotations:
            file_name = annotation["data_row"]["global_key"]
            coords = file_name.split("(")[-1].split(")")[0].split(",")
            x = int(coords[0])
            y = int(coords[1])

            adjustments = []
            verts = []
            labels = annotation["projects"][project_key]["labels"]
            for label in labels:
                annotated_objects = label["annotations"]["objects"]
                for annotated_object in annotated_objects:
                    if annotated_object["name"] == "Road Line":
                        for point in annotated_object["line"]:
                            verts.append([x + point["x"], y + point["y"]])
                    if annotated_object["name"] == "Building":
                        for point in annotated_object["polygon"]:
                            verts.append([x + point["x"], y + point["y"]])
                    if annotated_object["name"] == "Adjustment":
                        adjustments.append([])
                        for point in annotated_object["line"]:
                            adjustments[-1].append([x + point["x"], y + point["y"]])

            adjustments_aligned.extend(
                align_adjustments_to_verticies(verts, adjustments)
            )
            bar()
    return adjustments_aligned


def align_adjustments_to_verticies(verticies, adjustments):
    nearest_index = 0
    adjustments_aligned = []
    for line in adjustments:
        min_distance_global = float("inf")
        min_distance_index = -1
        for x_v, y_v in verticies:
            x_l_0, y_l_0 = line[0]
            dist_0 = math.dist([x_v, y_v], [x_l_0, y_l_0])

            x_l_1, y_l_1 = line[-1]
            dist_1 = math.dist([x_v, y_v], [x_l_1, y_l_1])

            min_dist_v = dist_0
            index_0 = 0
            index_1 = -1
            if dist_1 < dist_0:
                min_dist_v = dist_1
                index_0 = -1
                index_1 = 0

            if min_dist_v < min_distance_global:
                min_distance_global = min_dist_v
                min_distance_index = (index_0, index_1)

        adjustments_aligned.append(
            [line[min_distance_index[0]], line[min_distance_index[1]]]
        )
    return adjustments_aligned


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="reconstruct_ADJ",
        description="This program adjusts the building mask from the adjustments annotations.",
    )
    parser.add_argument(
        "--annotations_file", type=str, help="The path to the annotations file."
    )
    parser.add_argument(
        "--fused_annotations_folder",
        type=str,
        help="The path to the output fused annotations file.",
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
        help="The input geotif path map to be processed.",
    )
    parser.add_argument(
        "--geotif_annotation_map_file",
        type=str,
        help="The output mapping from geotif path to annotation path.",
    )
    args = parser.parse_args()

    try:
        os.makedirs(args.fused_annotations_folder)
    except FileExistsError as e:
        pass

    print("Loading Geotif File Path Mapping...")
    f = open(args.geotif_path_map_file, "r")
    geotif_path_map = json.loads(f.read())
    f.close()

    print("\n")
    print("Parsing annotations...")
    f = open(args.annotations_file, "r")
    annotations = f.readlines()
    f.close()

    parsed_annotations = [json.loads(a) for a in annotations]

    ortho_to_annotations = get_annotations_by_ortho(parsed_annotations)

    ortho_adjustments_path_map = {}

    for ortho in ortho_to_annotations.keys():

        # Get the path to the ortho
        valid_ortho = False
        try:
            if ortho in ADJ_LABELBOX_PREFIX_ORTHO_TITLE.keys():
                ortho_local_title = ADJ_LABELBOX_PREFIX_ORTHO_TITLE[ortho]
            else:
                ortho_local_title = ortho
            if not ortho_local_title is None:
                geotif_path = geotif_path_map[ortho_local_title]
                valid_ortho = True
        except KeyError:
            print("Skipping annotations for orthomosaic:", ortho)
            print("Was unable to find an orthomosaic with that title.")

        if valid_ortho:
            # Load the ortho
            print("Working:", ortho, "<>", ortho_local_title)
            adjustments = parse_adjustments_from_annotations(
                ortho_to_annotations[ortho], args.project_key
            )
            out_path = os.path.join(
                args.fused_annotations_folder, (ortho_local_title + ".json")
            )
            ortho_adjustments_path_map[os.path.split(geotif_path)[-1]] = out_path

            f = open(out_path, "w")
            f.write(json.dumps(adjustments))
            f.close()

            print(
                "Found", len(adjustments), "adjustments for orthomosaic:", geotif_path
            )

    f = open(args.geotif_annotation_map_file, "w")
    f.write(json.dumps(ortho_adjustments_path_map))
    f.close()
