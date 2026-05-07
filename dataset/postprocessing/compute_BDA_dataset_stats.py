import os
import json
import argparse
import pandas as pd


def count_labels(labeled_polygons, label_to_count):
    count = 0
    for p in labeled_polygons:
        if p["label"] == label_to_count:
            count += 1
    return count


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="compute_BDA_dataset_stats",
        description="This program computes high level statistics about the BDA annotations.",
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

    annotations_path_map = json.load(open(args.annotations_path_map))

    ortho_label_counts = {}

    for geotif_path, annotation_path in annotations_path_map.items():

        ortho_local_title = os.path.split(geotif_path)[1]

        try:
            # Load the annotations
            print("Loading the BDA annotations from:", annotation_path)
            f = open(annotation_path, "r")
            annotations_data = json.loads(f.read())
            f.close()

            ortho_label_counts[ortho_local_title] = {
                "1 - no damage": count_labels(annotations_data, "no damage"),
                "2 - minor damage": count_labels(annotations_data, "minor damage"),
                "3 - major damage": count_labels(annotations_data, "major damage"),
                "4 - destroyed": count_labels(annotations_data, "destroyed"),
                "5 - un-classified": count_labels(annotations_data, "un-classified"),
                "6 - obscured": count_labels(annotations_data, "obscured"),
                "7 - total": len(annotations_data),
            }

            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["1 - no damage"],
                "polygons with the no damage label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["2 - minor damage"],
                "polygons with the minor damage label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["3 - major damage"],
                "polygons with the major damage label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["4 - destroyed"],
                "polygons with the destroyed label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["5 - un-classified"],
                "polygons with the un-classified label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["6 - obscured"],
                "polygons with the obscured label",
            )
        except TypeError:
            print("Skipping ortho", geotif_path)

    print("Saving summary statistics file...")
    stats = pd.DataFrame(ortho_label_counts)
    stats.transpose().to_csv(args.output_stats_file_path)
    print("Done...")
