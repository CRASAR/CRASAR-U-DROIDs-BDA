import os
import json
import argparse
from collections import defaultdict
from copy import deepcopy

from dataset.constants import LABEL_SCORE_PRIORITY_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="deduplicate_views",
        description="This program removes any duplicate views within an annotation file.",
    )
    parser.add_argument("--annotations_path_map", help="Path to annotations path map.")
    parser.add_argument(
        "--out_folder",
        help="Path to folder for where the output annotations should be stored.",
    )
    parser.add_argument(
        "--output_annotation_path_map",
        help="The path to where output annoation path map should be stored.",
    )
    args = parser.parse_args()

    # Make directory if doesn't exist
    try:
        os.makedirs(args.out_folder)
    except FileExistsError as e:
        pass

    # Load Annotations Path Map
    annotations_path_map = json.load(open(args.annotations_path_map))
    output_path_map = {}

    duplicate_views = 0
    for geotif_path, annoations_path in annotations_path_map.items():
        # Read in annotations for current file
        annotations = json.load(open(annoations_path))

        valid_views = []
        current_labels = defaultdict(list)
        id_2_bundles = defaultdict(list)

        for view in annotations:
            if "id" in view.keys():
                current_labels[view["id"]].append(view["label"])

                clean_view = deepcopy(view)
                clean_view["label"] = view["label"].lower()
                del clean_view["filename"]

                id_2_bundles[clean_view["id"]].append(clean_view)
                
                if clean_view["id"] not in [v["id"] for v in valid_views]:
                    valid_views.append(clean_view)
                else:
                    duplicate_views += 1
                    if len(set(current_labels[view["id"]])) > 1:
                        print(
                            "Found Conflicting Duplicate View with id: ",
                            clean_view["id"], view["filename"]
                        )
                        print(
                            "\tWARNING: Found label disagreement within a duplicate:",
                            current_labels[clean_view["id"]], view["filename"]
                        )
                    else:
                        print(
                            "Found Non-Conflicting Duplicate View with id: ",
                            clean_view["id"], view["filename"]
                        )
        
        deduplicated_annotations = []
        for building_id, bundles in id_2_bundles.items():
            sorted_bundles = sorted(bundles, key=lambda x:LABEL_SCORE_PRIORITY_MAP[x["label"].lower()])
            deduplicated_annotations.append(sorted_bundles[-1])
        
        out_path = os.path.join(args.out_folder, os.path.split(annoations_path)[-1])
        output_path_map[geotif_path] = out_path
        print("Writing to:", out_path)
        f = open(out_path, "w")
        f.write(json.dumps(deduplicated_annotations, indent=4))
        f.close()


    print("Total Duplicate Views: ", duplicate_views)

    print("Writing Output Path Map...")
    f = open(args.output_annotation_path_map, "w")
    f.write(json.dumps(output_path_map))
    f.close()

