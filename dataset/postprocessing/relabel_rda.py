import json
import os
import argparse

from dataset.constants import PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS, PASSABLE_WITH_DIFFICULTY_FLOODING, PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION, \
                              NOT_PASSABLE_OBSTRUCTIONS, NOT_PASSABLE_FLOODING, NOT_PASSABLE_DESTRUCTION, NOT_ABLE_TO_DETERMINE, PARTICULATE_PARTIAL, \
                              PARTICULATE_TOTAL, ROAD_LINE, PARTIAL_OBSTRUCTION, PARTIAL_FLOODING, PARTIAL_DESTRUCTION, TOTAL_OBSTRUCTION, TOTAL_FLOODING,\
                              TOTAL_DESTRUCTION

label_remap = {
    PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS: PARTIAL_OBSTRUCTION,
    PASSABLE_WITH_DIFFICULTY_FLOODING: PARTIAL_FLOODING,
    PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION: PARTIAL_DESTRUCTION,
    NOT_PASSABLE_OBSTRUCTIONS: TOTAL_OBSTRUCTION,
    NOT_PASSABLE_FLOODING: TOTAL_FLOODING,
    NOT_PASSABLE_DESTRUCTION: TOTAL_DESTRUCTION,
    NOT_ABLE_TO_DETERMINE: NOT_ABLE_TO_DETERMINE,
    PARTICULATE_PARTIAL: PARTICULATE_PARTIAL,
    PARTICULATE_TOTAL: PARTICULATE_TOTAL,
    ROAD_LINE: ROAD_LINE,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="relabel_rda", description="This program relabels the RDA annotations to the final schema")
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--output_rda_annotations_folder', type=str, help="The path to the folder where the output data should be saved.")
    parser.add_argument('--output_rda_annotations_path_map', type=str, help="The output path map for the relabeled RDA data.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_rda_annotations_folder)
    except FileExistsError as e:
        pass

    with open(args.annotations_path_map, "r") as f:
        annotations_path_map = json.loads(f.read())
    output_annotations_path_map = {}

    for target_geotif_path, file in annotations_path_map.items():
        with open(os.path.join(file), "r") as f:
            data = json.loads(f.read())

        relabeled_data = data
        for i in range(0, len(relabeled_data["polygons"])):
            original_label = relabeled_data["polygons"][i]["label"]
            if original_label in label_remap:
                relabeled_data["polygons"][i]["label"] = label_remap[relabeled_data["polygons"][i]["label"]]
            else:
                print(f"[WARNING] Unmapped label '{original_label}' in file {file}. Keeping original.")

        out_file_name = os.path.join(args.output_rda_annotations_folder, file)
        with open(out_file_name, "w") as f:
            f.write(json.dumps(relabeled_data))

        output_annotations_path_map[target_geotif_path] = out_file_name

    print("Writing output annotations path map ...")
    with open(args.output_rda_annotations_path_map, "w") as f:
        f.write(json.dumps(output_annotations_path_map, indent=4, sort_keys=True))
    print("Done.")
