import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.utils.polygon_utils import get_polygon_id

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='add_BDA_metadata', description='This program adds metadata, like IDs and sources for the building polygons.')
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--meta_path_map', type=str, help="The path to the output meta annotations file path map.")
    parser.add_argument('--output_annotations_folder', type=str, help="The patht to the output annotations folder with the metadata applied.")
    args = parser.parse_args()

    annotations_path_map = json.load(open(args.annotations_path_map))
    output_map = {}

    print("Generating metadata...")
    for target_geotif_path, file in annotations_path_map.items():
        _, target_geotif = os.path.split(target_geotif_path)
        annotations_data = json.load(open(file))

        for i in range(0, len(annotations_data)):
            annotations_data[i]["source"] = "Microsoft"
            annotations_data[i]["id"] = get_polygon_id()

        output_path = os.path.join(args.output_annotations_folder, target_geotif + ".json")
        f = open(output_path, "w")
        f.write(json.dumps(annotations_data))
        f.close()
        output_map[target_geotif_path] = output_path


    print("Writing metadata annotations path map ...")
    f = open(args.meta_path_map, "w")
    f.write(json.dumps(output_map))
    f.close()
    print("Done.")