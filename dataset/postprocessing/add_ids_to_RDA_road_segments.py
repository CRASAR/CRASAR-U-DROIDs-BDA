import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.utils.line_utils import get_line_id

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='add_ids_to_RDA_road_segments', description='This program breaks the road lines into road line segments, and adds ids to each of the road line segments.')
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--output_annotations_folder', type=str, help="The patht to the output annotations folder with the metadata applied.")
    parser.add_argument('--output_annotations_path_map', type=str, help="The path to the output annotations file path map.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_annotations_folder)
    except FileExistsError as e:
        pass

    annotations_path_map = json.load(open(args.annotations_path_map))
    output_map = {}

    print("Generating metadata...")
    for target_geotif_path, file in annotations_path_map.items():
        _, target_geotif = os.path.split(target_geotif_path)
        annotations_data = json.load(open(file))

        restructured_annotations = {
            "polygons":annotations_data["polygons"],
            "road_lines":[]
        }

        #For every road line we have
        for i in range(0, len(annotations_data["road_lines"])):
            #Make sure that it has the same number of segments in pixel and spatial coordinate spaces
            assert(len(annotations_data["road_lines"][i]["EPSG:4326"]) == len(annotations_data["road_lines"][i]["pixels"]))
            #Compute the number of segments in the road line
            num_segments = len(annotations_data["road_lines"][i]["EPSG:4326"]) - 1
            
            #For each segment, generate the details of the segment in the new format
            for j in range(0, num_segments):
                new_line = {
                    "id": get_line_id(),
                    "pixels":[annotations_data["road_lines"][i]["pixels"][j], annotations_data["road_lines"][i]["pixels"][j+1]],
                    "EPSG:4326":[annotations_data["road_lines"][i]["EPSG:4326"][j], annotations_data["road_lines"][i]["EPSG:4326"][j+1]],
                    "source":annotations_data["road_lines"][i]["source"],
                    "label":annotations_data["road_lines"][i]["label"]
                }
                restructured_annotations["road_lines"].append(new_line)

        output_path = os.path.join(args.output_annotations_folder, target_geotif + ".json")
        f = open(output_path, "w")
        f.write(json.dumps(restructured_annotations))
        f.close()
        output_map[target_geotif_path] = output_path

    print("Writing output annotations path map ...")
    f = open(args.output_annotations_path_map, "w")
    f.write(json.dumps(output_map, indent=4, sort_keys=True))
    f.close()
    print("Done.")