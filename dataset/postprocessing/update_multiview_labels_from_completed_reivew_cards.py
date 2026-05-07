import os

import json
import argparse
import numpy as np

from collections import defaultdict
from PIL import Image

from dataset.constants import BDA_DAMAGE_CLASSES


def link_polygons_by_id(annotations_path_map):
    polygons_by_id = defaultdict(lambda: [])
    for filename, annotation_path in annotations_path_map.items():
        print("Loading", annotation_path)
        f = open(annotation_path, "r")
        polygon_data = json.loads(f.read())
        f.close()

        for p in polygon_data:
            p["filename"] = filename
            polygons_by_id[p["id"]].append(p)
    return polygons_by_id


def count_blue_pixel_mass(numpy_image_block):
    height, width, channels = numpy_image_block.shape
    rgb_block = np.copy(numpy_image_block[:, :, :3])

    mask = (rgb_block[..., 0] == rgb_block[..., 1]) & (
        rgb_block[..., 1] == rgb_block[..., 2]
    )
    rgb_block[mask] = [0, 0, 0]

    pixel_counts = np.sum(np.sum(rgb_block, axis=0), axis=0)
    return pixel_counts[2]  # 2 = blue channel


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
        "--review_cards_metadata",
        type=str,
        help="The path to the file where the review card metadata is saved.",
    )
    parser.add_argument(
        "--completed_reviews_folder",
        type=str,
        help="The path to the folder where the completed reviews are saved. If this field is set, then review cards for the cards already in the completed folder will not be completed.",
    )
    parser.add_argument(
        "--updated_building_labels_folder",
        type=str,
        help="The path to the folder where the building polygons with their updated labels will be saved.",
    )
    parser.add_argument(
        "--review_stats_file",
        type=str,
        help="The path to the statistics file that will be generated.",
    )
    parser.add_argument(
        "--include_all_buildings",
        action="store_true",
        help="When set, only building polygons that have been reviewed will be included in the output files.",
    )
    parser.add_argument(
        "--output_path_map",
        type=str,
        help="The path to the output path map that will be generated.",
    )
    parser.add_argument("--update", action="store_true", help="When set, output path map will be update with added reviews and not reset.")

    args = parser.parse_args()
    print("args.updated_building_labels_folder", args.updated_building_labels_folder)

    f = open(args.annotations_path_map, "r")
    annotations_path_map = json.loads(f.read())
    f.close()

    f = open(args.review_cards_metadata, "r")
    global_review_card_metadata = json.loads(f.read())
    f.close()

    all_polygons = link_polygons_by_id(annotations_path_map)

    data_by_filenames = defaultdict(lambda: [])
    reviewed_building_ids_by_file = defaultdict(lambda: [])

    review_transition_counts = defaultdict(
        lambda: {l: 0 for l in ["UNCHANGED"] + BDA_DAMAGE_CLASSES}
    )

    reviewed_ids = []
    for root, dirs, files in os.walk(args.completed_reviews_folder, topdown=False):
        for file in files:
            if file.endswith(".png"):
                building_id = file.replace(".png", "")
                building_id = building_id.replace(" (1)", "")
                building_id = building_id.replace("(1)", "")
                building_id = building_id.replace("copy", "")
                building_id = building_id.replace(" ", "")
                building_id = building_id.replace("(2)", "")
                building_id = building_id.replace("(3)", "")
                building_id = building_id.replace("(4)", "")
                building_id = building_id.replace("(5)", "")
                review_card_file = file
                try:
                    review_card_metadata = global_review_card_metadata[building_id]

                    review_card_image = Image.open(os.path.join(root, review_card_file))

                    for building_meta, label_volumes in review_card_metadata:
                        building_data = all_polygons[building_meta["id"]]
                        selected_building = None

                        for b in building_data:
                            if os.path.split(b["filename"])[-1].replace(".json", "") == os.path.split(building_meta["filename"])[-1].replace(".json", ""):
                                selected_building = b

                        max_count = 0
                        max_label = None

                        reviewed_building_ids_by_file[os.path.split(building_meta["filename"])[-1]].append(building_id)
                        
                        if not selected_building is None:
                            for label, volume_coords in label_volumes.items():
                                volume = np.asarray(review_card_image.crop(volume_coords))
                                count = count_blue_pixel_mass(volume)
                                if count > max_count:
                                    max_count = count
                                    max_label = label
                            if max_count > 1000:
                                review_transition_counts[building_meta["label"]][max_label] += 1
                                selected_building["label"] = max_label
                            else:
                                review_transition_counts[building_meta["label"]]["UNCHANGED"] += 1
                            data_by_filenames[os.path.split(building_meta["filename"])[-1]].append(selected_building)

                    reviewed_ids.append(building_id)
                except KeyError:
                    print("WARNING! Could Not Find Building ID: ", building_id)


    if args.include_all_buildings:
        for building_polygon_id in all_polygons.keys():
            for building_meta in all_polygons[building_polygon_id]:
                filename = os.path.split(building_meta["filename"])[-1]
                if not building_polygon_id in reviewed_building_ids_by_file[filename]:
                    data_by_filenames[os.path.split(building_meta["filename"])[-1]].append(building_meta)

    if args.update:
        f = open(args.output_path_map, "r")
        output_path_map = json.loads(f.read())
        f.close()
    else:
        output_path_map = {}

    os.makedirs(args.updated_building_labels_folder, exist_ok=True)
    for file_path in data_by_filenames.keys():
        filename = os.path.split(file_path)[-1].replace(".json", "")
        output_file = os.path.join(
            args.updated_building_labels_folder, filename + ".json"
        )
        print("Writing to", output_file)
        f = open(output_file, "w")
        f.write(json.dumps(data_by_filenames[file_path]))
        f.close()

        output_path_map[filename] = output_file

    f = open(args.review_stats_file, "w")
    f.write(json.dumps(review_transition_counts))
    f.close()

    f = open(args.output_path_map, "w")
    f.write(json.dumps(output_path_map))
    f.close()
