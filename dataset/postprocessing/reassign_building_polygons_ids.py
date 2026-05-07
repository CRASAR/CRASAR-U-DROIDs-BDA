import os
import json
import argparse
from shapely.geometry import Polygon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reassign_building_polygon_ids",
        description="This program reassigns building polygon ids for overlapping buildings.",
    )
    parser.add_argument("--annotations_path_map", help="Path to annotations path map.")
    parser.add_argument(
        "--output_folder",
        type=str,
        help="The path to the folder where the resulting annotations will be stored.",
    )
    parser.add_argument(
        "--output_annotations_path_map",
        type=str,
        help="The path map for the annotations.",
    )
    args = parser.parse_args()

    try:
        os.makedirs(args.output_folder)
    except FileExistsError as e:
        pass

    all_building_locations = {}
    output_path_map = {}
    total_reassigned_buildings = 0

    building_ids = []

    annotations_path_map = json.load(open(args.annotations_path_map, "r"))

    for geotif_path, annoations_path in annotations_path_map.items():
        _, file = os.path.split(annoations_path)
        print("Working", file)

        # Read in annotations for current file
        buildings = json.load(open(annoations_path))
        print("Looking at the ", len(buildings), " in this ortho...")
        for building in buildings:
            building_location = [[p["lon"], p["lat"]] for p in building["EPSG:4326"]]
            building_polygon = Polygon(building_location)
            reassigned = False

            if building["id"] not in building_ids:
                building_ids.append(building_ids)

            for existing_building in all_building_locations.keys():
                if (
                    existing_building.equals_exact(building_polygon, tolerance=1e-7)
                    and building["id"] != all_building_locations[existing_building]
                ):
                    print(
                        "\tReassigning Building ID for building within ",
                        building["id"],
                        "->",
                        all_building_locations[existing_building],
                        "|",
                        file,
                    )
                    building["id"] = all_building_locations[existing_building]
                    total_reassigned_buildings += 1
                    reassigned = True

            if not reassigned:
                all_building_locations[building_polygon] = building["id"]

        print("\tSaving files to the annotation file path... ", file)
        out_path = os.path.join(args.output_folder, file)
        with open(out_path, "w") as f:
            f.write(json.dumps(buildings, indent=4, sort_keys=True))

        output_path_map[geotif_path] = out_path

    with open(args.output_annotations_path_map, "w") as f:
        f.write(json.dumps(output_path_map))

    print("Total Unique Building IDs found in orignal dataset", len(building_ids))
    print("Total Reassigned Buildings: ", total_reassigned_buildings)
