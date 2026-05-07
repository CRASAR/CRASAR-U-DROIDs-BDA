import os
import json
import argparse
from shapely.geometry import Polygon

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="reassign_building_polygon_ids",
        description="This program reassigns building polygon ids for overlapping buildings.",
    )
    parser.add_argument("--suas_annotations_path_map", help="Path to suas annotations path map.")
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

    suas_polygon_ids = {}
    suas_building_polygon_ids = []
    suas_annotations_path_map = json.load(open(args.suas_annotations_path_map, "r"))
    if len(suas_annotations_path_map) != 52:
        print("HF number of suas orthos should be 52!")
        raise ValueError()
    for geotif_path, annotations_path in suas_annotations_path_map.items():
        _, file = os.path.split(annotations_path)
        buildings = json.load(open(annotations_path))
        for building in buildings:
            building_location = [[p["lon"], p["lat"]] for p in building["EPSG:4326"]]
            building_polygon = Polygon(building_location)

            if building["id"] not in suas_building_polygon_ids:
                suas_building_polygon_ids.append(building["id"])

            if building_polygon not in suas_polygon_ids.keys():
                suas_polygon_ids[building_polygon] = building["id"]
                

    output_path_map = {}
    total_reassigned_buildings = 0
    failed = 0
    number_files = 0 

    building_ids = []

    annotations_path_map = json.load(open(args.annotations_path_map, "r"))
    print("Current Number of Annotations Files that will be reassigned are ", len(annotations_path_map))

    for geotif_path, annoations_path in annotations_path_map.items():
        try:
            _, file = os.path.split(annoations_path)
            print("Working", file)
            number_files +=1

            # Read in annotations for current file
            buildings = json.load(open(annoations_path))
            print("Looking at the ", len(buildings), " in this ortho...")
            for building in buildings:
                building_location = [[p["lon"], p["lat"]] for p in building["EPSG:4326"]]
                building_polygon = Polygon(building_location)
                reassinged = False

                for existing_building in suas_polygon_ids.keys():
                    if (existing_building.equals_exact(building_polygon, tolerance=1e-7) and not reassinged):
                            building["id"] = suas_polygon_ids[existing_building]
                            total_reassigned_buildings += 1
                            reassinged = True
                    
                if not reassinged:
                    print("WARNING! Found building polygons location that is not within u-driods...")
                    print("\tBuiliding Location --- ", building_polygon)
                    failed += 1

                if building["id"] not in building_ids:
                    building_ids.append(building["id"])
        except TypeError:
            print("Skipping ortho ", geotif_path, " due to TypeError, probably NoneType.")

        print("\tSaving files to the annotation file path... ", file)
        out_path = os.path.join(args.output_folder, file)
        with open(out_path, "w") as f:
            f.write(json.dumps(buildings, indent=4, sort_keys=True))

        output_path_map[geotif_path] = out_path

    with open(args.output_annotations_path_map, "w") as f:
        f.write(json.dumps(output_path_map))

    print("Total files looked at ", number_files)
    print("Total Unique Building IDs found in u-driods dataset", len(suas_building_polygon_ids))
    print("Total Unique Building IDs found in Reassigned Dataset", len(building_ids))
    print("Total Reassigned Buildings Ratio: ", total_reassigned_buildings/(total_reassigned_buildings + failed))
    print("\tFailed - ", failed, "Reassigned - ", total_reassigned_buildings)
    
