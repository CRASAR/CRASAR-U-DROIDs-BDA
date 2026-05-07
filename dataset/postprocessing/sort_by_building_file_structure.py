import os
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="sort_by_building_file_structure",
        description="This program organizes the BDA annotations by the buildings.",
    )
    parser.add_argument(
        "--suas_annotations_path_map",
        type=str,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--satellite_annotations_folder",
        type=str,
        help="The path to the satellite annotations to be added.",
        default=None,
    )
    parser.add_argument(
        "--crewed_annotations_folder",
        type=str,
        help="The paht to the crewed annotations folder to be added.",
        default=None,
    )
    parser.add_argument(
        "--output_annotations_folder",
        type=str,
        help="The path to the output annotations folder updated polygons applied.",
    )
    parser.add_argument(
        "--output_annotations_path_map",
        type=str,
        help="The path map for the rebased annotations.",
    )
    args = parser.parse_args()

    try:
        os.makedirs(args.output_annotations_folder)
    except FileExistsError as e:
        pass

    print("Loading suas Annotation File Path Mapping...")
    with open(args.suas_annotations_path_map, "r") as f:
        annotation_path_map = json.loads(f.read())

    boundary_output_files = {}
    boundary_to_annotations_files = {}
    unique_building_ids = []
    print("Sorting the annotations based on boundaries...")
    for geotif_path, annotation_path in annotation_path_map.items():
        boundary_name = os.path.split(geotif_path)[1]
        print("\t Boundary Name: ", boundary_name)

        boundary_to_annotations_files[boundary_name] = []

        with open(annotation_path, "r") as f:
            suas_annotations = json.loads(f.read())
        boundary_to_annotations_files[boundary_name].append(suas_annotations)

        if args.satellite_annotations_folder is not None:
            for root, dir, files in os.walk(args.satellite_annotations_folder):
                for file in files:
                    if boundary_name in file:
                        print("\t\tAdding annotations to boundary: ", file)
                        with open(os.path.join(root, file), "r") as f:
                            satellite_annotations = json.loads(f.read())
                        boundary_to_annotations_files[boundary_name].append(
                            satellite_annotations
                        )

        if args.crewed_annotations_folder is not None:
            for root, dir, files in os.walk(args.crewed_annotations_folder):
                for file in files:
                    if boundary_name in file:
                        print("\t\tAdding annotations to boundary: ", file)
                        with open(os.path.join(root, file), "r") as f:
                            manned_annotations = json.loads(f.read())
                        boundary_to_annotations_files[boundary_name].append(
                            manned_annotations
                        )

    total_buildings = 0
    boundary_to_buidlings = {b: {} for b in boundary_to_annotations_files.keys()}
    for boundary, annotataions in boundary_to_annotations_files.items():
        print(
            "Merging Annotations for boundary ",
            boundary,
            "- Number of Views: ",
            len(annotataions),
        )
        boundary_total_buildings = 0
        for data in annotataions:
            for building in data:
                building_id = building["id"]
                if building_id not in boundary_to_buidlings[boundary].keys():
                    boundary_to_buidlings[boundary][building_id] = []
                    total_buildings += 1
                    boundary_total_buildings += 1
                boundary_to_buidlings[boundary][building_id].append(building)
                if building_id not in unique_building_ids:
                    unique_building_ids.append(building_id)
        print("\tMerged Building: ", boundary_total_buildings)

        out_file = boundary + ".json"
        out_path = os.path.join(args.output_annotations_folder, out_file)
        with open(out_path, "w") as f:
            f.write(
                json.dumps(boundary_to_buidlings[boundary], indent=4, sort_keys=True)
            )
        boundary_output_files[boundary] = out_path

    print("Generating Annotation Path Map...")
    with open(args.output_annotations_path_map, "w") as f:
        f.write(json.dumps(boundary_output_files))
    print("Done...")

    print(
        "Total Buildings (aggregated for each boundary - includes same building polygons)",
        total_buildings,
    )
    print("Total Unique Building within Dataset: ", len(unique_building_ids))
