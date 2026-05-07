import os
import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ortho_path_map", type=str, help="The path to the path map for the sUAS data."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="The path to the folder for all the data that we are working with now.",
    )
    parser.add_argument(
        "--ortho_to_data_path_map",
        type=str,
        help="The path to the file that will contain the mapping from ortho filenames to the data file names.",
    )
    parser.add_argument(
        "--satellite_naming_convention", action="store_true"
    )  # NOTE: Tag added to handle satellite orthos, may not be necessary
    parser.add_argument(
        "--crewed_naming_convention", action="store_true"
    )  # NOTE: Tag added to handle satellite orthos, may not be necessary
    args = parser.parse_args()

    print("Loading sUAS File Path Mapping...")
    f = open(args.ortho_path_map, "r")
    ortho_path_map = json.loads(f.read())
    f.close()

    result = {}

    for imagery_file, path in ortho_path_map.items():
        hit = False
        for data_file in os.listdir(args.data_folder):
            if args.satellite_naming_convention:
                if data_file.split("geo.tif.json")[0] in imagery_file:
                    hit = True
                    result[imagery_file] = os.path.join(args.data_folder, data_file)
                    print("Linked", imagery_file, "to", data_file)
            elif args.crewed_naming_convention:
                if data_file.split("geo.tif.json")[0] in imagery_file:
                    hit = True
                    result[imagery_file] = os.path.join(args.data_folder, data_file)
                    print("Linked", imagery_file, "to", data_file)
            else:
                if data_file.split(".geo.tif")[0] in imagery_file:
                    hit = True
                    result[imagery_file] = os.path.join(args.data_folder, data_file)
                    print("Linked", imagery_file, "to", data_file)
        if not hit:
            result[imagery_file] = None

    print("Writing output path map to", args.ortho_to_data_path_map)
    f = open(args.ortho_to_data_path_map, "w")
    f.write(json.dumps(result))
    f.close()
