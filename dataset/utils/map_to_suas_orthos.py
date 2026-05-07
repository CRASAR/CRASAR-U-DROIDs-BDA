import json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suas_ortho_path_map",
        type=str,
        help="The path to the path map for the sUAS data.",
    )
    parser.add_argument(
        "--other_ortho_path_map",
        type=str,
        help="The path to the path map for all the data that we are working with now.",
    )
    parser.add_argument(
        "--other_to_suas_path_map",
        type=str,
        help="The path to the file that will contain the mapping from ortho filenames to the sUAS file names.",
    )
    args = parser.parse_args()

    print("Loading sUAS File Path Mapping...")
    f = open(args.suas_ortho_path_map, "r")
    suas_ortho_path_map = json.loads(f.read())
    f.close()

    print("Loading other File Path Mapping...")
    f = open(args.other_ortho_path_map, "r")
    other_ortho_path_map = json.loads(f.read())
    f.close()

    result = {}

    for other_file, other_path in other_ortho_path_map.items():
        hit = False
        for suas_file, suas_path in suas_ortho_path_map.items():
            if suas_file.split(".geo.tif")[0] in other_path:
                result[other_path] = suas_path
                hit = True
                print("Linked", suas_file, "to", other_path)
        if not hit:
            result[other_path] = None

    f = open(args.other_to_suas_path_map, "w")
    f.write(json.dumps(result))
    f.close()
