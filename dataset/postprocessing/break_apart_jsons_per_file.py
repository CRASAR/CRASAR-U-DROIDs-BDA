import os
import json
import argparse

from collections import defaultdict
from copy import deepcopy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Break apart an input json per key into files per that key.")
    parser.add_argument("--input_json", type=str, help="The path to the input data.")
    parser.add_argument("--output_folder", type=str, help="The path to the folder where the output files should be placed.")
    parser.add_argument("--jds_version", type=str, help="The value to be placed in the jds_version field")
    parser.add_argument("--payload_version", type=str, help="The value to be placed in the payload_version field")
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)

    data_per_file = defaultdict(lambda: [])
    for building_id in data.keys():
        for view in data[building_id]:
            udpated_dict = view
            view["view_id"] = view["id"]
            view["building_id"] = building_id
            view["jds_version"] = args.jds_version
            view["payload_version"] = args.payload_version
            data_per_file[view["filename"]].append(view)

    output_dir_specific = os.path.join(args.output_folder, *os.path.split(args.input_json.replace(".json", ""))[-1].split("_"))
    os.makedirs(output_dir_specific, exist_ok=True)
    for file in data_per_file.keys():
        with open(os.path.join(output_dir_specific, file), "w") as f:
            f.write(json.dumps(data_per_file[file], indent=4))

