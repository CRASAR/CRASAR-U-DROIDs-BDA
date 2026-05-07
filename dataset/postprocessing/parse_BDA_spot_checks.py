import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

import argparse
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="fuse_path_maps",
        description="This This is a utility script that combines path maps.",
    )
    parser.add_argument(
        "--bda_spot_check_csv_path",
        type=str,
        help="The path to the CSV with all the BDA spot checks",
    )
    parser.add_argument(
        "--spot_check_file_output_path",
        type=str,
        help="The second path map to be fused",
    )
    args = parser.parse_args()

    os.makedirs(args.spot_check_file_output_path, exist_ok=True)

    # Load the data from the csv
    data = pd.read_csv(args.bda_spot_check_csv_path)

    # For every row in the csv
    for index, r in data.iterrows():
        # Get the data and the file name that it corresponds to
        source_file = r.iloc[0]
        spot_check_data = r.iloc[1]

        # Look to see if there is valid data
        valid = False
        if not pd.isnull(spot_check_data):
            check = spot_check_data.replace("\n", "").replace(" ", "").lower()
            if check != "none":
                valid = True

        print("Working", source_file)

        # If the data is valid then we write it to a file
        if valid:
            f = open(
                os.path.join(args.spot_check_file_output_path, source_file + ".txt"),
                "w",
            )
            lines = []
            for line in spot_check_data.split("\n"):
                if "->" in line:
                    lines.append(line.strip().replace("\n", "").replace("\r", ""))
            f.write("\n".join(lines))
            print("\t" + str(len(lines)), "Spot checks parsed. Done...")

        # Otherwise we skip it
        else:
            print("\tSkipped - No Data")
