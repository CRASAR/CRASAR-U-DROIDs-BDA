import os
import json
import argparse

def generate_empty_adjustments_data():
    return [[[0,0], [0,0]]]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='make_empty_adj', description='This program adjusts the building mask from the adjustments annotations.')
    parser.add_argument('--geotif_path_map_file', type=str, help='The path map for the geotifs.')
    parser.add_argument('--target_geotif', type=str, help="The geotif for which we want to generate a bank adjustments file")
    parser.add_argument('--adjustments_folder', type=str, help='The folder where the adjustments should be written.')
    parser.add_argument('--output_adjustment_map_path', type=str, help='The output mapping from geotif path to adjustment path.')
    args = parser.parse_args()

    out_path = os.path.join(args.adjustments_folder, args.target_geotif + ".json")

    path_map = {}
    if args.geotif_path_map_file:
        with open(args.geotif_path_map_file) as geotif_path_map_file_object:
            geotif_path_map = json.load(geotif_path_map_file_object)
        geotif_path = geotif_path_map[args.target_geotif]
        path_map = {geotif_path: out_path}
    else:
        print("Warning: No input geotif path map passed. Output adjustment map will not be generated.")

    print("Writing blank adjustments file...")
    with open(out_path, "w") as f:
        f.write(json.dumps(generate_empty_adjustments_data()))
    print("Done")

    if args.geotif_path_map_file and args.output_adjustment_map_path:
        print("Writing output addons annotations path map ...")
        with open(args.output_adjustment_map_path, "w") as path_map_out:
            path_map_out.write(json.dumps(path_map))
        print("Done.")
