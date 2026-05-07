import os
import argparse
import json

import rasterio
from area import area

from dataset.utils.ortho_utils import compute_ortho_polygon

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='The folder of input geotifs to be processed.')
    parser.add_argument('--output_folder', type=str, help='The output folder.')
    args = parser.parse_args()

    try:
        os.makedirs(args.output_folder)
    except FileExistsError as e:
        pass

    areas = {}

    for (root,dirs,files) in os.walk(args.input_folder, topdown=True):
        for file in files:
            if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
                out_file = file + ".json"
                out_path = os.path.join(args.output_folder, out_file)

                #Load the ortho
                input_geotif = os.path.join(root, file)
                print("Loading external base geotif from:", input_geotif)
                valid_ortho = True

                try:
                    input_data = rasterio.open(input_geotif, "r")
                except rasterio.errors.RasterioIOError as e:
                    print("Skipping", input_geotif, "because of the following error...")
                    print(e)
                    valid_ortho = False

                if valid_ortho:
                    #Compute the polygon that defines the boundaries of the ortho
                    print("Computing Bounding Polygon...")
                    valid_boundary=False
                    try:
                        shapes = compute_ortho_polygon(input_data)
                        valid_boundary=True
                    except rasterio.errors.CRSError as e:
                        print("Skipping " + str(file) + " as no valid boundary could be found because " + str(e))

                    if valid_boundary:
                        #Save that polygon
                        print("Writing Bounding Polygon...")
                        with open(out_path, "w") as bound_out:
                            bound_out.write(json.dumps(shapes))

                        #Compute the area
                        print("Computing area...")
                        a_sum = 0
                        for shape in shapes:
                            a = area(shape["geometry"])
                            a_sum += a
                        areas[file] = a_sum

                        print("Found an area of", a_sum, "m^2 from geotif", input_geotif)
                        print("Done...")

    #Save that polygon
    with open(os.path.join(args.output_folder, "totals.json"), "w") as total_stats_out:
        total_stats_out.write(json.dumps(areas))
