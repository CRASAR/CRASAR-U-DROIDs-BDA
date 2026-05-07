import os
import argparse
import json

import rasterio

UNKNOWN_CRS = "UNKNOWN_CRS"
UNKNOWN_UNIT = "UNKNOWN_UNIT"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder',
                        type=str,
                        help='The input geotifs to be processed.')
    parser.add_argument('--output_file',
                        type=str,
                        help='The output file.')
    parser.add_argument('--target_file_postfix',
                        type=str,
                        help="The file postfix that is used to determine if files should be selected for inspection.",
                        default=".tif")
    parser.add_argument('--default_crs_linear_unit',
                        type=str,
                        help="The units used in the CRS if the CRS linear unit is not known in the data.")
    parser.add_argument('--default_crs_name',
                        type=str,
                        help="The CRS name used if it is not known in the data.")
    args = parser.parse_args()

    gsds = {}

    for (root,dirs,files) in os.walk(args.input_folder, topdown=True):
        for file in files:
            if file.endswith(args.target_file_postfix):

                #Load the ortho
                input_geotif = os.path.join(root, file)
                print("Getting GSD from:", input_geotif)
                input_data = rasterio.open(input_geotif, "r")

                # Get the affine transformation coefficients
                transform = input_data.transform

                # Calculate the pixel dimensions in the x and y directions
                pixel_width = transform[0]
                pixel_height = transform[4]

                # Compute the ground sample distance
                gsd_x = abs(pixel_width)
                gsd_y = abs(pixel_height)

                # Get the units from the data or the default
                unit = UNKNOWN_UNIT
                if not input_data.crs is None:
                    unit = input_data.crs.linear_units
                elif not args.default_crs_linear_unit is None:
                    unit = args.default_crs_linear_unit

                # Get the name of the CRS from the data or the default
                crs_name = UNKNOWN_CRS
                if not input_data.crs is None:
                    crs_name = input_data.crs.to_string()
                elif not args.default_crs_name is None:
                    unit = args.default_crs_name

                if gsd_x != gsd_y:
                    print("Warning... GSD_x and GSD_y are different.")

                x_dim_area = gsd_x * input_data.width
                y_dim_area = gsd_y * input_data.height
                area = x_dim_area * y_dim_area

                gsds[file] = {
                    "CRS": crs_name,
                    "gsd_x": gsd_x,
                    "gsd_y":gsd_y,
                    "gsd_units":str(unit) + "/px",
                    "pixels_x":input_data.width,
                    "pixels_y":input_data.height,
                    "height":y_dim_area,
                    "width":x_dim_area,
                    "height_and_width_units":str(unit),
                    "area":area,
                    "area_units":str(unit)+"^2"
                }

    #Save that polygon
    with open(os.path.join(args.output_file), "w") as f:
        f.write(json.dumps(gsds))
