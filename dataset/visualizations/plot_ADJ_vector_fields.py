""" 
Summary:
This script computes the vector fields from the adjustments. The vector fields are computed by matching the adjustments to the grid points. 
The grid points are generated from the boundary of the polygon. The best line is selected for each grid point.
The vector field is computed from the best line. The vector field is then plotted.

instructions:
update the direcotry variables to the correct path: annotation_dir, ortho_dir, boundary_dir (line 29-31)

usage:
python plot_ADJ_vector_fields.py --annotation_fused_file <annotation_fused_file> --ortho_file <ortho_file> --boundary_file <boundary_file> --area <area> --disaster <disaster>

Example:
python plot_ADJ_vector_fields.py  --annotation_fused_file 1001-San-Carlos-Island.geo.tif.json  --ortho_file 1001-San-Carlos-Island.geo.tif --boundary_file 1001-San-Carlos-Island.geo.tif.json --area 1.06  --disaster Hurricane-Ian

python plot_ADJ_vector_fields.py \
    --annotation_fused_file 090401-DMS-Assessment-Westpark.geo.tif.json \
    --ortho_file 090401-DMS-Assessment-Westpark.geo.tif\
    --boundary_file 090401-DMS-Assessment-Westpark.geo.tif.json\
    --area 1.06 \
    --disaster Hurricane-Harvey

"""

import os

import json
import rasterio
import shapely
import argparse

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import shape

from dataset.utils.adjustment_utils import match_polygon_to_adjustment


def compute_vector_field(line):
    dx = line[1][0] - line[0][0]
    dy = line[1][1] - line[0][1]
    return dx, dy


# generate vector fields using the best lines
def generate_vector_fields_from_best_lines(best_lines):
    vector_fields = []
    for line in best_lines:
        vector_field = compute_vector_field(line)
        vector_fields.append(vector_field)
    return vector_fields


def plot_vector_field(
    x_grid,
    y_grid,
    vector_fields,
    annotation_fused=[],
    filename="???",
    area="???",
    disaster_name="???",
    output_folder=".",
    image_background=None,
):

    x = [x for x in x_grid]
    y = [y for y in y_grid]

    x, y = np.meshgrid(x, y, indexing="ij")

    x_m = np.reshape(x, len(x) * len(y))
    y_m = np.reshape(y, len(x) * len(y))

    x, y, u, v = [], [], [], []  # for plotting
    for i in range(len(x_m)):

        x.append(x_m[i])
        y.append(y_m[i])

        u.append(vector_fields[i][0])
        v.append(vector_fields[i][1])

    if not image_background is None:
        plt.imshow(image_background, alpha=0.5)

    plt.quiver(x, y, u, v)
    plt.xlabel("Pixels(x)", fontsize=12)
    plt.ylabel("Pixels(y)", fontsize=12)

    plt.title(
        f"{filename} | {disaster_name}\nNumber of Adjustments = {len(annotation_fused)} | Area = {area} mi$^2$",
        fontsize=14,
    )
    plt.xticks([])
    plt.yticks([])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(f"{output_folder}/{filename}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute vector fields from adjustments"
    )
    parser.add_argument(
        "--annotation_fused_file", type=str, help="Path to the annotation fused file"
    )
    parser.add_argument("--ortho_file", type=str, help="Path to the ortho file")
    parser.add_argument("--boundary_file", type=str, help="Path to the boundary file")
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the output folder",
        default="vector_field_figures",
    )
    parser.add_argument("--area", type=float, help="Area of the boundary", default=0.00)
    parser.add_argument(
        "--disaster", type=str, help="Disaster name", default="disaster-name"
    )
    parser.add_argument(
        "--vector_count",
        type=int,
        help="The number of vectors in the field on both the x and y axes.",
        default=25,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        help="The factor by which imagery will be downsamples",
        default=10.0,
    )

    args = parser.parse_args()
    annotation_fused = json.load(open(args.annotation_fused_file, "r"))
    input_data = rasterio.open(args.ortho_file, "r")
    boundary_data = json.load(open(args.boundary_file, "r"))

    image_background = input_data.read(
        out_shape=(
            4,
            int(input_data.width / args.scale_factor),
            int(input_data.height / args.scale_factor),
        )
    )
    image_background = np.moveaxis(image_background, 0, 2)

    # parse the boundary data
    polygon_boundaries = []
    for i in range(0, len(boundary_data)):
        polygon_boundaries.append(shape(boundary_data[i]["geometry"]))

    # convert the list of polygons to a multipolygon
    polygon_boundary = shapely.MultiPolygon(polygon_boundaries)

    # generate grid from the boundary
    min_x, min_y, max_x, max_y = polygon_boundary.bounds

    inter_vector_x_spacing = (
        int(input_data.height / args.scale_factor) / args.vector_count
    )
    inter_vector_y_spacing = (
        int(input_data.width / args.scale_factor) / args.vector_count
    )
    x_grid = np.linspace(
        0 - inter_vector_x_spacing / 10,
        int(input_data.height / args.scale_factor) + inter_vector_x_spacing / 10,
        args.vector_count,
    )
    y_grid = np.linspace(
        0 - inter_vector_y_spacing / 10,
        int(input_data.width / args.scale_factor) + inter_vector_y_spacing / 10,
        args.vector_count,
    )

    # find best lines for each grid point against all adjustment lines
    best_lines = []
    for x in x_grid:
        for y in y_grid:
            point = [{"x": x * args.scale_factor, "y": y * args.scale_factor}]

            best_line = match_polygon_to_adjustment(annotation_fused, point)
            best_lines.append(best_line)

    # generate vector fields using the best lines
    vector_fields = generate_vector_fields_from_best_lines(best_lines)

    filename = os.path.split(args.ortho_file)[-1].split(".")[0]

    # plot vector field
    plot_vector_field(
        x_grid,
        y_grid,
        vector_fields,
        annotation_fused,
        filename,
        area=args.area,
        disaster_name=args.disaster,
        output_folder=args.output_folder,
        image_background=image_background,
    )
