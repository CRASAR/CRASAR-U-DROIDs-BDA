import os
import json
import argparse
import difflib
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
from pyproj import Transformer

from dataset.constants import BDA_DAMAGE_CLASSES, BDA_CATEGORY_COLOR_MAP, LAT_LON_CRS
from dataset.utils.adjustment_utils import (
    apply_adjustments,
    match_polygon_to_adjustment,
)


# pylint: disable-next=too-many-branches
def generate_debug_bda_ortho(
    annotations_path_map,
    adjustments_path_map,
    geotif_path_map,
    geotif_review_folder,
    target_geotif,
    channels=4,
    ignore_last_channel=True,
    swap_xy=False,
):
    with open(geotif_path_map) as geotif_path:
        input_geotif_path = json.loads(geotif_path.read())[target_geotif]

    with open(annotations_path_map) as annotation_path:
        annotations_dict = json.loads(annotation_path.read())

    if target_geotif in annotations_dict.keys():
        annotations_path = annotations_dict[target_geotif]
    else:
        annotations_path = annotations_dict[input_geotif_path]

    adjustments_path = None

    try:
        with open(adjustments_path_map) as adj_path:
            adjustments_dict = json.loads(adj_path.read())
        if input_geotif_path in adjustments_dict.keys():
            adjustments_path = adjustments_dict[input_geotif_path]
        else:
            adjustments_path = adjustments_dict[target_geotif]
    except KeyError:
        print("Could not find adjustments for", target_geotif)
        print("Key Error:", input_geotif_path)
        print("Generating unadjusted debug BDA geotiff")
    
    # Load the ortho
    print("Loading external base geotif from:", input_geotif_path)
    input_geotiff_data = rasterio.open(input_geotif_path, "r")
    channel_data = []

    if channels == 1:
        channel_data = [input_geotiff_data.read(1).copy() for _ in range(3)]
        channels = 3
    else:
        for i in range(0, channels):
            channel_data.append(input_geotiff_data.read(i + 1))
    print("Done...")

    # Load the annotations
    print("Loading the BDA annotations from:", annotations_path)
    with open(annotations_path, "r") as f:
        data = json.loads(f.read())

    adjustments = {}
    if not adjustments_path is None:
        print("Parsing adjustments...")
        print("Loading Adjustments from: ", adjustments_path)
        with open(adjustments_path, "r") as f:
            adjustments = json.loads(f.read())

    coord_system = str(input_geotiff_data.crs)
    coord_transformer = Transformer.from_crs(LAT_LON_CRS, coord_system)

    # Convert all the polygons into shapely geometries, and group them based on the damage label
    polygons = {c: [] for c in BDA_DAMAGE_CLASSES}
    for polygon in data:

        if polygon["source"] == "custom":
            verts = []
            # If addon annotation, do not adjust...
            for point in polygon["pixels"]:
                x, y = point["x"], point["y"]
                x_source, y_source = rasterio.transform.xy(
                    input_geotiff_data.transform, y, x
                )
                if swap_xy:
                    verts.append((y_source, x_source))
                else:
                    verts.append((x_source, y_source))
        else:
            pixels = []
            if swap_xy:
                for point in polygon["EPSG:4326"]:
                    x_source, y_souce = coord_transformer.transform(
                        point["lon"], point["lat"]
                    )
                    y_p, x_p = rasterio.transform.rowcol(
                        input_geotiff_data.transform, x_source, y_souce
                    )
                    pixels.append({"x": x_p, "y": y_p})
            else:
                pixels = polygon["pixels"]
            best_adjustment = match_polygon_to_adjustment(adjustments, pixels)
            if best_adjustment is None:
                best_adjustment_options = []
            else:
                best_adjustment_options = [best_adjustment]

            verts = []

            for point in pixels:
                x_adj, y_adj = apply_adjustments(
                    best_adjustment_options, point["x"], point["y"]
                )

                # Flip the y and x axis to align the data correctly in the coordinate space
                x_source, y_source = rasterio.transform.xy(
                    input_geotiff_data.transform, y_adj, x_adj
                )

                verts.append((x_source, y_source))
        label = difflib.get_close_matches(polygon["label"], BDA_DAMAGE_CLASSES, n=1)

        polygons[label[0]].append(Polygon(verts))

    polygon_gdfs = {
        c: gpd.GeoDataFrame({"geometry": polygons[c]}, crs=input_geotiff_data.crs)
        for c in BDA_DAMAGE_CLASSES
    }
    print("Done...")

    print("Masking...")
    for c in BDA_DAMAGE_CLASSES:
        if len(polygon_gdfs[c].geometry) > 0:

            mask = geometry_mask(
                polygon_gdfs[c].geometry,
                out_shape=input_geotiff_data.shape,
                transform=input_geotiff_data.transform,
                invert=True,
            )
            print("\tCategory:", c)

            channel_data[0][mask] = (
                channel_data[0][mask] * 0.6 + BDA_CATEGORY_COLOR_MAP[c][0] * 0.4
            )
            channel_data[1][mask] = (
                channel_data[1][mask] * 0.6 + BDA_CATEGORY_COLOR_MAP[c][1] * 0.4
            )
            channel_data[2][mask] = (
                channel_data[2][mask] * 0.6 + BDA_CATEGORY_COLOR_MAP[c][2] * 0.4
            )
            if not ignore_last_channel:
                channel_data[3][mask] = (
                    channel_data[3][mask] * 0.6 + BDA_CATEGORY_COLOR_MAP[c][3] * 0.4
                )

    print("Writing output geotif...")
    # Write the modified data to a new GeoTIFF file with an alpha channel
    output_geotif_path = os.path.join(geotif_review_folder, ("review_" + target_geotif))
    with rasterio.open(
        output_geotif_path,
        "w",
        driver="GTiff",
        height=input_geotiff_data.height,
        width=input_geotiff_data.width,
        count=channels,
        dtype="uint8",
        crs=input_geotiff_data.crs,
        transform=input_geotiff_data.transform,
    ) as dst:
        rgba = np.array(channel_data)
        dst.write(rgba)
    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="debug_BDA",
        description="This program reconstructs with the ortho along with the annotated building mask overlayed.",
    )
    parser.add_argument(
        "--annotations_path_map",
        type=str,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--adjustments_path_map",
        type=str,
        help="The path to the adjustments file path map.",
    )
    parser.add_argument(
        "--geotif_path_map", type=str, help="The input geotif path map."
    )
    parser.add_argument("--target_geotif", type=str, help="The input geotif title.")
    parser.add_argument(
        "--geotif_review_folder",
        type=str,
        help="The path to the folder that will contain the output review orthomosaic.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=4,
        help="The number of channels in the orthomosaic.",
    )
    parser.add_argument(
        "--swap_xy",
        action="store_true",
        help="Swap the xy axes of the building polygons.",
    )
    parser.add_argument(
        "--plot_crs",
        action="store_true",
        help="Plot the crs data instead of the pixel data.",
    )
    args = parser.parse_args()

    generate_debug_bda_ortho(
        args.annotations_path_map,
        args.adjustments_path_map,
        args.geotif_path_map,
        args.geotif_review_folder,
        args.target_geotif,
        args.channels,
        swap_xy=args.swap_xy,
    )
