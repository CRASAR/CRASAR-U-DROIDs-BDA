import os
import json
import argparse

import geopandas as gpd
import rasterio
import shapely

from rasterio.features import geometry_mask
from shapely.geometry import Polygon, LineString

from dataset.constants import RDA_DATASET_CLASSES, RDA_COLOR_MAP, ROAD_LINE, ROAD_DEBUG_WIDTH
from dataset.utils.adjustment_utils import apply_adjustments, match_polygon_to_adjustment, match_vertex_to_adjustment


def generate_debug_rda_ortho(annotations_path_map,
                             adjustments_path_map,
                             geotif_path_map,
                             geotif_review_folder,
                             target_geotif,
                             channels=4,
                             unadjust_roads=False,
                             ignore_last_channel=True):
    input_geotif_path = json.load(open(geotif_path_map))[target_geotif]
    annotations_path = json.load(open(annotations_path_map))[input_geotif_path]
    adjustments_path = None
    try:
        adjustments_path = json.load(open(adjustments_path_map))[input_geotif_path]
    except KeyError:
        print("Could not find adjustments for", target_geotif)
        print("Generating unadjusted debug RDA geotiff")

    # Load the ortho
    print("Loading external base geotif from:", input_geotif_path)
    input_geotiff_data = rasterio.open(input_geotif_path, "r")
    channel_data = []
    for i in range(0, channels):
        channel_data.append(input_geotiff_data.read(i+1))
    print("Done...")

    #Load the annotations
    print("Loading the RDA annotations from:", annotations_path)
    f = open(annotations_path, "r")
    data = json.loads(f.read())
    f.close()

    adjustments = {}
    if not adjustments_path is None:
        print("Parsing adjustments...")
        f = open(adjustments_path, "r")
        adjustments = json.loads(f.read())
        f.close()

    #Convert all the polygons into shapely geometries, and group them based on the damage label
    polygons = {c:[] for c in RDA_DATASET_CLASSES}
    for polygon in data["polygons"]:
        best_adjustment = match_polygon_to_adjustment(adjustments, polygon["pixels"])
        best_adjustment = None
        if best_adjustment is None:
            best_adjustment_options = []
        else:
            best_adjustment_options = [best_adjustment]

        verts = []
        for point in polygon["pixels"]:
            x_adj, y_adj = apply_adjustments(best_adjustment_options, point["x"], point["y"])
            x, y = rasterio.transform.xy(input_geotiff_data.transform, y_adj, x_adj)
            verts.append([x,y])
        polygons[polygon["label"]].append(Polygon(verts))



    road_lines = []
    road_line_polys = []

    for line in data["road_lines"]:
        # Align all points within the line according to the best adjustment
        verts = []
        for point in line["pixels"]:
            # Find the best adjustment for the roadline
            best_adjustment = match_vertex_to_adjustment(adjustments, point)
            if unadjust_roads:
                best_adjustment = None
            if best_adjustment is None:
                best_adjustment_options = []
            else:
                best_adjustment_options = [best_adjustment]

            if line["source"] == "custom":
                best_adjustment_options = []

            x_adj, y_adj = apply_adjustments(best_adjustment_options, point["x"], point["y"])
            x, y = rasterio.transform.xy(input_geotiff_data.transform, y_adj, x_adj)

            verts.append([x,y])
        road_lines.append(LineString(verts))
        poly_buf = shapely.buffer(LineString(verts), ROAD_DEBUG_WIDTH)
        road_line_polys.append(poly_buf)

    polygon_gdfs = {c: gpd.GeoDataFrame({'geometry': polygons[c]}, crs=input_geotiff_data.crs) for c in RDA_DATASET_CLASSES}
    roadline_gdf = gpd.GeoDataFrame({'geometry': road_line_polys}, crs=input_geotiff_data.crs)
    print("Done...")

    print("Masking...")
    for c in [ROAD_LINE] + RDA_DATASET_CLASSES:

        if c in RDA_DATASET_CLASSES:
            gdf = polygon_gdfs[c]
        else:
            gdf = roadline_gdf

        if len(gdf.geometry) > 0:
            mask = geometry_mask(gdf.geometry, out_shape=input_geotiff_data.shape, transform=input_geotiff_data.transform, invert=True)
            print("\tCategory:", c)

            channel_data[0][mask] = channel_data[0][mask]*0.6 + RDA_COLOR_MAP[c][0]*0.4
            channel_data[1][mask] = channel_data[1][mask]*0.6 + RDA_COLOR_MAP[c][1]*0.4
            channel_data[2][mask] = channel_data[2][mask]*0.6 + RDA_COLOR_MAP[c][2]*0.4
            if not ignore_last_channel:
                channel_data[3][mask] = channel_data[3][mask]*0.6 + RDA_COLOR_MAP[c][3]*0.4


    print("Writing output geotif...")
    # Write the modified data to a new GeoTIFF file with an alpha channel
    output_geotif_path = os.path.join(geotif_review_folder, ("review_" + target_geotif))
    with rasterio.open(output_geotif_path, 'w',
                       driver='GTiff',
                       height=input_geotiff_data.height,
                       width=input_geotiff_data.width,
                       count=channels,
                       dtype='uint8',
                       crs=input_geotiff_data.crs,
                       transform=input_geotiff_data.transform) as dst:
        for i, band in enumerate(channel_data):
            dst.write(band, i+1)
    print("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='debug_RDA',
                                     description='This program reconstructs with the ortho along with the annotated RDA masks, and road lines overlayed.')
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--adjustments_path_map', type=str, help="The path to the adjustments file path map.", default=None)
    parser.add_argument('--geotif_path_map', type=str, help='The input geotif path map.')
    parser.add_argument('--target_geotif', type=str, help='The input geotif title.')
    parser.add_argument('--geotif_review_folder', type=str, help="The path to the folder that will contain the output review orthomosaic.")
    parser.add_argument('--unadjust', action='store_true')
    parser.add_argument('--channels', type=int, default=4, help="The number of channels in the orthomosaic.")
    args = parser.parse_args()

    generate_debug_rda_ortho(args.annotations_path_map,
                             args.adjustments_path_map,
                             args.geotif_path_map,
                             args.geotif_review_folder,
                             args.target_geotif,
                             args.channels,
                             args.unadjust)
