import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
modeling_path = os.path.dirname(current)
dataset_path = os.path.dirname(modeling_path)
sys.path.append(modeling_path)
sys.path.append(dataset_path)

from dataset.constants import EVENTS

import argparse
import rasterio
import json
import tables as tb

class OrthoInfo(tb.IsDescription):
    gsd_x = tb.Float32Col(dflt=1, pos = 1)
    gsd_y = tb.Float32Col(dflt=1, pos = 2)
    epsg = tb.Int32Col(dflt=1, pos = 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='BDA_Tile_Data_Generator', description='This program generates tiles and their labeled building masks from geotiffs.')
    parser.add_argument('--orthos_path', type=str, help="Path to folder with orthomosaics")
    parser.add_argument('--geotif_path_map_file', type=str, help='The input geotif to be processed.')
    parser.add_argument('--out_folder', type=str, help="Path to folder where sampled image tiles will be stored.")
    parser.add_argument('--pytables_complib', type=str, help="The name of the pytables filter compression lib", default="blosc:lz4")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_folder), exist_ok=True)

    print("Loading Geotif File Path Mapping...")
    f = open(args.geotif_path_map_file, "r")
    geotif_path_map = json.loads(f.read())
    f.close()

    sample_tiles = {event: [] for event in EVENTS}

    table_filters = tb.Filters(complevel=1, complib=args.pytables_complib, fletcher32=False, shuffle=True)

    print("Found", len(geotif_path_map), "geotifs...")
    for geotif in geotif_path_map:
        print("Working geotif:", geotif)

        output_table_path = os.path.join(args.out_folder, geotif + ".h5")

        # Load the ortho
        print("\tOpening base geotif from:", geotif_path_map[geotif])
        raster_data = rasterio.open(geotif_path_map[geotif], "r")
        transform = raster_data.transform

        print("\tWriting data to h5:", output_table_path)
        fileh = tb.open_file(output_table_path, mode = "w")
        pixel_data_group = fileh.create_group(fileh.root, "pixel_data")
        meta_data_group = fileh.create_group(fileh.root, "meta")

        gsd_info_table = fileh.create_table(meta_data_group, "OrthoInfo", OrthoInfo)
        gsd_r = gsd_info_table.row
        gsd_r["gsd_x"] = transform[0]
        gsd_r["gsd_y"] = transform[4]
        gsd_r["epsg"] = int(str(raster_data.crs).split(":")[-1])
        gsd_r.append()

        at_matrix = [transform[i] for i in range(0, 6)]
        affine_transform_info_table = fileh.create_array(meta_data_group, "AffineTransform", at_matrix)

        channel = raster_data.read(1)
        channel_shape = channel.shape

        rgb_tb_array = fileh.create_earray(where=pixel_data_group,
                                           name="rgb",
                                           shape=(channel_shape[0], channel_shape[1], 0),
                                           filters=table_filters,
                                           expectedrows=3,
                                           atom=tb.UInt8Atom(),
                                           chunkshape=(2048, 2048, 3))

        rgb_tb_array.append(channel.reshape(channel_shape[0], channel_shape[1], 1))
        rgb_tb_array.flush()
        channel = None

        rgb_tb_array.append(raster_data.read(2).reshape(channel_shape[0], channel_shape[1], 1))
        rgb_tb_array.flush()
        rgb_tb_array.append(raster_data.read(3).reshape(channel_shape[0], channel_shape[1], 1))
        rgb_tb_array.flush()

        fileh.close()
        print("\tDone...")
    