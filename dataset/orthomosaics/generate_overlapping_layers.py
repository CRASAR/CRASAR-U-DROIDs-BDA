import os
import json
import numpy as np
import fnmatch
import rasterio
import argparse

from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.windows import Window
from shapely.geometry import box, shape
from alive_progress import alive_bar
from pyproj import Transformer
from collections import defaultdict

from dataset.constants import LAT_LON_CRS


def reproject_raster(in_path, out_path):

    with rasterio.open(in_path) as src_in:
        src_crs = src_in.crs
        transform, width, height = calculate_default_transform(
            src_crs, crs, src_in.width, src_in.height, *src_in.bounds
        )
        kwargs_rast = src_in.meta.copy()

        kwargs_rast.update(
            {"crs": crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(out_path, "w", **kwargs_rast) as dst_in:
            for i in range(1, src_in.count + 1):
                reproject(
                    source=rasterio.band(src_in, i),
                    destination=rasterio.band(dst_in, i),
                    src_transform=src_in.transform,
                    src_crs=src_in.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                )
    return out_path


def recurse_transform(l, transform):
    if isinstance(l, dict):
        coords = recurse_transform(l["coordinates"], transform)
        res = l.copy()
        res["coordinates"] = coords
        return res
    elif isinstance(l, list) and len(l) == 2 and isinstance(l[0], float):
        x_t, y_t = transform.transform(l[0], l[1])
        return [x_t, y_t]
    else:
        res = []
        for l_i in l:
            res.append(recurse_transform(l_i, transform))
        return res


def transform_polygon_boundaries(boundary_data, transform):
    geoms_value_pairs = []
    for i in range(0, len(boundary_data)):
        s = shape(recurse_transform(boundary_data[i]["geometry"], transform))
        geoms_value_pairs.append([s, 1.0])

    return geoms_value_pairs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="generate_overlapping_layers",
        description="This program takes a query geotif, and associated geotif boundary and a selection of target imagery and constructs a layer from that target imagery of the same area.",
    )
    parser.add_argument(
        "--query_geotif_path_map", type=str, help="The path to the geotif path map."
    )
    parser.add_argument(
        "--query_geotif_boundary_folder",
        type=str,
        help="The path to the boundaries folder.",
    )
    parser.add_argument(
        "--search_geotif_folder", type=str, help="The input geotif path map."
    )
    parser.add_argument(
        "--output_geotif_folder", type=str, help="The input geotif title."
    )
    parser.add_argument(
        "--target_mission", type=str, help="The name of the geotif used as a query."
    )
    parser.add_argument(
        "--search_align_foldername",
        action="store_true",
        help="If set, then all of the files found under the same folder will be aligned into one orthomosaic.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_geotif_path = json.load(open(args.query_geotif_path_map))[args.target_mission]

    # Load the query ortho
    print("Loading query geotif from:", input_geotif_path)
    query_geotif_data = rasterio.open(input_geotif_path, "r")
    query_box = box(*query_geotif_data.bounds)
    print(query_geotif_data.bounds)
    print("Done...")

    # Load the boundary file associated with this geotif
    boundary_file = None
    for file in os.listdir(args.query_geotif_boundary_folder):
        if args.target_mission in file:
            boundary_file = os.path.join(args.query_geotif_boundary_folder, file)
    print("Found boundary file:", boundary_file)
    f = open(os.path.join(boundary_file), "r")
    boundary_data = json.load(f)
    f.close()
    print("Done...")

    matched_files = []

    print("Searching for overlapping bounding boxes...")
    file_structure = [(r, d, f) for r, d, f in os.walk(args.search_geotif_folder)]
    with alive_bar(total=len(file_structure)) as status_bar:
        for root, dirnames, filenames in file_structure:
            for ortho_file in fnmatch.filter(filenames, "*.tif*"):
                # Get the geotif object
                search_geotif_data = rasterio.open(os.path.join(root, ortho_file), "r")

                # Build the coordinate transformer
                coord_transformer = Transformer.from_crs(
                    search_geotif_data.crs.to_string(),
                    query_geotif_data.crs.to_string(),
                    always_xy=True,
                )

                # Convert the bounding box to the query crs
                minx, miny = coord_transformer.transform(
                    search_geotif_data.bounds[0], search_geotif_data.bounds[1]
                )
                maxx, maxy = coord_transformer.transform(
                    search_geotif_data.bounds[2], search_geotif_data.bounds[3]
                )
                search_box_query_crs = box(minx, miny, maxx, maxy)

                intersection_area = query_box.intersection(search_box_query_crs).area
                if intersection_area > 0:
                    matched_files.append(os.path.join(root, ortho_file))
            status_bar()
    print("Done...")
    print("Matched", len(matched_files), "tiles")

    print("Aligning files")
    unit2files = defaultdict(list)
    for path in matched_files:
        if args.search_align_foldername:
            folder, file = os.path.split(path)
            unit = os.path.split(folder)[1]
            unit2files[unit].append(path)
        else:
            folder, file = os.path.split(path)
            unit = file.split(".tif")[0] + ".tif"
            unit2files[unit].append(path)
    print("Done...")

    print("Merging down orthomosaic")
    for i, unit in enumerate(unit2files.keys()):
        out_path = os.path.join(
            args.output_geotif_folder, args.target_mission + "_" + unit + ".geo.tif"
        )
        tmp_path = os.path.join(
            args.output_geotif_folder, args.target_mission + "_" + unit + "_tmp.geo.tif"
        )
        full_paths = unit2files[unit]
        print("Working:", out_path)

        # Open one to get the metadata
        tmp = rasterio.open(full_paths[0])
        print("\tMerging", len(full_paths), "tiles")
        mosaic, merged_transform = merge(full_paths)
        mosaic = np.ubyte(mosaic)
        output_meta = tmp.meta.copy()
        output_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": merged_transform,
            }
        )
        if args.debug:
            print(
                "\t\tFound an orthomosaic with the following dimensions:",
                mosaic.shape,
                "made from",
                len(full_paths),
                "tiles.",
            )
            print("\t\t", output_meta)

        print("\tRasterizing Boundary...")
        boundary_transformer = Transformer.from_crs(
            LAT_LON_CRS, tmp.crs.to_string(), always_xy=True
        )
        raster_geom = transform_polygon_boundaries(boundary_data, boundary_transformer)
        rasterized_boundary = rasterize(
            raster_geom,
            out_shape=(mosaic.shape[1], mosaic.shape[2]),
            fill=0.0,
            default_value=1.0,
            transform=merged_transform,
            all_touched=True,
            dtype=np.ubyte,
        )
        rasterized_boundary_stacked = np.stack(
            [rasterized_boundary] * mosaic.shape[0], axis=0
        )

        print("\tMasking Pixel Data...")
        # Consider if there is a way to do this in place, *= perhaps
        masked_mosaic = np.multiply(mosaic, rasterized_boundary_stacked, dtype=np.ubyte)
        mosaic = None
        rasterized_boundary_stacked = None

        pixel_sum = np.sum(masked_mosaic)
        print("\tFound the following data:", pixel_sum)

        # If we have data to write
        if pixel_sum > 0:
            print("\tWriting intermediate orthomosaic...")

            with rasterio.open(tmp_path, "w", **output_meta) as m:
                m.write(masked_mosaic)
            masked_mosaic = None

            src = rasterio.open(tmp_path)
            print("\tWriting terminal orthomosaic...")

            q2s_coord_transformer = Transformer.from_crs(
                query_geotif_data.crs.to_string(), tmp.crs.to_string(), always_xy=True
            )

            pyproj_bounds = q2s_coord_transformer.transform_bounds(
                *query_geotif_data.bounds
            )

            x_0_p, y_0_p = rasterio.transform.rowcol(
                merged_transform, pyproj_bounds[0], pyproj_bounds[1]
            )
            x_1_p, y_1_p = rasterio.transform.rowcol(
                merged_transform, pyproj_bounds[2], pyproj_bounds[3]
            )

            # Swap bounds for the windowing logic
            minx = min(y_0_p, y_1_p)
            maxx = max(y_0_p, y_1_p)
            miny = min(x_0_p, x_1_p)
            maxy = max(x_0_p, x_1_p)

            if args.debug:
                print("\tDebug Info...")
                print(
                    "\t\tquery_geotif_data.crs:",
                    query_geotif_data.crs.to_string(),
                    "target_geotif_data.crs:",
                    tmp.crs.to_string(),
                )
                print("\t\tquery_geotif_data.bounds:", query_geotif_data.bounds)
                print("\t\tTransformed query_geotif_data.bounds:", pyproj_bounds)
                print("\t\tmaxy:", maxy, "minx:", minx, "miny:", miny, "maxx:", maxx)
                print(
                    "\t\tGenerating window with shape:",
                    minx,
                    miny,
                    maxx - minx,
                    maxy - miny,
                )

            w = Window(minx, miny, maxx - minx, maxy - miny)

            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "height": w.height,
                    "width": w.width,
                    "transform": rasterio.windows.transform(w, src.transform),
                }
            )

            with rasterio.open(out_path, "w", **kwargs) as dst:
                dst.write(src.read(window=w))
            src.close()

            dst_debug = rasterio.open(out_path)
            print("Terminal orthomosaic bounds:", dst.bounds)

            try:
                os.remove(tmp_path)
            except PermissionError as e:
                print("Failed to remove temporary file with error", e)

        else:
            print("\tFound no valid data. Skipping...")
    print("Done...")
