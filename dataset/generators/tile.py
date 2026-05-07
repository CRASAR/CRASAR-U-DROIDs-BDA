import os
import argparse
import json
import numpy as np
import rasterio
from rasterio.crs import CRS

from PIL import Image, ImageDraw
from pyproj import Transformer
from alive_progress import alive_bar
from shapely import Polygon

from dataset.constants import LAT_LON_CRS, MAJOR_DAMAGE, MINOR_DAMAGE
from dataset.utils.ortho_utils import crop_to_file, transform_bounds
from dataset.utils.polygon_utils import get_mask_from_polygons
from dataset.utils.draw_utils import (
    draw_roads_on_ortho_img,
    draw_buildings_on_ortho_img,
    get_road_polylines_from_osm_data,
)
from dataset.generators.osm_data_generator import (
    get_osm_roads_buildings_and_nodes_for_bounding_box,
)
from dataset.generators.geojson_building_polygon_data_generator import (
    load_buildings_from_geojson,
    get_geojson_buildings_for_bounding_box,
)

# 2048 x 2048
BUILDING_POLYGON_LIMIT = 20

def remove_duplicate_buildings(details):
    dict_buildings = {}
    cleaned_polygons = []

    for polygon in details["polygons"]:
        #pylint: disable-next=consider-iterating-dictionary
        if Polygon(polygon) not in dict_buildings.keys():
            dict_buildings[Polygon(polygon)] = 1
            cleaned_polygons.append(polygon)

    details["polygons"] = cleaned_polygons

    return details

# pylint: disable=too-many-branches
def image_tile_generate(
    input_geotif,
    tile_width,
    tile_height,
    out_folder,
    building_polygon_source,
    tmp_folder,
    generate_debug_images,
    tile_width_overlap_ratio,
    tile_height_overlap_ratio,
    mask_format,
    save_tmp_folder,
    geojson_building_data,
    swap_lat_lon=True,
    preannotated=False,
    annotations=None,
    prioritize=False,
    skip_roads=False,
    coord_manual=None
):

    source_file = (
        os.path.split(input_geotif)[-1] + "_" + str(tile_width) + "_" + str(tile_height)
    )

    OSM_BUILDING_PATH = os.path.join(out_folder, source_file, "building_mask")
    GEOJSON_BUILDING_PATH = os.path.join(
        out_folder, source_file, "building_mask"
    )
    building_mask_save_path = (
        OSM_BUILDING_PATH if building_polygon_source == "OSM" else GEOJSON_BUILDING_PATH
    )

    priority_path_building = os.path.join(
        out_folder, source_file, "building_mask_priority"
    )
    non_priority_path_building = os.path.join(
        out_folder, source_file, "building_mask_non_priority"
    )

    os.makedirs(tmp_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, source_file, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, source_file, "road_mask"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, source_file, "building_mask"), exist_ok=True)

    os.makedirs(OSM_BUILDING_PATH, exist_ok=True)
    os.makedirs(GEOJSON_BUILDING_PATH, exist_ok=True)

    os.makedirs(
        os.path.join(out_folder, source_file, "road_mask_priority"), exist_ok=True
    )
    os.makedirs(
        os.path.join(out_folder, source_file, "road_mask_non_priority"), exist_ok=True
    )
    os.makedirs(
        os.path.join(out_folder, source_file, "rgb_road_non_priority"), exist_ok=True
    )
    os.makedirs(
        os.path.join(out_folder, source_file, "rgb_road_priority"), exist_ok=True
    )
    os.makedirs(
        os.path.join(out_folder, source_file, "rgb_building_non_priority"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(out_folder, source_file, "rgb_building_priority"), exist_ok=True
    )
    os.makedirs(os.path.join(out_folder, source_file, "rgb_priority"), exist_ok=True)
    os.makedirs(
        os.path.join(out_folder, source_file, "rgb_non_priority"), exist_ok=True
    )
    os.makedirs(priority_path_building, exist_ok=True)
    os.makedirs(non_priority_path_building, exist_ok=True)

    if generate_debug_images:
        os.makedirs(os.path.join(out_folder, source_file, "debug"), exist_ok=True)

    # Load the ortho
    print("Loading external base geotif from:", input_geotif)
    input_data = rasterio.open(input_geotif, "r")
    print("Done...")

    x_steps = np.arange(0, input_data.width, tile_width / tile_width_overlap_ratio)
    y_steps = np.arange(0, input_data.height, tile_height / tile_height_overlap_ratio)

    count_images = 0
    # pylint: disable=too-many-nested-blocks
    with alive_bar(total=len(x_steps) * len(y_steps)) as alivebar:
        for x_step in x_steps:
            for y_step in y_steps:

                x_step = int(x_step)
                y_step = int(y_step)

                tmp_file = os.path.join(tmp_folder, "tmp.geo.tiff")
                crop_to_file(
                    input_data, tmp_file, x_step, y_step, tile_width, tile_height
                )
                cropped_data = rasterio.open(tmp_file, "r")

                # Reshape the image so it is in the correct format for image writing
                try:
                    color_data = np.stack(
                        [
                            cropped_data.read(1),
                            cropped_data.read(2),
                            cropped_data.read(3),
                        ],
                        axis=2,
                    )
                except IndexError:
                    color_data = np.stack(
                        [cropped_data.read(1)] * 3,
                        axis=2,
                    )

                # If there is any color data in the tile
                if np.sum(color_data) > 0:

                    # Get the OSM data for the ortho's bounding box
                    ortho_lat_lon_bounds = transform_bounds(cropped_data, LAT_LON_CRS)
                    osm_bbox = [
                        ortho_lat_lon_bounds.bottom,
                        ortho_lat_lon_bounds.left,
                        ortho_lat_lon_bounds.top,
                        ortho_lat_lon_bounds.right,
                    ]

                    road_and_building_data = {"buildings": [], "roads": []}
                    if not skip_roads and not building_polygon_source.upper() == "GEOJSON":
                        road_and_building_data = (
                            get_osm_roads_buildings_and_nodes_for_bounding_box(osm_bbox)
                        )

                    if building_polygon_source.upper() == "GEOJSON":
                        geojson_bbox = [
                            ortho_lat_lon_bounds.left,
                            ortho_lat_lon_bounds.bottom,
                            ortho_lat_lon_bounds.right,
                            ortho_lat_lon_bounds.top,
                        ]
                        road_and_building_data["buildings"] = (
                            get_geojson_buildings_for_bounding_box(
                                geojson_bbox, geojson_building_data
                            )["buildings"]
                        )

                    # Initialize a transform to convert from lat lon to the target crs of the ortho
                    crs_cropped = cropped_data.crs
                    if crs_cropped is None:
                        crs_cropped = CRS.from_epsg(coord_manual)
                    coord_transformer = Transformer.from_crs(
                        LAT_LON_CRS, crs_cropped.to_string()
                    )

                    file_prefix = (
                        source_file
                        + "_tile_("
                        + str(x_step)
                        + ","
                        + str(y_step)
                        + ")_"
                        + str(cropped_data.width)
                        + "x"
                        + str(cropped_data.height)
                        + "_"
                    )

                    # Save the image to disk
                    rgb_filename = file_prefix + "rgb.png"
                    im_color = Image.fromarray(color_data)
                    if not prioritize:
                        # Only Save RGB for all if not prioritizing to avoid duplicate tiles 
                        im_color.save(
                            os.path.join(out_folder, source_file, "rgb", rgb_filename)
                        )

                    if not skip_roads:
                        road_mask = Image.fromarray(np.zeros_like(color_data))
                        road_mask_draw = ImageDraw.Draw(road_mask)

                        road_polygons = draw_roads_on_ortho_img(
                            cropped_data,
                            road_and_building_data,
                            coord_transformer,
                            road_mask_draw,
                            color="white",
                            swap_xy=swap_lat_lon,
                        )
                        road_polylines = get_road_polylines_from_osm_data(
                            road_and_building_data,
                            cropped_data,
                            coord_transformer,
                            swap_xy=swap_lat_lon,
                        )

                        road_polygon_details = {
                            "source": rgb_filename,
                            "polygons": road_polygons,
                        }
                        road_polyline_details = {
                            "source": rgb_filename,
                            "polylines": road_polylines,
                        }

                        if mask_format in ("IMG","BOTH"):
                            road_mask.save(
                                os.path.join(
                                    out_folder,
                                    source_file,
                                    "road_mask",
                                    file_prefix + "road_mask.png",
                                )
                            )
                        if mask_format in ("POLY","BOTH"):
                            with open(
                                os.path.join(
                                    out_folder,
                                    source_file,
                                    "road_mask",
                                    file_prefix + "road_mask.json",
                                ),
                                "w",
                            ) as outfile:
                                outfile.write(json.dumps(road_polygon_details, indent=4))
                            with open(
                                os.path.join(
                                    out_folder,
                                    source_file,
                                    "road_mask",
                                    file_prefix + "road_lines.json",
                                ),
                                "w",
                            ) as outfile:
                                outfile.write(json.dumps(road_polyline_details, indent=4))

                    building_mask = Image.fromarray(np.zeros_like(color_data))
                    building_mask_draw = ImageDraw.Draw(building_mask)
                    if preannotated:
                        building_polygons, labels = draw_buildings_on_ortho_img(
                            cropped_data,
                            road_and_building_data,
                            coord_transformer,
                            building_mask_draw,
                            color="white",
                            swap_xy=swap_lat_lon,
                            preannotated=preannotated,
                            annotations=annotations,
                        )
                        building_polygon_details = {
                            "source": rgb_filename,
                            "polygons": building_polygons,
                            "labels": labels,
                        }
                    else:
                        building_polygons = draw_buildings_on_ortho_img(
                            cropped_data,
                            road_and_building_data,
                            coord_transformer,
                            building_mask_draw,
                            color="white",
                            swap_xy=swap_lat_lon,
                        )
                        building_polygon_details = {
                            "source": rgb_filename,
                            "polygons": building_polygons,
                        }
                    if mask_format in ("IMG","BOTH"):
                        building_mask.save(
                            os.path.join(
                                building_mask_save_path,
                                file_prefix
                                + "_"
                                + building_polygon_source
                                + "_"
                                + "building_mask.png",
                            )
                        )

                    # Remove potential duplicate buildings...
                    building_polygon_details = remove_duplicate_buildings(
                        building_polygon_details
                    )

                    if mask_format in ("POLY","BOTH"):
                        with open(
                            os.path.join(
                                building_mask_save_path,
                                file_prefix
                                + "_"
                                + building_polygon_source
                                + "_"
                                + "building_mask.json",
                            ),
                            "w",
                        ) as outfile:
                            outfile.write(
                                json.dumps(building_polygon_details, indent=4)
                            )

                    if generate_debug_images:
                        debug_image = Image.fromarray(color_data)
                        debug_image_draw = ImageDraw.Draw(debug_image)

                        color = (255, 0, 0, 125)
                        if not skip_roads:
                            draw_roads_on_ortho_img(
                                cropped_data,
                                road_and_building_data,
                                coord_transformer,
                                debug_image_draw,
                                color=color,
                                swap_xy=swap_lat_lon,
                            )
                        draw_buildings_on_ortho_img(
                            cropped_data,
                            road_and_building_data,
                            coord_transformer,
                            debug_image_draw,
                            color=color,
                            swap_xy=swap_lat_lon,
                        )

                        if not skip_roads:
                            get_mask_from_polygons(debug_image_draw, road_polygons, color)
                        get_mask_from_polygons(
                            debug_image_draw, building_polygons, color
                        )

                        debug_image.save(
                            os.path.join(
                                out_folder,
                                source_file,
                                "debug",
                                file_prefix + "debug.png",
                            )
                        )

                    # Priority Sorting
                    img = im_color.convert("L")
                    img_width, img_height = img.size
                    img_data = np.array(img)
                    number_black_pixels = np.count_nonzero(img_data == 0)

                    if prioritize:
                        preannotated_labels_change = True
                        if preannotated:
                            # Prioritize based on the preannotations
                            if (MINOR_DAMAGE in labels) or (MAJOR_DAMAGE in labels):
                                preannotated_labels_change = False
                        if (
                            (len(building_polygons) == 0)
                            or (len(building_polygons) >= BUILDING_POLYGON_LIMIT)
                            or (number_black_pixels >= (img_height * img_width * 0.125))
                            or preannotated_labels_change
                        ):
                            if not skip_roads:
                                if len(road_polygons) > 0:
                                    im_color.save(
                                        os.path.join(
                                            out_folder,
                                            source_file,
                                            "rgb_road_non_priority",
                                            rgb_filename,
                                        )
                                    )
                                    if mask_format in ("IMG", "BOTH"):
                                        road_mask.save(
                                            os.path.join(
                                                out_folder,
                                                source_file,
                                                "road_mask_non_priority",
                                                file_prefix + "road_mask.png",
                                            )
                                        )
                                    if mask_format in ("POLY", "BOTH"):
                                        with open(
                                            os.path.join(
                                                out_folder,
                                                source_file,
                                                "road_mask_non_priority",
                                                file_prefix + "road_mask.json",
                                            ),
                                            "w",
                                        ) as outfile:
                                            outfile.write(
                                                json.dumps(road_polygon_details, indent=4)
                                            )
                                        with open(
                                            os.path.join(
                                                out_folder,
                                                source_file,
                                                "road_mask_non_priority",
                                                file_prefix + "road_lines.json",
                                            ),
                                            "w",
                                        ) as outfile:
                                            outfile.write(
                                                json.dumps(road_polyline_details, indent=4)
                                            )
                            if len(building_polygons) > 0:
                                im_color.save(
                                    os.path.join(
                                        out_folder,
                                        source_file,
                                        "rgb_building_non_priority",
                                        rgb_filename,
                                    )
                                )
                                if mask_format in ("IMG","BOTH"):
                                    building_mask.save(
                                        os.path.join(
                                            non_priority_path_building,
                                            file_prefix
                                            + "_"
                                            + building_polygon_source
                                            + "_"
                                            + "building_mask.png",
                                        )
                                    )
                                if mask_format in ("POLY","BOTH"):
                                    with open(
                                        os.path.join(
                                            non_priority_path_building,
                                            file_prefix
                                            + "_"
                                            + building_polygon_source
                                            + "_"
                                            + "building_mask.json",
                                        ),
                                        "w",
                                    ) as outfile:
                                        outfile.write(
                                            json.dumps(
                                                building_polygon_details, indent=4
                                            )
                                        )

                            ####
                        else:
                            im_color.save(
                                os.path.join(
                                    out_folder,
                                    source_file,
                                    "rgb_building_priority",
                                    rgb_filename,
                                )
                            )
                            if len(road_polygons) > 0:
                                im_color.save(
                                    os.path.join(
                                        out_folder,
                                        source_file,
                                        "rgb_road_priority",
                                        rgb_filename,
                                    )
                                )

                                if not skip_roads:
                                    if mask_format in ("IMG","BOTH"):
                                        road_mask.save(
                                            os.path.join(
                                                out_folder,
                                                source_file,
                                                "road_mask_priority",
                                                file_prefix + "road_mask.png",
                                            )
                                        )
                                    if mask_format in ("POLY","BOTH"):
                                        with open(
                                            os.path.join(
                                                out_folder,
                                                source_file,
                                                "road_mask_priority",
                                                file_prefix + "road_mask.json",
                                            ),
                                            "w",
                                        ) as outfile:
                                            outfile.write(
                                                json.dumps(road_polygon_details, indent=4)
                                            )
                                        with open(
                                            os.path.join(
                                                out_folder,
                                                source_file,
                                                "road_mask_priority",
                                                file_prefix + "road_lines.json",
                                            ),
                                            "w",
                                        ) as outfile:
                                            outfile.write(
                                                json.dumps(road_polyline_details, indent=4)
                                            )

                            if mask_format in ("IMG","BOTH"):
                                building_mask.save(
                                    os.path.join(
                                        priority_path_building,
                                        file_prefix
                                        + "_"
                                        + building_polygon_source
                                        + "_"
                                        + "building_mask.png",
                                    )
                                )
                            if mask_format in ("POLY","BOTH"):
                                with open(
                                    os.path.join(
                                        priority_path_building,
                                        file_prefix
                                        + "_"
                                        + building_polygon_source
                                        + "_"
                                        + "building_mask.json",
                                    ),
                                    "w",
                                ) as outfile:
                                    outfile.write(
                                        json.dumps(building_polygon_details, indent=4)
                                    )
                            count_images += 1
                            ####

                cropped_data.close()
                # pylint: disable-next=not-callable
                alivebar()

        if save_tmp_folder:
            # pylint: disable-next=no-member
            os.rmtree(tmp_folder)

        print("Total Images Prioritized: ", count_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tile_and_generate_masks_all",
        description="This program extracts tile from geotifs and generates masks for roads and buildings based on open street map data.",
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        help="The width of the generated tiles in pixels.",
        default=2048,
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        help="The height of the generated tiles in pixels.",
        default=2048,
    )
    parser.add_argument(
        "--tile_width_overlap_ratio",
        type=float,
        help="The ratio of width-wise overlap between adjacent tiles.",
        default=1.05,
    )
    parser.add_argument(
        "--tile_height_overlap_ratio",
        type=float,
        help="The ratio of height-wise overlap between adjacent tiles.",
        default=1.05,
    )
    parser.add_argument(
        "--building_polygon_source",
        type=str,
        help="The option that defines what source should be used for building polygons (OSM, GEOJSON)",
        default="OSM",
    )
    parser.add_argument(
        "--geojson_building_polygon_path",
        type=str,
        help="The path to the geojson building polygon goejson, if using geojson building polygons",
    )
    parser.add_argument(
        "--tmp_folder",
        type=str,
        help="The path to temporary storage where intermediate outputs will be saved.",
        default="./tmp/",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        help="The path to the output location where the tiles will be saved.",
        default="./out/",
    )
    parser.add_argument(
        "--mask_format",
        type=str,
        help="The format that the masks will be saved in (IMG, POLY, BOTH)",
        default="BOTH",
    )
    parser.add_argument(
        "--input_geotif", type=str, nargs="+", help="The input geotif to be processed."
    )
    parser.add_argument("--generate_debug_images", action="store_true")
    parser.add_argument("--prioritize", action="store_true")
    parser.add_argument("--save_tmp_folder", action="store_true")
    parser.add_argument("--swap_ortho_xy", action="store_true")
    parser.add_argument("--preannotated", action="store_true")
    parser.add_argument(
        "--annotation_file",
        type=str,
        help="The path to the preannoatation json file for the ortho.",
    )
    parser.add_argument("--skip_roads", action="store_true")
    parser.add_argument("--coords_manual", type=int, default=None, help="Specifiy the CRS manually if needed. Note: PEMA data is 2271")
    args = parser.parse_args()

    file_list = (
        str(args.input_geotif)
        .replace(",", "")
        .replace("[", "")
        .replace("]", "")
        .split()
    )
    geojson_building_data_1 = None
    if args.building_polygon_source.upper() == "GEOJSON":
        print(
            "Loading external building polygons from:",
            args.geojson_building_polygon_path,
        )
        geojson_building_data_1 = load_buildings_from_geojson(
            args.geojson_building_polygon_path
        )
        print("Done...")

    annotation_data = []
    if args.preannotated:
        with open(args.annotation_file, "r") as f:
            annotation_data = json.loads(f.read())

    for tif_file in file_list:
        tif_file = tif_file.replace("'", "")
        image_tile_generate(
            tif_file,
            args.tile_width,
            args.tile_height,
            args.out_folder,
            args.building_polygon_source,
            args.tmp_folder,
            args.generate_debug_images,
            args.tile_width_overlap_ratio,
            args.tile_height_overlap_ratio,
            args.mask_format,
            args.save_tmp_folder,
            geojson_building_data_1,
            args.swap_ortho_xy,
            args.preannotated,
            annotation_data,
            args.prioritize,
            args.skip_roads,
            args.coords_manual,
        )
