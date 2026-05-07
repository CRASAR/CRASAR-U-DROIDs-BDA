import os
import json
import rasterio
import argparse

from shapely.geometry import Polygon
from collections import defaultdict
from rasterio.windows import Window
from pyproj import Transformer
from dataset.constants import LAT_LON_CRS

from PIL import Image, ImageDraw

from alive_progress import alive_bar

from dataset.constants import BDA_DAMAGE_CLASSES, BDA_CATEGORY_COLOR_MAP
from dataset.utils.adjustment_utils import (
    apply_adjustments,
    match_polygon_to_adjustment,
)

ORTHO_VIEWS_TO_IGNORE = [
    "103001008699D200",
    "1030010087C1A800",
    "1050010011549F00",
    "105001001292E300",
    "1001-Harlem-Heights.geo.tif_20220929a_RGB.geo.tif",
    "1001-Iona-Point.geo.tif_20220930d_RGB.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif_20220930d_RGB.geo.tif",
    "1002-Boca-Grande.1.geo.tif_20221001b_RGB.geo.tif",
    # Channels = 1 ... Pre-Disaster does not need group review
    "1001-Iona-Point.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif_1040010077338D00-visual.tif.geo.tif",
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif_105001000C7F2100-visual.tif.geo.tif",
    "20230831-Jena-SteinhatcheeRiverSouth.geo.tif_1040010061205400-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.2.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.2.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.1.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.1.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1002-Sanibel-Causeway-North.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.2.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1001-McGregor-College-Pkwy-South.1.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "20230830-SteinhatcheeRiver.geo.tif_105001000C7F2100-visual.tif.geo.tif",
    "20230830-SteinhatcheeRiver.geo.tif_1040010077338D00-visual.tif.geo.tif",
    "20230830-SteinhatcheeRiver.geo.tif_1040010061205400-visual.tif.geo.tif",
    "10142018-MexicoBeach.geo.tif_10400100291A5A00.tif.geo.tif",
    "10132018-MexicoBeach.geo.tif_10400100291A5A00.tif.geo.tif",
    "1002-Palm-Acers.4.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Palm-Acers.4.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1002-Palm-Acers.4.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1002-Palm-Acers.3.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Palm-Acers.3.geo.tif_10200100BC21AC00-visual.tif.geo.tif",
    "1002-Palm-Acers.3.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1002-Palm-Acers.1.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1002-Palm-Acers.1.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1002-Boca-Grande.3.geo.tif_1040010072528400-visual.tif.geo.tif",
    "1002-Sanibel-Causeway-North.geo.tif_10200100BC21AC00-visual.tif.geo.tif",
    "1002-Sanibel-Causeway-North.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1001-Ft-Myers-Beach-DIRT.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-Ft-Myers-Beach-Boone.geo.tif_10200100B5D89900-visual.tif.geo.tif",
    "1001-Ft-Myers-Beach-Boone.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-Iona-Point.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-Iona-Point.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1001-Harlem-Heights.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1001-Harlem-Heights.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1001-Harlem-Heights.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1001-Ft-Myers-Beach-DIRT.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1001-Ft-Myers-Beach-Boone.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Palm-Acers.2.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Palm-Acers.2.geo.tif_10200100BC21AC00-visual.tif.geo.tif",
    "1002-Palm-Acers.2.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1002-Palm-Acers.1.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Kelly-Road.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Kelly-Road.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1002-Kelly-Road.geo.tif_10200100B01E7400-visual.tif.geo.tif",
    "1002-Ft-Myers-Beach-TFD.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "1002-Ft-Myers-Beach-TFD.geo.tif_10200100B5D89900-visual.tif.geo.tif",
    "1002-Ft-Myers-Beach-TFD.geo.tif_10200100B0214600-visual.tif.geo.tif",
    "1002-Ft-Myers-Beach-LCSO.geo.tif_10400100703E5500-visual.tif",
    "1002-Ft-Myers-Beach-LCSO.geo.tif_10300100DA368900-visual.tif.geo.tif",
    "1002-Ft-Myers-Beach-LCSO.geo.tif_10200100B0214600-visual.tif",
    "105001002D409800",
    "105001002A491900",
    "1040010072528400",
    "10200100C0DA0700",
    "10400100703E5500",
    "10200100B0214600",
    "10200100B01E7400",
]


def link_polygons_by_id(annotations_path_map):
    polygons_by_id = defaultdict(lambda: [])
    for filename, annotation_path in annotations_path_map.items():
        print("Loading", annotation_path)
        f = open(annotation_path, "r")
        polygon_data = json.loads(f.read())
        f.close()

        for p in polygon_data:
            p["filename"] = filename
            polygons_by_id[p["id"]].append(p)
    return polygons_by_id


def make_review_card(
    buildings,
    geotif_path_map,
    card_path,
    read_buffer_relative_size=0.5,
    write=True,
    swap_xy=False,
    adjust=False,
    adjustments=None,
):
    metadata = []

    images = []
    masks = []
    valid_buildings = []
    for building in buildings:
        imagery_path = building["filename"]

        valid = True
        for key in ORTHO_VIEWS_TO_IGNORE:
            if key in imagery_path:
                valid = False

        if valid:
            adjustments_data = []
            if adjust:
                try:
                    annotation_path = adjustments[imagery_path]
                    with open(annotation_path, "r") as f:
                        adjustments_data = json.loads(f.read())
                except KeyError:
                    print(
                        "KeyError when loading adjustments, proceeding without them...",
                        imagery_path,
                    )

            if (
                "MexicoBeach" in imagery_path
                or "Pecan-Grove" in imagery_path
                or swap_xy
            ):
                input_geotiff_data = rasterio.open(imagery_path, "r")
                coord_system = str(input_geotiff_data.crs)
                coord_transformer = Transformer.from_crs(LAT_LON_CRS, coord_system)
                coords = []
                pixels = []

                # Adjust the points...
                for p in building["EPSG:4326"]:
                    coords.append({"lat": p["lat"], "lon": p["lon"]})
                    x_source, y_souce = coord_transformer.transform(p["lon"], p["lat"])
                    y_p, x_p = rasterio.transform.rowcol(
                        input_geotiff_data.transform, x_source, y_souce
                    )
                    pixels.append({"x": x_p, "y": y_p})
                polygon = Polygon([[p["x"], p["y"]] for p in pixels])
            else:
                polygon = Polygon([[p["x"], p["y"]] for p in building["pixels"]])
                pixels = building["pixels"]

            if adjust:
                best_adjustment = match_polygon_to_adjustment(adjustments_data, pixels)
                if best_adjustment is None:
                    best_adjustment_options = []
                else:
                    best_adjustment_options = [best_adjustment]

                adjusted_pixels = []
                for point in pixels:
                    x, y = apply_adjustments(
                        best_adjustment_options, point["x"], point["y"]
                    )
                    adjusted_pixels.append({"x": x, "y": y})

                polygon = Polygon([[p["x"], p["y"]] for p in adjusted_pixels])
                pixels = adjusted_pixels

            minx, miny, maxx, maxy = polygon.bounds

            dx = maxx - minx
            dy = maxy - miny
            maxd = max(dx, dy)
            maxd_buffer = read_buffer_relative_size * maxd

            if (
                "MexicoBeach" in imagery_path
                or "Pecan-Grove" in imagery_path
                or swap_xy
            ):
                offset_polygon = Polygon(
                    [
                        [(p["x"] - minx) + maxd_buffer, (p["y"] - miny) + maxd_buffer]
                        for p in pixels
                    ]
                )
            else:
                offset_polygon = Polygon(
                    [
                        [(p["x"] - minx) + maxd_buffer, (p["y"] - miny) + maxd_buffer]
                        for p in pixels
                    ]
                )

            with rasterio.open(imagery_path) as src:
                # Define the window you want to read (col_off, row_off, width, height)
                window = Window(
                    minx - maxd_buffer,
                    miny - maxd_buffer,
                    dx + maxd_buffer * 2,
                    dy + maxd_buffer * 2,
                )

                # Read the data within the specified window
                data = src.read((1, 2, 3), window=window)
                data_transposed = data.transpose(1, 2, 0)

            images.append(Image.fromarray(data_transposed.astype("uint8"), "RGB"))

            mask = Image.new(
                "RGB", (int(dx + maxd_buffer * 2), int(dy + maxd_buffer * 2)), "black"
            )
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(
                offset_polygon.exterior.coords,
                fill=tuple(BDA_CATEGORY_COLOR_MAP[building["label"]]),
            )
            masks.append(mask)
            valid_buildings.append(building)

    if len(images) > 0:
        imagery_total_width = sum([im.width for im in images])
        max_height = max([im.height for im in images])

        edge_buffer = 10
        label_region_width = images[0].width

        id_label_height = 20
        label_description_height = 40

        label_volume_height = 40

        width = edge_buffer + label_region_width + imagery_total_width + edge_buffer
        height = (
            edge_buffer
            + id_label_height
            + max_height * 2
            + label_description_height
            + label_volume_height * 6
            + edge_buffer
        )

        new_image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(new_image)
        x_pos = edge_buffer + label_region_width
        for building, im, mask in zip(valid_buildings, images, masks):
            new_image.paste(im, (x_pos, edge_buffer + id_label_height))
            new_image.paste(mask, (x_pos, edge_buffer + max_height + id_label_height))
            label_message = (
                building["label"].upper() + "\n"
            )  # + building["status"].lower()

            # Draw the label description text
            _, _, w, h = draw.textbbox((0, 0), label_message)
            draw.text(
                (
                    (im.width - w) / 2 + x_pos,
                    (label_description_height - h) / 2
                    + edge_buffer
                    + max_height * 2
                    + id_label_height,
                ),
                label_message,
                fill=(0, 0, 0),
            )

            # Draw the annotation volumes
            shape_meta = {}
            for i in range(0, len(BDA_DAMAGE_CLASSES)):
                cur_y_pos = (
                    edge_buffer
                    + id_label_height
                    + max_height * 2
                    + label_description_height
                    + label_volume_height * i
                )
                shape = [
                    x_pos,
                    cur_y_pos,
                    x_pos + im.width,
                    cur_y_pos + label_volume_height,
                ]
                draw.rectangle(shape, fill="white", outline="black")

                bda_label = str(BDA_DAMAGE_CLASSES[i])
                _, _, w, h = draw.textbbox((0, 0), bda_label)
                draw.text(
                    (
                        edge_buffer + (label_region_width - w) / 2,
                        (label_volume_height - h) / 2 + cur_y_pos,
                    ),
                    bda_label,
                    fill=(0, 0, 0),
                )
                shape_meta[bda_label] = shape
            x_pos += im.width
            metadata.append([building, shape_meta])

        id_message = "ID:    " + valid_buildings[0]["id"]
        _, _, w, h = draw.textbbox((0, 0), id_message)
        draw.text(
            ((width - w) / 2, (id_label_height - h) / 2 + edge_buffer),
            id_message,
            fill=(0, 0, 0),
        )

        if write:
            new_image.save(card_path)

    return metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="rebase_annotated_polygons",
        description="This program takes the annotated building polygons and fuses them with the unannotated initial polygons that were used to seed the annotation. As a result, this handles any failed tiles during the initial annotation process.",
    )
    parser.add_argument(
        "--annotations_path_map",
        type=str,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--geotif_path_map", type=str, help="The input geotif path map."
    )
    parser.add_argument(
        "--output_review_cards_folder",
        type=str,
        help="The path to the folder where the review cards will be saved.",
    )
    parser.add_argument(
        "--output_review_cards_metadata_path",
        type=str,
        help="The path to the folder where the review card metadata will be saved.",
    )
    parser.add_argument(
        "--completed_reviews_folder",
        type=str,
        help="The path to the folder where the completed reviews are saved. If this field is set, then review cards for the cards already in the completed folder will not be completed.",
    )
    parser.add_argument(
        "--disagree_only",
        action="store_true",
        help="Only generate cards for samples where there is a disagreement in the labels.",
    )
    parser.add_argument(
        "--swap_xy",
        action="store_true",
        help="Swap the xy axes of the building polygons.",
    )
    parser.add_argument(
        "--adjustments_path_map",
        type=str,
        help="The path to the adjustments file path map.",
    )
    parser.add_argument(
        "--adjust_card",
        action="store_true",
        help="Adjust the buildings within the review card.",
    )
    args = parser.parse_args()

    f = open(args.annotations_path_map, "r")
    annotations_path_map_raw = json.loads(f.read())
    f.close()

    annotations_path_map = {}
    for filename, annotation_path in annotations_path_map_raw.items():
        valid = True
        for key in ORTHO_VIEWS_TO_IGNORE:
            if key in filename:
                print("Skipping ortho view: ", filename)
                valid = False
        if valid:
            annotations_path_map[filename] = annotation_path

    f = open(args.geotif_path_map, "r")
    geotif_path_map = json.loads(f.read())
    f.close()

    f = open(args.adjustments_path_map, "r")
    adjustments_path_map = json.loads(f.read())
    f.close()

    all_polygons = link_polygons_by_id(annotations_path_map)
    skipped_samples = 0

    all_meta = {}

    with alive_bar(total=len(all_polygons)) as bar:
        for key, buildings in all_polygons.items():

            all_done = True

            parent_ortho_filename = (
                os.path.split(buildings[0]["filename"])[-1].split(".tif_")[0] + ".tif"
            )

            card_filename = key + ".png"
            card_metadata_filename = key + ".json"

            image_folder_path = os.path.join(
                args.output_review_cards_folder, parent_ortho_filename
            )
            image_path = os.path.join(image_folder_path, card_filename)

            include_sample = (
                (not args.disagree_only)
                or any(b["label"] != buildings[0]["label"] for b in buildings)
                or len(buildings) == 1
            )

            if include_sample and not (args.completed_reviews_folder is None):
                candidate_completed_existing_review_card_path = os.path.join(
                    args.completed_reviews_folder,
                    "multiview_review",
                    parent_ortho_filename,
                    card_filename,
                )

                # If either the card or the metadata doesnt exist, then its a valid sample
                include_sample = not os.path.exists(
                    candidate_completed_existing_review_card_path
                )

                if not include_sample:
                    skipped_samples += 1
                    print(
                        "Skipping, a reviewed file already exists...",
                        candidate_completed_existing_review_card_path,
                    )

            valid = all_done and include_sample

            if valid:
                os.makedirs(image_folder_path, exist_ok=True)

            card_meta = make_review_card(
                buildings,
                geotif_path_map,
                card_path=image_path,
                write=valid,
                swap_xy=args.swap_xy,
                adjust=args.adjust_card,
                adjustments=adjustments_path_map,
            )
            all_meta[key] = card_meta

            bar()

    f = open(os.path.join(args.output_review_cards_metadata_path), "w")
    f.write(json.dumps(all_meta))
    f.close()
    print("Total samples:", len(all_polygons), "Skipped samples:", skipped_samples)
