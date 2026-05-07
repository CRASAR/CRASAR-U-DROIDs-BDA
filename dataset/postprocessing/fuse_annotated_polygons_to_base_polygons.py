
import os

from shapely.geometry import Polygon
from pyproj import Transformer
import argparse
from alive_progress import alive_bar

import json
import numpy as np
import rasterio


from dataset.constants import LAT_LON_CRS, UNCLASSIFIED
from dataset.utils.ortho_utils import transform_bounds
from dataset.generators.geojson_building_polygon_data_generator import (
    load_buildings_from_geojson,
    get_geojson_buildings_for_bounding_box,
)
from dataset.postprocessing.reconstruct_from_BDA_annotations import (
    cascade_fuse_polygons,
)

def rebase_annotated_polygons(
    annotations_path_map,
    geotif_path_map,
    target_geotif,
    geojson_building_data,
    output_folder_path,
    output_path_map,
    swap_xy=False,
):
    print("Working", target_geotif)
    print("\tLoading geotiff path map")
    input_geotif_path = json.load(open(geotif_path_map))[target_geotif]

    print("\tLoading annotations")
    annotations_path = json.load(open(annotations_path_map))[input_geotif_path]
    f = open(annotations_path, "r")
    annotated_buildings = json.loads(f.read())
    f.close()

    print("\tLoading external base geotif from:", input_geotif_path)
    input_geotiff_data = rasterio.open(input_geotif_path, "r")

    coord_system = str(input_geotiff_data.crs)
    print("\tGenerating transformer between:", coord_system, "and", LAT_LON_CRS)
    coord_transformer = Transformer.from_crs(LAT_LON_CRS, coord_system)

    ortho_lat_lon_bounds = transform_bounds(input_geotiff_data, LAT_LON_CRS)
    geojson_bbox = [
        ortho_lat_lon_bounds.left,
        ortho_lat_lon_bounds.bottom,
        ortho_lat_lon_bounds.right,
        ortho_lat_lon_bounds.top,
    ]

    buildings = get_geojson_buildings_for_bounding_box(
        geojson_bbox, geojson_building_data
    )["buildings"]
    unannotated_polygons = []
    for base_building in buildings:
        unannotated_polygons.append(
            Polygon([[p["lat"], p["lon"]] for p in base_building["geometry"]])
        )

    print("Merging down existing polygons...")
    fused_polygons = cascade_fuse_polygons(
        [[UNCLASSIFIED, p, "DONE"] for p in unannotated_polygons]
    )
    unannotated_polygons_fused = [x[1] for x in fused_polygons]

    print("Finding matches between annotated and unannotated polygons")
    matched_base_polygon_indicies = []
    with alive_bar(total=len(annotated_buildings)) as bar:
        for annotated_building in annotated_buildings:
            if swap_xy:
                coords = [[p["lon"], p["lat"]] for p in annotated_building["EPSG:4326"]]
            else:
                coords = [[p["lat"], p["lon"]] for p in annotated_building["EPSG:4326"]]
            ab_polygon = Polygon(coords)
            ious = []

            for base_polygon in unannotated_polygons_fused:
                ious.append(
                    ab_polygon.intersection(base_polygon).area
                    / ab_polygon.union(base_polygon).area
                )

            max_iou = max(ious)

            if max_iou == 0:
                matched_base_polygon_indicies.append(-1)
            else:
                matched_base_polygon_indicies.append(np.argmax(ious))

            if max_iou <= 0.25:
                print("WARNING: Match found, but intersection/union was", max_iou)
                print("\t\tAnnotated:", ab_polygon)
                print("\t\tBase:", unannotated_polygons_fused[np.argmax(ious)])
                print("\t\tLabel:", annotated_building["label"])
            bar()

    print("Merging down paired annotated polygons")
    rebased_annotation_pairs = []
    for annotation_idx, base_building_idx in enumerate(matched_base_polygon_indicies):
        if base_building_idx == -1:
            label = annotated_buildings[annotation_idx]["label"]
            inital_polygon = Polygon(
                [
                    [p["lat"], p["lon"]]
                    for p in annotated_buildings[annotation_idx]["EPSG:4326"]
                ]
            )
            rebased_annotation_pairs.append([label, inital_polygon])
        else:
            label = annotated_buildings[annotation_idx]["label"]
            rebased_annotation_pairs.append(
                [label, unannotated_polygons_fused[base_building_idx]]
            )
    rebased_fused_polygons = cascade_fuse_polygons(rebased_annotation_pairs)

    individual_polygons = []
    for label, polygon in rebased_fused_polygons:
        if polygon.geom_type == "MultiPolygon":
            individual_polygons.extend([[label, p] for p in polygon.geoms])
        elif polygon.geom_type == "Polygon":
            individual_polygons.append([label, polygon])

    print("Rebasing Annotations")
    rebased_annotations = []
    for label, polygon in individual_polygons:
        rebased_annotation = {}
        coords = []
        pixels = []
        # Since we are working with lats and lons in the polygon, we need to convert and get pixels
        for p in polygon.exterior.coords:
            coords.append({"lat": p[0], "lon": p[1]})
            x_source, y_souce = coord_transformer.transform(p[0], p[1])
            y_p, x_p = rasterio.transform.rowcol(
                input_geotiff_data.transform, x_source, y_souce
            )
            pixels.append({"x": x_p, "y": y_p})
        rebased_annotation["pixels"] = pixels
        rebased_annotation["EPSG:4326"] = coords
        rebased_annotation["label"] = label
        rebased_annotations.append(rebased_annotation)

    print("\tWriting rebased annotations...")
    path = os.path.join(output_folder_path, target_geotif + ".json")
    f = open(path, "w")
    f.write(json.dumps(rebased_annotations))
    f.close()
    output_path_map[input_geotif_path] = path
    print("\tDone")


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
        "--base_polygons_folder_path",
        type=str,
        help="The path to the unannotated base polygons folder",
    )
    parser.add_argument(
        "--output_annotations_folder",
        type=str,
        help="The path to the output annotations folder updated polygons applied.",
    )
    parser.add_argument(
        "--output_annotations_path_map",
        type=str,
        help="The path map for the rebased annotations.",
    )
    args = parser.parse_args()

    resulting_path_map = {}

    print("\tLoading base polygons")
    ian_fmc_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_ian_fmc.geojson"))
    ian_fmb_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_ian_fmb.geojson"))
    ian_boca_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_ian_boca_grande.geojson"))
    mussett_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "mussett_bayou_fire.geojson"))
    mayfield_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "mayfield_tornado.geojson"))
    russellville_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "russellville_tornado.geojson"))
    kilauea_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "kilauea_eruption.geojson"))
    michael_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_michael.geojson"))
    ida_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_ida.geojson"))
    idalia_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_idalia.geojson"))
    laura_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_laura.geojson"))
    harvey_building_data = load_buildings_from_geojson(os.path.join(args.base_polygons_folder_path, "hurricane_harvey.geojson"))
    harvey_sienna_building_data = load_buildings_from_geojson(
        os.path.join(args.base_polygons_folder_path, "Texas.geojson")
    )

    # Ian Boca Grande
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.1.geo.tif", ian_boca_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.2.geo.tif", ian_boca_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.3.geo.tif", ian_boca_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.4.geo.tif", ian_boca_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.5.geo.tif", ian_boca_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.6.geo.tif", ian_boca_building_data, args.output_annotations_folder, resulting_path_map)

    # Ian Fort Myers
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.1.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.2.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.3.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.4.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Sanibel-Causeway-North.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Summerlin-San-Carlos.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Palmeto-Palms.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-McGregor-College-Pkwy-South.3.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-McGregor-College-Pkwy-South.2.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-McGregor-College-Pkwy-South.1.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Kennedy-Green-Mobile-Homes.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Iona-Point.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Harlem-Heights.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Kelly-Road.geo.tif", ian_fmc_building_data, args.output_annotations_folder, resulting_path_map)

    # Ian Fort Myers Beach
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-San-Carlos-Island.geo.tif", ian_fmb_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Ft-Myers-Beach-DIRT.geo.tif", ian_fmb_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1001-Ft-Myers-Beach-Boone.geo.tif", ian_fmb_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Ft-Myers-Beach-LCSO.geo.tif", ian_fmb_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "1002-Ft-Myers-Beach-TFD.geo.tif", ian_fmb_building_data, args.output_annotations_folder, resulting_path_map)

    # Laura
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "0827-A-01.geo.tif", laura_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "0827-B-02.geo.tif", laura_building_data, args.output_annotations_folder, resulting_path_map)

    # Mussett
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif", mussett_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif", mussett_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif", mussett_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-NorthOf98.geo.tif", mussett_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-01.geo.tif", mussett_building_data, args.output_annotations_folder, resulting_path_map)

    # Mayfield
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20211214-Mayfield.geo.tif", mayfield_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20211213-Candle-Factory-AO.geo.tif", mayfield_building_data, args.output_annotations_folder, resulting_path_map)

    # Russellville
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20211215-Russelville-Middle.geo.tif", russellville_building_data, args.output_annotations_folder, resulting_path_map)

    # Kilauea
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "2018-05-18-X5-visible-Kahukai.geo.tif", kilauea_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "2018-05-18-X5-visible-Geothermal.geo.tif", kilauea_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "2018-05-18-X4S-visible-CentralPark.geo.tif", kilauea_building_data, args.output_annotations_folder, resulting_path_map)

    # Michael
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "10132018-MexicoBeach.geo.tif", michael_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "10142018-MexicoBeach.geo.tif", michael_building_data, args.output_annotations_folder, resulting_path_map)

    # Idalia
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20230831-Jena-SteinhatcheeRiverSouth.geo.tif", idalia_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20230830-SteinhatcheeRiver.geo.tif", idalia_building_data, args.output_annotations_folder, resulting_path_map)

    # Ida
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20210901-Cocodrie-2.geo.tif", ida_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20210901-Cocodrie-3.geo.tif", ida_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20210901-Cocodrie-1.geo.tif", ida_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20210902-LA-DIV-01.geo.tif", ida_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "20210831-LA-DIV-01.geo.tif", ida_building_data, args.output_annotations_folder, resulting_path_map)

    # Harvey
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "090302-Pecan-Grove-Levee.geo.tif", harvey_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(
        args.annotations_path_map,
        args.geotif_path_map,
        "090402-DMS-Assessment-Sienna-Village.geo.tif",
        harvey_sienna_building_data,
        args.output_annotations_folder,
        resulting_path_map,
        swap_xy=False,
    )
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "090401-DMS-Assessment-Westpark.geo.tif", harvey_building_data, args.output_annotations_folder, resulting_path_map)
    rebase_annotated_polygons(args.annotations_path_map, args.geotif_path_map, "090403-Lancaster-Canyon-Gate.geo.tif", harvey_building_data, args.output_annotations_folder, resulting_path_map)

    print("Writing Output Path Map...")
    f = open(args.output_annotations_path_map, "w")
    f.write(json.dumps(resulting_path_map))
    f.close()
