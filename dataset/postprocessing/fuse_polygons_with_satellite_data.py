import os
import json
import argparse

from collections import defaultdict

import shapely
from pyproj import Transformer

LAT_LON_CRS = "EPSG:4326"

def load_bounds_from_maxar_meta(metadata_folder, initial_meta_boundaries=None):
    if initial_meta_boundaries:
        meta_boundaries = initial_meta_boundaries
    else:
        meta_boundaries = defaultdict(list)

    # Get all the files we need to pull bounds for
    paths = []
    for path_root, _, files in os.walk(metadata_folder):
        for file in files:
            if ".json" in file:
                paths.append(os.path.join(path_root, file))

    # Parse the paths
    for file in paths:
        print("Parsing:", file)
        filename_quadkey = os.path.split(file)[-1].split(".")[0]

        with open(file, "r") as f_metadata:
            meta_data = json.loads(f_metadata.read())

        coord_transformer = Transformer.from_crs(meta_data["properties"]["proj:epsg"], LAT_LON_CRS)

        if len(meta_data["properties"]["proj:geometry"]["coordinates"]) == 1:
            transformed_coords = []
            for x_source, y_source in meta_data["properties"]["proj:geometry"]["coordinates"][0]:
                x_t, y_t = coord_transformer.transform(x_source, y_source)
                transformed_coords.append([x_t, y_t])

            bounds = shapely.Polygon(transformed_coords)
        else:
            polygons = []
            for shape in meta_data["properties"]["proj:geometry"]["coordinates"]:
                sub_polygon_coords = []
                for x_source, y_source in shape[0]:
                    x_t, y_t = coord_transformer.transform(x_source, y_source)
                    sub_polygon_coords.append([x_t, y_t])
                polygons.append(shapely.Polygon(sub_polygon_coords))

            bounds = shapely.MultiPolygon(polygons)

        meta_boundaries[filename_quadkey].append([bounds, meta_data])
    return meta_boundaries

if __name__ == "__main__":
    program_description = 'This program takes a set of tiles, and a set of links to those tiles, and then downlaods the json metadata from MAXAR.'
    parser = argparse.ArgumentParser(prog='download_maxar_tile_info', description=program_description)
    parser.add_argument('--annotations_path_map', type=str, help="The path map to the annotated building polygons.")
    parser.add_argument('--metadata_folders', action="extend", nargs="+", type=str, help="The path to the folder where the metadata jsons will be written.")
    parser.add_argument('--output_building_polygons_folder', type=str, help="The path to the folder where the output building polygons will be saved.")
    parser.add_argument('--output_annotation_path_map', type=str, help="The path map that will be generated.")
    args = parser.parse_args()

    os.makedirs(args.output_building_polygons_folder, exist_ok=True)

    boundary_meta_by_quadkey = defaultdict(list)
    for folder in args.metadata_folders:
        boundary_meta_by_quadkey = load_bounds_from_maxar_meta(folder, boundary_meta_by_quadkey)

    with open(args.annotations_path_map, "r") as f_read_path_map:
        annotation_path_map = json.loads(f_read_path_map.read())

    output_annotation_path_map = {}

    for _, path in annotation_path_map.items():
        output_path = path
        if not path is None:
            filename = os.path.split(path)[-1]
            output_annotation_path_map[filename] = output_path
            with open(path, "r") as f_read_buildings:
                buildings = json.loads(f_read_buildings.read())

            satellite_key = None

            for quadkey in boundary_meta_by_quadkey.keys():
                if quadkey in filename:
                    satellite_key = quadkey

            if satellite_key:
                for i, building in enumerate(buildings):
                    if not building is None:
                        building_polygon = shapely.Polygon([(p["lat"], p["lon"]) for p in building["EPSG:4326"]])

                        max_intersection_area = 0
                        max_intersection_meta = None

                        for tile_poly, meta in boundary_meta_by_quadkey[satellite_key]:
                            intersection_area = tile_poly.intersection(building_polygon).area
                            if intersection_area > max_intersection_area:
                                max_intersection_area = intersection_area
                                max_intersection_meta = meta
                        if max_intersection_area:
                            date, time = max_intersection_meta["properties"]["datetime"].split(" ")
                            building["view_properties"] = {}
                            building["view_properties"]["date"] = date
                            building["view_properties"]["time"] = time
                            building["view_properties"]["provider"] = "Maxar"
                            building["view_properties"]["platform"] = max_intersection_meta["properties"]["platform"]
                            building["view_properties"]["off_nadir"] = max_intersection_meta["properties"]["view:off_nadir"]
                            building["view_properties"]["azimuth"] = max_intersection_meta["properties"]["view:azimuth"]
                            building["view_properties"]["incidence_angle"] = max_intersection_meta["properties"]["view:incidence_angle"]
                            building["view_properties"]["sun_azimuth"] = max_intersection_meta["properties"]["view:sun_azimuth"]
                            building["view_properties"]["sun_elevation"] = max_intersection_meta["properties"]["view:sun_elevation"]
                            buildings[i] = building

                output_path = os.path.join(args.output_building_polygons_folder, filename)
                output_annotation_path_map[filename] = output_path
                print("Writing", output_path)
                with open(output_path, "w") as f_write_buildings:
                    f_write_buildings.write(json.dumps(buildings, indent=4))
            else:
                print("Could not find quadkey for", filename)

    root, _ = os.path.split(args.output_annotation_path_map)
    os.makedirs(root, exist_ok=True)

    with open(args.output_annotation_path_map, "w") as f_output_path_map:
        f_output_path_map.write(json.dumps(output_annotation_path_map))
