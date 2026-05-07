import os
import json
import argparse
import rasterio
import geopandas as gpd

from shapely.geometry import Polygon
from pyproj import Transformer

from dataset.constants import RDA_LABELBOX_CLASSES, PASSABLE_WITH_DIFFICULTY_FLOODING, PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS, \
                              PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION, NOT_ABLE_TO_DETERMINE, NOT_PASSABLE_DESTRUCTION, \
                              NOT_PASSABLE_FLOODING, NOT_PASSABLE_OBSTRUCTIONS, LAT_LON_CRS, PARTICULATE_PARTIAL, \
                              PARTICULATE_TOTAL, ROAD_LINE


def convert_labeled_polygons_to_json_xy_and_lat_lon(labeled_polygons, geotiff_data):
    result = []
    transform = geotiff_data.transform
    coord_transformer = Transformer.from_crs("EPSG:4326", geotiff_data.crs.to_string())
    for label, polygon in labeled_polygons:
        individual_polys = []
        if polygon.geom_type == "MultiPolygon":
            individual_polys = list(polygon.geoms)
        else:
            individual_polys = [polygon]

        for p in individual_polys:
            pixel_coords_polygon = []
            target_crs_polygon = []

            for lon, lat in list(zip(*p.exterior.coords.xy)):

                target_crs_polygon.append({"lat": lat, "lon": lon})

                #Flip the y and x axis to align the data correctly in the coordinate space
                src_x, src_y = coord_transformer.transform(lat, lon)
                x_t, y_t = ~transform * (src_x, src_y)

                pixel_coords_polygon.append({"x": x_t, "y": y_t})


        result.append({"source": "Annotators",
                        "label":label,
                       "pixels":pixel_coords_polygon,
                       LAT_LON_CRS:target_crs_polygon})
    return result

def convert_road_lines_to_json_xy_and_lat_lon(road_lines, geotiff_data):
    result = []
    transform = geotiff_data.transform
    coord_transformer = Transformer.from_crs(LAT_LON_CRS, geotiff_data.crs.to_string())
    for label, road_line in road_lines:
        pixel_coord_line = []
        target_crs_line = []
        for lon, lat in list(zip(*road_line.coords.xy)):

            target_crs_line.append({"lat": lat, "lon": lon})

            src_x, src_y = coord_transformer.transform(lat, lon)
            x_t, y_t = ~transform * (src_x, src_y)

            pixel_coord_line.append({"x": x_t, "y": y_t})

        result.append({"source": "custom",
                    "label":label,
                    "pixels":pixel_coord_line,
                    LAT_LON_CRS:target_crs_line})


    return result


def parse_rda_labels(spot_checks_gdf):
    for index, row in spot_checks_gdf.iterrows():
        label_name = str(row["name"]).lower()
        if label_name == "clear" or label_name == "del":
            spot_checks_gdf.at[index, 'name'] = "del"
        elif label_name == "passable-with-difficulty (obstructions)" or \
             label_name == "passable-with-difficulty (obstruction)" or \
             label_name == "passable with difficulty (obstruction)" or \
             label_name == "passable with difficulty (obstructions)" or \
             label_name == "pdo":
            spot_checks_gdf.at[index, 'name'] = PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS
        elif label_name == "passable with difficulty (road condition)" or label_name == "pdrc":
            spot_checks_gdf.at[index, 'name'] = PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION
        elif label_name == "passable with difficulty (flooding)" or label_name == "pdf":
            spot_checks_gdf.at[index, 'name'] = PASSABLE_WITH_DIFFICULTY_FLOODING
        elif label_name == "not passable (obstruction)" or \
             label_name == "not passable (obstructions)" or \
             label_name == "not-passable (obstructions)" or \
             label_name == "npo":
            spot_checks_gdf.at[index, 'name'] = NOT_PASSABLE_OBSTRUCTIONS
        elif label_name == "npf":
            spot_checks_gdf.at[index, 'name'] = NOT_PASSABLE_FLOODING
        elif label_name == "npd" or \
             label_name == "des" or \
             label_name == "not passable (destruction)":
            spot_checks_gdf.at[index, 'name'] = NOT_PASSABLE_DESTRUCTION
        elif label_name == "no able to determine" or label_name == "not able to determine" or label_name == "nd":
            spot_checks_gdf.at[index, 'name'] = NOT_ABLE_TO_DETERMINE
        elif label_name == "parp":
            spot_checks_gdf.at[index, 'name'] = PARTICULATE_PARTIAL
        elif label_name == "part" or label_name == "oart":
            spot_checks_gdf.at[index, 'name'] = PARTICULATE_TOTAL
        elif label_name == "road":
            spot_checks_gdf.at[index, 'name'] = ROAD_LINE
        else:
            print("Warning: Unknown label found --- " + label_name)

    return spot_checks_gdf

def combine_spot_checks(spot_checks_gdf, spot_checks_dict, added_roads_dict):
    for _, row in spot_checks_gdf.iterrows():
        key = row["name"]
        if key == ROAD_LINE:
            added_roads_dict[key].append(row.geometry)
        else:
            spot_checks_dict[key].append(row.geometry)

    return spot_checks_dict, added_roads_dict

def integrate_spot_checks(annotation_data, spot_polygons):
    # Implementing changes according to the following heirachy of labels...

    polygons = {r:[] for r in RDA_LABELBOX_CLASSES}
    for polygon in annotation_data["polygons"]:
        p_shape = Polygon([(p["lon"], p["lat"]) for p in polygon["EPSG:4326"]]).buffer(0)
        polygons[polygon["label"]].append(p_shape)

    # First, delete any polygons that were marked as del
    deleted_labels = 0
    for polygon in spot_polygons["del"]:
        for r in RDA_LABELBOX_CLASSES:
            for poly_r in polygons[r]:
                if poly_r.intersects(polygon):
                    print("Deleting Annotation with ", r, " label")
                    polygons[r].remove(poly_r)
                    deleted_labels += 1

    # Do nothing with no annotations

    # Add any intersection/labels with passable with difficulty (obstruction)
    pdo_labels = 0
    for polygon in spot_polygons[PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS]:
        print("Adding Annotation with label " + PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS)
        polygons[PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS].append(polygon)
        pdo_labels += 1

    # Add any intersection/labels with passable with difficulty (road condition)
    pdrc_labels = 0
    for polygon in spot_polygons[PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION]:
        print("Adding Annotation with label " + PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION)
        polygons[PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION].append(polygon)
        pdrc_labels += 1

    # Add any intersection/labels with passable with difficulty (flooding)
    pdf_labels = 0
    for polygon in spot_polygons[PASSABLE_WITH_DIFFICULTY_FLOODING]:
        print("Adding Annotation with label " + PASSABLE_WITH_DIFFICULTY_FLOODING)
        polygons[PASSABLE_WITH_DIFFICULTY_FLOODING].append(polygon)
        pdf_labels += 1

    # Now, change any intersection/labels with not passable (obstruction)
    npo_add_labels = 0
    npo_change_labels = 0
    for polygon in spot_polygons[NOT_PASSABLE_OBSTRUCTIONS]:
        print("Adding Annotation with label " + NOT_PASSABLE_OBSTRUCTIONS)
        polygons[NOT_PASSABLE_OBSTRUCTIONS].append(polygon)
        npo_add_labels += 1
        for poly_r in polygons[PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS]:
            if poly_r.intersects(polygon):
                print("Merging Annotation with " + PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS + " for " + NOT_PASSABLE_OBSTRUCTIONS)
                # Remove polygon and add the difference of the two polygons ...
                polygons[PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS].remove(poly_r)
                diff_polygon = poly_r.difference(polygon)
                if diff_polygon.area > 0:
                    polygons[PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS].append(diff_polygon)
                npo_change_labels += 1

    # And, change any intersection/labels not passable (flooding)
    npf_add_labels = 0
    npf_change_labels = 0
    for polygon in spot_polygons[NOT_PASSABLE_FLOODING]:
        print("Adding Annotation with label " + NOT_PASSABLE_FLOODING)
        polygons[NOT_PASSABLE_FLOODING].append(polygon)
        npf_add_labels += 1
        for poly_r in polygons[PASSABLE_WITH_DIFFICULTY_FLOODING]:
            if poly_r.intersects(polygon):
                print("Merging Annotation with " + PASSABLE_WITH_DIFFICULTY_FLOODING + " for " + NOT_PASSABLE_FLOODING)
                # Remove polygon and add the difference of the two polygons ...
                polygons[PASSABLE_WITH_DIFFICULTY_FLOODING].remove(poly_r)
                diff_polygon = poly_r.difference(polygon)
                if diff_polygon.area > 0:
                    polygons[PASSABLE_WITH_DIFFICULTY_FLOODING].append(diff_polygon)
                npf_change_labels += 1

    # And, change any intersection/labels not passable (destruction)
    npd_add_labels = 0
    npd_change_labels = 0
    for polygon in spot_polygons[NOT_PASSABLE_DESTRUCTION]:
        print("Adding Annotation with label " + NOT_PASSABLE_DESTRUCTION)
        polygons[NOT_PASSABLE_DESTRUCTION].append(polygon)
        npd_add_labels += 1
        for poly_r in polygons[PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION]:
            if poly_r.intersects(polygon):
                print("Merging Annotation with " + PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION + " for " + NOT_PASSABLE_DESTRUCTION)
                # Remove polygon and add the difference of the two polygons ...
                polygons[PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION].remove(poly_r)
                diff_polygon = poly_r.difference(polygon)
                if diff_polygon.area > 0:
                    polygons[PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION].append(diff_polygon)
                npd_change_labels += 1

    # Finally, change intersection/labels not able to determine
    labels_to_consider = [
        PASSABLE_WITH_DIFFICULTY_FLOODING,
        PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS,
        PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION,
        NOT_PASSABLE_DESTRUCTION,
        NOT_PASSABLE_OBSTRUCTIONS,
        NOT_PASSABLE_FLOODING]

    nd_add_labels = 0
    nd_change_labels = 0
    for polygon in spot_polygons[NOT_ABLE_TO_DETERMINE]:
        print("Adding Annotation with label " + NOT_ABLE_TO_DETERMINE)
        polygons[NOT_ABLE_TO_DETERMINE].append(polygon)
        nd_add_labels += 1
        for r in (labels_to_consider):
            for poly_r in polygons[r]:
                if poly_r.intersects(polygon):
                    print("Merging Annotation with " + r+ " for " + NOT_PASSABLE_DESTRUCTION)
                    # Remove polygon and add the difference of the two polygons ...
                    polygons[r].remove(poly_r)
                    diff_polygon = poly_r.difference(polygon)
                    if diff_polygon.area > 0:
                        polygons[r].append(diff_polygon)
                    nd_change_labels += 1

    # At the end, add any annotations regarding the particulates
    parp_labels = 0
    for polygon in spot_polygons[PARTICULATE_PARTIAL]:
        print("Adding Annotation with label " + PARTICULATE_PARTIAL)
        polygons[PARTICULATE_PARTIAL].append(polygon)
        parp_labels += 1

    # And, handle any overlap with the Particulate Total Class
    part_add_labels = 0
    part_change_labels = 0
    for polygon in spot_polygons[PARTICULATE_TOTAL]:
        print("Adding Annotation with label " + PARTICULATE_TOTAL)
        polygons[PARTICULATE_TOTAL].append(polygon)
        part_add_labels += 1
        for poly_r in polygons[PARTICULATE_PARTIAL]:
            if poly_r.intersects(polygon):
                print("Merging Annotation with " + PARTICULATE_PARTIAL + " for " + PARTICULATE_TOTAL)
                # Remove polygon and add the difference of the two polygons ...
                polygons[PARTICULATE_PARTIAL].remove(poly_r)
                diff_polygon = poly_r.difference(polygon)
                if diff_polygon.area > 0:
                    polygons[PARTICULATE_PARTIAL].append(diff_polygon)
                part_change_labels += 1


    print("--- Ortho Spot Checks Overview ---")
    print("Deleted " + str(deleted_labels) + " labels.")
    print("Added " + str(pdo_labels) + " " + PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS + " labels.")
    print("Added " + str(pdrc_labels) + " " + PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION + " labels.")
    print("Added " + str(pdf_labels) + " " + PASSABLE_WITH_DIFFICULTY_FLOODING + " labels.")

    print("Added " + str(npo_add_labels) + " " + NOT_PASSABLE_OBSTRUCTIONS + " labels.")
    print("Merged " + str(npo_change_labels) + " " + PASSABLE_WITH_DIFFICULTY_OBSTRUCTIONS + " with " + NOT_PASSABLE_OBSTRUCTIONS + " labels.")

    print("Added " + str(npf_add_labels) + " " + NOT_PASSABLE_FLOODING + " labels.")
    print("Merged " + str(npf_change_labels) + " " + PASSABLE_WITH_DIFFICULTY_FLOODING + " with " + NOT_PASSABLE_FLOODING + " labels.")

    print("Added " + str(npd_add_labels) + " " + NOT_PASSABLE_DESTRUCTION + " labels.")
    print("Merged " + str(npd_change_labels) + " " + PASSABLE_WITH_DIFFICULTY_ROAD_CONDITION + " with " + NOT_PASSABLE_DESTRUCTION + " labels.")

    print("Added " + str(nd_add_labels) + " " + NOT_ABLE_TO_DETERMINE + " labels.")
    print("Merged " + str(nd_change_labels) + " with " + NOT_ABLE_TO_DETERMINE + " labels.")

    print("Added " + str(parp_labels) + " " + PARTICULATE_PARTIAL + " labels.")
    print("Added " + str(part_add_labels) + " " + PARTICULATE_TOTAL + " labels.")
    print("Merged " + str(part_change_labels) +  " " + PARTICULATE_PARTIAL + " with " + PARTICULATE_TOTAL + " labels.")

    return polygons


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='integrate_RDA_spot_checks',
        description='This program combines the spot checks that were manually entered and udpates the annotations file.')
    parser.add_argument('--spot_checks_folder', type=str, help="The path to the spot checks folder.")
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--spot_check_path_map', type=str, help="The path to the output spot checked annotations file path map.")
    parser.add_argument('--output_annotations_folder', type=str, help="The patht to the output annotations folder with the spot checks applied.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_annotations_folder)
    except FileExistsError as e:
        pass

    try:
        os.makedirs(args.output_annotations_folder)
    except FileExistsError as e:
        pass

    annotations_path_map = json.load(open(args.annotations_path_map))
    num_spot_checked = 0
    not_found = 0
    spot_check_map = {}

    for geotif_path, annotation_path in annotations_path_map.items():

        target_geotif = os.path.split(geotif_path)[1]

        #Load the annotations
        print("Loading the RDA annotations from:", annotation_path)
        f = open(annotation_path, "r")
        annotations_data = json.loads(f.read())
        f.close()

        print("Loading the target geotiff metadata from:", geotif_path)
        input_geotiff_data = rasterio.open(geotif_path, "r")

        coord_system = str(input_geotiff_data.crs)
        print("Generating transformer between:", coord_system, "and", LAT_LON_CRS)
        coord_transformer = Transformer.from_crs(coord_system, LAT_LON_CRS)

        spot_check_polygons = {c:[] for c in (RDA_LABELBOX_CLASSES + ["del"])}
        spot_check_data_roadlines = {ROAD_LINE: []}

        spot_check_path = os.path.join(args.spot_checks_folder, target_geotif + "-RDA_CUSTOM_LABELS.geojson")
        if os.path.exists(spot_check_path):

            print("Loading spot Checks for " + spot_check_path)
            spot_check_data = gpd.read_file(spot_check_path)
            spot_check_data = parse_rda_labels(spot_check_data)

            # Sort spot checks based on labels ... Separate spot checks and added roadlines ...
            spot_check_polygons, spot_check_data_roadlines = combine_spot_checks(spot_check_data, spot_check_polygons, spot_check_data_roadlines)

            print("Spot Checking Annotations ...")
            spot_check_annotations_data = integrate_spot_checks(annotations_data, spot_check_polygons)

            # Get x,y and the lat,lon for the spot checked annotations ...
            labeled_polygons = []
            for label, polygons in spot_check_annotations_data.items():
                for polygon in polygons:
                    labeled_polygons.append([label, polygon])
            annotations_data["polygons"] = convert_labeled_polygons_to_json_xy_and_lat_lon(labeled_polygons, input_geotiff_data)

            # Get x,y and the lat,lon for addon roadlines ...
            added_roadlines = []
            for label, roadlines in spot_check_data_roadlines.items():
                for road in roadlines:
                    added_roadlines.append([label, road])
            annotations_data["road_lines"].extend(convert_road_lines_to_json_xy_and_lat_lon(added_roadlines, input_geotiff_data))

            print("\n")
            print("Writing polygons to json file ...")
            out_file = target_geotif + ".json"
            out_path = os.path.join(args.output_annotations_folder, out_file)
            f = open(out_path, "w")
            f.write(json.dumps(annotations_data, indent=4, sort_keys=True))
            f.close()
            print("Polygons saved at", out_path)
            spot_check_map[geotif_path] = out_path

            num_spot_checked += 1

        else:
            print("Could not find spot checks for " + target_geotif)
            print("Saving original annotations ...")

            print("\n")
            print("Writing polygons to json file ...")
            out_file = target_geotif + ".json"
            out_path = os.path.join(args.output_annotations_folder, out_file)
            f = open(out_path, "w")
            f.write(json.dumps(annotations_data, indent=4, sort_keys=True))
            f.close()
            print("Polygons saved at", out_path)
            spot_check_map[geotif_path] = out_path

            not_found += 1

    print("Writing output spot checked annotations path map ...")
    f = open(args.spot_check_path_map, "w")
    f.write(json.dumps(spot_check_map))
    f.close()
    print("Done.")

    print("Spot Checked", num_spot_checked, "Ortho Annotations. Did Not Find", not_found , "Ortho Spot Checks.")