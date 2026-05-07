import argparse
import json
import os
import shapely
import rasterio

from shapely.geometry import LineString, Polygon
from pyproj import Transformer

from dataset.constants import RDA_LABELBOX_CLASSES, LAT_LON_CRS, ORTHO_GSD
from dataset.utils.adjustment_utils import apply_adjustments, match_vertex_to_adjustment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='reconstruct_RDA_polygons', description='This program reconstructs the polygons that intersect adjusted roadlines.')
    parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
    parser.add_argument('--adjustments_path_map', type=str, help="The path to the adjustments file path map.", default=None)
    parser.add_argument('--buffer_distance', type=float, help="The spatial distance to buffer in centimeters.")
    parser.add_argument('--output_annotations_folder', type=str, help="The path to the output annotations folder updated polygons applied.")
    parser.add_argument('--output_annotations_path_map', type=str, help="The path map for the rebased annotations.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_annotations_folder)
    except FileExistsError as e:
        pass

    annotations_path_map = json.load(open(args.annotations_path_map))
    adjustments_path_map = json.load(open(args.adjustments_path_map))
    output_path_map = {}

    for geotif_path, annotation_path in annotations_path_map.items():

        ortho_local_title = os.path.split(geotif_path)[1]

        #Load the annotations
        print("Loading the RDA annotations from:", annotation_path)
        f = open(annotation_path, "r")
        annotations_data = json.loads(f.read())
        f.close()

        #Load adjustments
        try:
            print("Loading the RDA Adjustments from:", adjustments_path_map[geotif_path])
            f = open(adjustments_path_map[geotif_path], "r")
            adjustments = json.loads(f.read())
            f.close()
        except KeyError as e:
            print("Found no adjustments for ", ortho_local_title)
            adjustments=[]

        # Load geotiff
        print("Loading external base geotif from:", geotif_path)
        input_data = rasterio.open(geotif_path, "r")
        print("Done...")

        polygons = {r:[] for r in RDA_LABELBOX_CLASSES}
        for polygon in annotations_data["polygons"]:
            p_shape = Polygon([(p["x"], p["y"]) for p in polygon["pixels"]]).buffer(0)
            polygons[polygon["label"]].append(p_shape)

        buffered_polygons = {r: [] for r in RDA_LABELBOX_CLASSES}
        result = []
        coord_transformer = Transformer.from_crs(input_data.crs.to_string(), LAT_LON_CRS)
        transform = input_data.transform

        # Convert Spatial Distance to Pixel Distance based on GSD
        buffer_distance = args.buffer_distance / ORTHO_GSD[ortho_local_title]

        #Apply adjustments to line before intersection
        for line in annotations_data["road_lines"]:
		    # Align all points within the line according to the best adjustment
            verts = []
            for point in line["pixels"]:
			    # Find the best adjustment for the roadline
                best_adjustment = match_vertex_to_adjustment(adjustments, point)
                if best_adjustment is None:
                    best_adjustment_options = []
                else:
                    best_adjustment_options = [best_adjustment]

                if line["source"] == "custom":
                    best_adjustment_options = []

                x_adj, y_adj = apply_adjustments(best_adjustment_options, point["x"], point["y"])
                verts.append([x_adj, y_adj])

            adjusted_roadline = LineString(verts)

            annotated_polygons = []

            for r in RDA_LABELBOX_CLASSES :
                for poly_r in polygons[r]:

                    ls_poly_r = shapely.intersection(adjusted_roadline, poly_r)
                    if ls_poly_r.length > 0:

                        intersection_poly = shapely.buffer(ls_poly_r, buffer_distance, cap_style="flat")
                        individual_polys = list(intersection_poly.geoms) if intersection_poly.geom_type == "MultiPolygon" else [intersection_poly]

                        for buffered_poly in individual_polys:

                            pixel_coords_polygon = []
                            target_crs_polygon = []

                            for x, y in list(zip(*buffered_poly.exterior.coords.xy)):
                                pixel_coords_polygon.append({"x": x, "y": y})

                                x_source, y_source = rasterio.transform.xy(input_data.transform, y, x)
                                x_t, y_t = coord_transformer.transform(x_source, y_source)

                                target_crs_polygon.append({"lat": x_t, "lon": y_t})


                            result.append({
                                "source": line["source"],
                                "label": r,
                                "pixels": pixel_coords_polygon,
                                LAT_LON_CRS: target_crs_polygon
                            })

        annotations_data["polygons"] = result

        print("\tWriting reconstructed RDA Polygon annotations...")
        path = os.path.join(args.output_annotations_folder, ortho_local_title + ".json")
        f = open(path, "w")
        f.write(json.dumps(annotations_data, indent=4, sort_keys=True))
        f.close()
        output_path_map[geotif_path] = path

    print("Writing Output Path Map...")
    f = open(args.output_annotations_path_map, "w")
    f.write(json.dumps(output_path_map))
    f.close()
    print("\tDone")
