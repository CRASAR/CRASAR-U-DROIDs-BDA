import math
import shapely
from shapely import LineString, Polygon, MultiPolygon, MultiLineString
from shapely.validation import explain_validity

from dataset.constants import ROAD_DEBUG_SIZE_MAP, LAT_LON_CRS, UNCLASSIFIED


def truncate_polygon_vertexes(polygon, height, width):
    image = Polygon([(0, 0), (0, height), (width, height), (width, 0), (0, 0)])
    poly = Polygon(polygon)

    if not poly.is_valid:
        print("Non - Valid polygon, trying to fix....\n")
        poly = poly.buffer(0)
        if not poly.is_valid:
            print(f"Failed to fix the polygon. Explanation: {explain_validity(poly)}")
            return []
    intersection_polygon = shapely.intersection(image, poly)

    truncated_polygons = []
    if intersection_polygon.area > 0:
        if isinstance(intersection_polygon, Polygon):
            truncated_polygons = [list(intersection_polygon.exterior.coords)]
        elif isinstance(intersection_polygon, MultiPolygon):
            truncated_polygons = [
                list(x.exterior.coords) for x in intersection_polygon.geoms
            ]

    return truncated_polygons


def truncate_polyline_vertexes(line, height, width):
    image = Polygon([(0, 0), (0, height), (width, height), (width, 0), (0, 0)])
    line_str = LineString(line)
    intersection_line = shapely.intersection(image, line_str)
    truncated_polylines = []
    if isinstance(intersection_line, LineString):
        truncated_polylines = [list(intersection_line.coords)]
    elif isinstance(intersection_line, MultiLineString):
        truncated_polylines = [list(x.coords) for x in intersection_line.geoms]
    return truncated_polylines


def get_road_polylines_from_osm_data(
    data, ortho_data, coord_transformer, swap_xy=False
):
    roads = data["roads"]
    lines = []
    for road in roads:
        line = []
        if "geometry" in road.keys():
            for i in range(0, len(road["geometry"])):
                x = road["geometry"][i]["lat"]
                y = road["geometry"][i]["lon"]
                if swap_xy:
                    x_t, y_t = coord_transformer.transform(y, x)
                else:
                    x_t, y_t = coord_transformer.transform(x, y)
                row, col = ortho_data.index(x_t, y_t)
                line.append((col, row))
            truncated_data = truncate_polyline_vertexes(
                line, ortho_data.height, ortho_data.width
            )
            if len(truncated_data) > 0:
                lines.extend(truncated_data)

    return lines


def draw_roads_on_ortho_img(
    ortho_data,
    geometry_data,
    coord_transformer,
    im_draw,
    color="white",
    swap_xy=False,
):
    polygons = []
    for road in geometry_data["roads"]:
        if "geometry" in road.keys():
            line_transformed = []
            for i in range(0, len(road["geometry"])):
                try:
                    x = road["geometry"][i]["lat"]
                    y = road["geometry"][i]["lon"]
                    if swap_xy:
                        x_t, y_t = coord_transformer.transform(y, x)
                    else:
                        x_t, y_t = coord_transformer.transform(x, y)
                    row, col = ortho_data.index(x_t, y_t)
                    line_transformed.append((col, row))

                except KeyError:
                    pass

            width = ROAD_DEBUG_SIZE_MAP[road["tags"]["highway"]] / 2
            poly_buf = shapely.buffer(LineString(line_transformed), width)
            shape = list(poly_buf.exterior.coords)
            truncated_polygons = truncate_polygon_vertexes(
                shape, ortho_data.height, ortho_data.width
            )
            if len(truncated_polygons) > 0:
                polygons.extend(truncated_polygons)
                for p in truncated_polygons:
                    im_draw.polygon(p, fill=color)
    return polygons

# pylint: disable-next=too-many-branches
def draw_buildings_on_ortho_img(
    ortho_data,
    geometry_data,
    coord_transformer,
    im_draw,
    color="white",
    swap_xy=False,
    preannotated=False,
    annotations=None,
):
    polygons = []
    labels = []
    for building in geometry_data["buildings"]:
        if "geometry" in building.keys():
            shape = []
            verts = []
            for i in range(0, len(building["geometry"])):
                try:
                    x = building["geometry"][i]["lat"]
                    y = building["geometry"][i]["lon"]
                    if swap_xy:
                        x_t, y_t = coord_transformer.transform(y, x)
                    else:
                        x_t, y_t = coord_transformer.transform(x, y)

                    row, col = ortho_data.index(x_t, y_t)
                    shape.append((col, row))
                    verts.append((x, y))
                except KeyError:
                    pass
            spatial_building = Polygon(verts)

            if preannotated:
                label = match_polygon_annotation(spatial_building, annotations)

            truncated_polygons = truncate_polygon_vertexes(
                shape, ortho_data.height, ortho_data.width
            )

            deduped_polygons = []
            for p in truncated_polygons:
                if not list_contains_polygon(polygons, p):
                    deduped_polygons.append(p)

            if len(deduped_polygons) > 0:
                polygons.extend(deduped_polygons)

                if preannotated:
                    labels.append(label)
                for p in deduped_polygons:
                    im_draw.polygon(p, fill=color)
    if preannotated:
        return polygons, labels
    return polygons


def list_contains_polygon(l, target):
    for l_i in l:
        if target == l_i:
            return True
    return False


def match_polygon_annotation(building, annotations):
    centriod = building.centroid
    min_dis = float("inf")
    current_label = None
    for annotation in annotations:
        verts = []
        for latlon in annotation[LAT_LON_CRS]:
            verts.append((latlon["lat"], latlon["lon"]))
        centriod_annotations = Polygon(verts).centroid
        dist = math.dist(
            [centriod.x, centriod.y], [centriod_annotations.x, centriod_annotations.y]
        )

        if dist < min_dis:
            min_dis = dist
            current_label = annotation["label"]

    if current_label is None:
        print(
            "Warning: Could not find match for polygons with preannotations, return un-classified."
        )
        current_label = UNCLASSIFIED
    return current_label

