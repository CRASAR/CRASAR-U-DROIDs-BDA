import json
import labelbox.data.annotation_types as lb_types
from shapely import Polygon

def get_lb_building_annotations_for_image(
    polygon_file, default_label_name, preannotated=False
):
    annotations = []
    with open(polygon_file, "r") as f:
        data = json.loads(f.read())

    source = data["source"]
    polygons = data["polygons"]
    if preannotated:
        labels = data["labels"]

    seen_polygons = []

    for i, polygon in enumerate(polygons):
        p = lb_types.Polygon(points=[lb_types.Point(x=x, y=y) for x, y in polygon])
        
        if preannotated:
            annotations.append(lb_types.ObjectAnnotation(name=labels[i], value=p))
        else:
            if Polygon(polygon) not in seen_polygons:
                annotations.append(
                    lb_types.ObjectAnnotation(name=default_label_name, value=p)
                )
                seen_polygons.append(Polygon(polygon))

    return source, annotations


def get_lb_road_line_annotations_for_image(polyline_file, default_label_name):
    annotations = []
    with open(polyline_file, "r") as f:
        data = json.loads(f.read())

    source = data["source"]
    polylines = data["polylines"]

    for line in polylines:
        if len(line) > 0:
            p = lb_types.Line(points=[lb_types.Point(x=x, y=y) for x, y in line])
            annotations.append(
                lb_types.ObjectAnnotation(name=default_label_name, value=p)
            )

    return source, annotations
