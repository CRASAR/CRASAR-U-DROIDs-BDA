import math
import copy
from shapely.geometry import LineString

def apply_adjustments(adjustments, x, y):
	min_dist = float("inf")
	best_line = None
	if(len(adjustments) == 0):
		return x, y
	for line in adjustments:
		dist = math.dist([x, y], line[0])
		if(dist < min_dist):
			min_dist = dist
			best_line = line

	dx = best_line[1][0] - best_line[0][0]
	dy = best_line[1][1] - best_line[0][1]

	return x+dx, y+dy

def match_polygon_to_adjustment(adjustments, polygon):
	min_dist = float("inf")
	best_line = None
	for line in adjustments:
		for vertex in polygon:
			x = vertex["x"]
			y = vertex["y"]
			dist = math.dist([x, y], line[0])
			if(dist < min_dist):
				min_dist = dist
				best_line = line
	return best_line

def match_vertex_to_adjustment(adjustments, vertex, tuple=False):
	min_dist = float("inf")
	best_line = None
	for line in adjustments:
		if tuple:
			x,y = vertex
		else:
			x = vertex["x"]
			y = vertex["y"]
		dist = math.dist([x, y], line[0])
		if(dist < min_dist):
			min_dist = dist
			best_line = line
	return best_line

def apply_adjustment_to_polygon(adjustment, polygon):
	adjusted_polygon = copy.deepcopy(polygon)

	dx = adjustment[1][0] - adjustment[0][0]
	dy = adjustment[1][1] - adjustment[0][1]

	new_coords = []
	for i in range(0, len(adjusted_polygon["pixels"])):
		adjusted_polygon["pixels"][i]["x"] += dx
		adjusted_polygon["pixels"][i]["y"] += dy

	return adjusted_polygon

def adjust_road_lines(road_lines, adjustments):
    adjusted_road_lines = copy.deepcopy(road_lines)
    for i in range(0, len(road_lines)):
        coords = []
        for j in range(0, len(road_lines[i]["pixels"])):
            
            best_adjustment = match_vertex_to_adjustment(adjustments, road_lines[i]["pixels"][j])
            if best_adjustment is None:
                best_adjustment_options = []
            else:
                best_adjustment_options = [best_adjustment]
            
            x_adj, y_adj = apply_adjustments(best_adjustment_options, road_lines[i]["pixels"][j]["x"], road_lines[i]["pixels"][j]["y"])
            adjusted_road_lines[i]["pixels"][j]["x"] = x_adj
            adjusted_road_lines[i]["pixels"][j]["y"] = y_adj
    return adjusted_road_lines

def adjust_building_polygons(building_polygons, adjustments):
	adjusted_polygons = []
	for polygon in building_polygons:
		adjustment = match_polygon_to_adjustment(adjustments, polygon["pixels"])
		adjusted_polygons.append(apply_adjustment_to_polygon(adjustment, polygon))
	return adjusted_polygons


