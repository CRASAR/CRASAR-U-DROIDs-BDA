from OSMPythonTools.overpass import overpassQueryBuilder, Overpass
import time
import random

VALID_HIGHWAYS = [
	"road",
	"service",
	"motorway", 
	"trunk", 
	"primary", 
	"secondary", 
	"tertiary", 
	"residential", 
	"motorway_link", 
	"trunk_link", 
	"primary_link", 
	"secondary_link", 
	"tertiary_link", 
	"living_street", 
	"turning_loop", 
	"turning_circle"
]


def safe_query(overpass, query, retries=3):
    for i in range(retries):
        try:
            return overpass.query(query)
        except Exception as e:
            print(f"Query failed (attempt {i+1}/{retries}): {e}")
            if i < retries - 1:
                sleep_time = 5 * (2 ** i) + random.random()
                print(f"Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                raise


def get_osm_roads_buildings_and_nodes_for_bounding_box(bbox, date="2022-09-27T00:00:00Z"):

	overpass = Overpass()
	# Include road as long as at least one node is within bounding box
	query = overpassQueryBuilder(bbox=bbox, elementType=['node', 'way'], includeGeometry=True) 
	query = query.replace("out body;", "way(bn); (._;>;); out geom;").replace("[timeout:25][out:json];", "[timeout:25][out:json][date:\"" + str(date) + "\"][bbox];")
	result = safe_query(overpass, query)
	data = result.toJSON()

	buildings = []
	roads = []
	nodes = {}

	for elem in data["elements"]:
		if("tags" in elem.keys()):
			if "building" in elem["tags"].keys():
				if elem["tags"]["building"] == "yes" or elem["tags"]["building"] == "house":
					buildings.append(elem)
			else:
				if "highway" in elem["tags"]:
					hw = elem["tags"]["highway"] 
					if(hw in VALID_HIGHWAYS):
						roads.append(elem)

	return {"buildings":buildings, "roads":roads}