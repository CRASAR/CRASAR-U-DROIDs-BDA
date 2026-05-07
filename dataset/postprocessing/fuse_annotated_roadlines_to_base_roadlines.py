import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
generators_path = os.path.dirname(current)
dataset_path = os.path.dirname(generators_path)
sys.path.append(generators_path)
sys.path.append(dataset_path)

from dataset.constants import LAT_LON_CRS, ROAD_LINE
from dataset.utils.ortho_utils import transform_bounds
from dataset.generators.osm_data_generator import get_osm_roads_buildings_and_nodes_for_bounding_box
from dataset.utils.draw_utils import get_road_polylines_from_osm_data

from pyproj import Transformer

import json
import rasterio
import argparse

def rebase_annotated_roadlines(annotations_path_map, geotif_path_map, target_geotif, output_folder_path, output_path_map, swap_xy=False):
	print("Working", target_geotif)
	print("\tLoading geotiff path map")
	input_geotif_path = json.load(open(geotif_path_map))[target_geotif]
	
	print("\tLoading annotations")
	annotations_path = json.load(open(annotations_path_map))[input_geotif_path]
	f = open(annotations_path, "r")
	annotations = json.loads(f.read())
	f.close()

	print("\tLoading external base geotif from:", input_geotif_path)
	input_geotiff_data = rasterio.open(input_geotif_path, "r")

	coord_system = str(input_geotiff_data.crs)
	print("\tGenerating transformer between:", coord_system, "and", LAT_LON_CRS)
	coord_transformer = Transformer.from_crs(LAT_LON_CRS, coord_system)

	ortho_lat_lon_bounds = transform_bounds(input_geotiff_data, LAT_LON_CRS)
	osm_bbox = [ortho_lat_lon_bounds.bottom, ortho_lat_lon_bounds.left,
					            ortho_lat_lon_bounds.top, ortho_lat_lon_bounds.right]
	
	road_and_building_data = get_osm_roads_buildings_and_nodes_for_bounding_box(osm_bbox)
	road_polylines = get_road_polylines_from_osm_data(
					    road_and_building_data, input_geotiff_data, coord_transformer, swap_xy=swap_xy)

	coord_transformer = Transformer.from_crs(input_geotiff_data.crs.to_string(), LAT_LON_CRS)
	rebased_roadlines = []
	for roadline in road_polylines:
		pixel_coord_line = []
		target_crs_line = []
		for point in roadline:
			x, y = point
			pixel_coord_line.append({"x": x, "y": y})

			#Flip the y and x axis to align the data correctly in the coordinate space
			x_source, y_source = rasterio.transform.xy(input_geotiff_data.transform, y, x)
			if(swap_xy):
				x_t, y_t = coord_transformer.transform(y_source, x_source)
			else:
				x_t, y_t = coord_transformer.transform(x_source, y_source)
			target_crs_line.append({"lat": x_t, "lon": y_t})
		
		if((len(pixel_coord_line) > 0) and (len(target_crs_line) > 0)):
			rebased_roadlines.append({"source": "OSM",
							"label":ROAD_LINE, 
		           			"pixels":pixel_coord_line,
		           			LAT_LON_CRS:target_crs_line})
		
	annotations["road_lines"] = rebased_roadlines
	

	print("\tWriting rebased annotations...")
	path = os.path.join(output_folder_path, target_geotif + ".json")
	f = open(path, "w")
	f.write(json.dumps(annotations, indent=4, sort_keys=True))
	f.close()
	output_path_map[input_geotif_path] = path
	print("\tDone")

if __name__ == "__main__":

	parser = argparse.ArgumentParser(prog='rebase_annotated_roadlines', description='This program takes the annotated building polygons and fuses them with the unannotated initial polygons that were used to seed the annotation. As a result, this handles any failed tiles during the initial annotation process.')
	parser.add_argument('--annotations_path_map', type=str, help="The path to the annotations file path map.")
	parser.add_argument('--geotif_path_map', type=str, help='The input geotif path map.')
	parser.add_argument('--output_annotations_folder', type=str, help="The path to the output annotations folder updated polygons applied.")
	parser.add_argument('--output_annotations_path_map', type=str, help="The path map for the rebased annotations.")
	args = parser.parse_args()

	resulting_path_map = {}
	try:
		os.makedirs(args.output_annotations_folder)
	except FileExistsError as e:
		pass

	#Ian Boca Grande
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.1.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.2.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.3.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.4.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.5.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Boca-Grande.6.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Ian Fort Myers
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.1.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.2.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.3.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Palm-Acers.4.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Sanibel-Causeway-North.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Summerlin-San-Carlos.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Palmeto-Palms.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-McGregor-College-Pkwy-South.3.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-McGregor-College-Pkwy-South.2.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-McGregor-College-Pkwy-South.1.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Kennedy-Green-Mobile-Homes.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Iona-Point.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Harlem-Heights.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Kelly-Road.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Ian Fort Myers Beach
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-San-Carlos-Island.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Ft-Myers-Beach-DIRT.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1001-Ft-Myers-Beach-Boone.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Ft-Myers-Beach-LCSO.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "1002-Ft-Myers-Beach-TFD.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Laura
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "0827-A-01.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "0827-B-02.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Mussett
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-SouthOf98-LakeParkCove.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-SouthOf98-DelbertLn.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-SouthOf98-AnchorLakeDr.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-NorthOf98.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "05-08-2020-MussettBayouFire-01.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Mayfield
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20211214-Mayfield.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20211213-Candle-Factory-AO.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Russellville
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20211215-Russelville-Middle.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Kilauea
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "2018-05-18-X5-visible-Kahukai.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "2018-05-18-X5-visible-Geothermal.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "2018-05-18-X4S-visible-CentralPark.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Michael
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "10132018-MexicoBeach.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "10142018-MexicoBeach.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Idalia
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20230831-Jena-SteinhatcheeRiverSouth.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20230830-SteinhatcheeRiver.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Ida
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20210901-Cocodrie-2.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20210901-Cocodrie-3.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20210901-Cocodrie-1.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20210902-LA-DIV-01.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20210831-LA-DIV-01.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Harvey
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "090302-Pecan-Grove-Levee.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "090402-DMS-Assessment-Sienna-Village.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "090401-DMS-Assessment-Westpark.geo.tif", args.output_annotations_folder, resulting_path_map)
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "090403-Lancaster-Canyon-Gate.geo.tif", args.output_annotations_folder, resulting_path_map)

	#Champlain Towers
	rebase_annotated_roadlines(args.annotations_path_map, args.geotif_path_map, "20210703-Champlain-Towers -South.geo.tif", args.output_annotations_folder, resulting_path_map)
	
	print("Writing Output Path Map...")
	f = open(args.output_annotations_path_map, "w")
	f.write(json.dumps(resulting_path_map))
	f.close()