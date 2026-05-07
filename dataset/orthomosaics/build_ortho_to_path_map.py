import os
import json
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--ortho_dir_path', type=str, help="The path to the folder that contains all orthomosaics for the project.")
	parser.add_argument('--output_json_path', type=str, help="The path to the file that will contain the mapping from ortho filenames to full paths.")
	args = parser.parse_args()

	result = {}

	for (root,dirs,files) in os.walk(args.ortho_dir_path, topdown=True): 
		for file in files:
			if file.endswith(".tif") or file.endswith(".tiff"):
				
				#Load the ortho
				geotiff_path = os.path.join(root, file)

				result[file] = geotiff_path

	f = open(args.output_json_path, "w")
	f.write(json.dumps(result))
	f.close()