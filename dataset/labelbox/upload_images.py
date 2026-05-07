import labelbox as lb
import os
import argparse
from alive_progress import alive_bar
import json

parser = argparse.ArgumentParser(description='Upload Images to Labelbox.')
parser.add_argument('--images_folder', help="The path to the folder that contains the images to upload", type=str)
parser.add_argument('--api_key_file', help="The path to the file containing the API key", type=str)
parser.add_argument('--upload_chunk_size', help="The number of images that are uploaded in one request", type=int, default=16)
parser.add_argument('--dataset_name', help="The name of the dataset that will be created", default=None)
parser.add_argument('--create_dataset', action="store_true", help="If flag is set, a new dataset will be created, otherwise an update will be attempted.")
args = parser.parse_args()

#Load the API key
f = open(args.api_key_file)
API_KEY = f.readlines()[0].replace(" ", "").replace("\n", "").replace("\r", "")
f.close()
client = lb.Client(API_KEY)

#Get the dataset name
dataset_name = os.path.split(args.images_folder)[-1]
if(not args.dataset_name is None):
	dataset_name = args.dataset_name

images = []
for file in os.listdir(args.images_folder):
	images.append({
	    "row_data": os.path.join(args.images_folder, file),
	    "global_key": file
	})

print("Uploading RGB Images From: ", args.images_folder)
dataset = None
if(args.create_dataset):
	dataset = client.create_dataset(name=dataset_name)
	print("Created new dataset with name:", dataset_name)
else:
	dataset = client.get_datasets(where=lb.Dataset.name == dataset_name).get_one()
	print("Founding existing dataset with name:", dataset_name)

with open('images.json', 'w') as f:
	json.dump(images, f)

i = 0
with alive_bar(len(images)/args.upload_chunk_size) as bar:
	while i < len(images):
		try:
			task = dataset.create_data_rows(images[i:i+args.upload_chunk_size])
			task.wait_till_done()
		
			if(not task.errors is None):
				print(f"Errors: {task.errors}")
			
		except Exception as e:
			print(e)
			print("Facing issue with current chunk, " + str(images[i:i+args.upload_chunk_size]) +"\n")

		i += args.upload_chunk_size
		bar()
	
print("Done")