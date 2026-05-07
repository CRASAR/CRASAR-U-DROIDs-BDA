import labelbox as lb
import labelbox.data.annotation_types as lb_types
import uuid
import os
import argparse

from labelbox_utils import get_lb_building_annotations_for_image

parser = argparse.ArgumentParser(description='Upload Labels to Labelbox.')
parser.add_argument('--api_key_file', help="The path to the file containing the API key", type=str)
parser.add_argument('--project_id', help="The project ID for this task on labelbox", type=str)
parser.add_argument('--dataset_name', help="The name of the dataset batch", type=str)
parser.add_argument('--label_folder', help="The path to the folder where the labels are stored", type=str)
parser.add_argument('--label_file_postfix', help="The postfix to match against when searching for label files", type=str, default="_building_mask.json")
parser.add_argument('--default_label', help="The postfix to match against when searching for label files", type=str, default=None)
parser.add_argument('--priority', help="Priority Number 1(highest) - 5 (lowest)", type=int)
args = parser.parse_args()

default_label = "Not Able To Determine"
if(args.default_label):
	default_label = args.default_label

#Load the API Key
f = open(args.api_key_file)
API_KEY = f.readlines()[0].replace(" ", "").replace("\n", "").replace("\r", "")
f.close()
client = lb.Client(API_KEY)

#Get the dataset name
dataset_name = os.path.split(args.label_folder)[-1]
if(not args.dataset_name is None):
	dataset_name = args.dataset_name

#Get the project from the ID
bda_project = client.get_project(project_id = args.project_id)

#Get the files that contain the labels that we want to upload and load them
target_files = []
annotations = {}
for file in os.listdir(args.label_folder):
	if(file.endswith(args.label_file_postfix)):
		target_files.append(file)
		source_file, annotation = get_lb_building_annotations_for_image(os.path.join(args.label_folder, file), default_label)
		annotations[source_file] = annotation

#Attach the files to the project in a batch for annotation
batch = bda_project.create_batch(
  "batch-" + str(dataset_name) + "_priority_" + str(args.priority), # Each batch in a project must have a unique name
  global_keys=list(annotations.keys()), # A list of data rows or data row ids
  priority=args.priority, # priority between 1(Highest) - 5(lowest) 
)
print("Batch: ", batch)

#Upload the labels that we have
labels = []
for global_key in annotations.keys():
	labels.append(lb_types.Label(data=lb_types.ImageData(global_key=global_key), annotations=annotations[global_key]))

upload_job = lb.MALPredictionImport.create_from_objects(
  client = client,
  project_id = args.project_id,
  name = "mal_job"+str(uuid.uuid4()),
  predictions = labels)

print("Errors:", upload_job.errors)