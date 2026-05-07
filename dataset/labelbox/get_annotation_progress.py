import labelbox
import argparse
import pandas as pd


def get_annotion_progress(client, status, labels, progress):
    for i, label in enumerate(labels):
        dataset_name = labels[i]["data_row"]["details"]["dataset_name"]

        if dataset_name not in progress.keys():
            progress[dataset_name] = {"Tiles": 0, "Labeled": 0, "Reviewed": 0}
            dataset_id = labels[i]["data_row"]["details"]["dataset_id"]
            dataset = client.get_dataset(dataset_id)
            progress[dataset_name]["Tiles"] = len(list(dataset.data_rows()))
        
        if status == "Done":
            progress[dataset_name]["Reviewed"] += 1
        else:
            progress[dataset_name]["Labeled"] += 1
    
    return progress

def get_percentage_progress(progress):
    for key in progress.keys():
        progress[key]["Labeled"] = (progress[key]["Tiles"] - progress[key]["Labeled"])/progress[key]["Tiles"]
        progress[key]["Reviewed"] = progress[key]["Reviewed"]/progress[key]["Tiles"]
    return progress

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='annotator statistics', description='This program computes the annotation progress.')
    parser.add_argument('--api_key_file', type=str, help='The path to the api key file.')
    parser.add_argument('--project_key', type=str, help='The labelbox project key.')
    parser.add_argument('--progress_file', type=str, help='The path to csv file where the annotation progress details will be saved.')
    args = parser.parse_args()

    #Load the API Key
    f = open(args.api_key_file)
    API_KEY = f.readlines()[0].replace(" ", "").replace("\n", "").replace("\r", "")
    f.close()

    annotation_progress = {}

    # Set parameters for export
    PROJECT_ID = args.project_key
    client = labelbox.Client(api_key = API_KEY)
    project = client.get_project(PROJECT_ID)
    params={
	    "data_row_details": True,
	    "metadata_fields": False,
	    "attachments": False,
	    "project_details": True,
	    "performance_details": True,
	    "label_details": True,
	    "interpolated_frames": False
    }

    # Extract Reviewed Labels
    filters={
        "workflow_status": "Done"
    }
    print("Exporting Done Labels from Project ....\n")
    export_labels = project.export_v2(params=params, filters=filters)
    export_labels.wait_till_done()
    if export_labels.errors:
        print(export_labels.errors)
    export_json = export_labels.result

    progress = get_annotion_progress(client, "Done", export_json, annotation_progress)

    # Extract Annotated Labels
    filters={
        "workflow_status": "ToLabel"
    }
    print("Exporting Annotated Labels from Project ....\n")
    export_labels = project.export_v2(params=params, filters=filters)
    export_labels.wait_till_done()
    if export_labels.errors:
        print(export_labels.errors)
    export_json = export_labels.result

    progress = get_annotion_progress(client, "ToLabel", export_json, annotation_progress)

    # Calculate Percentage for Labeled and Reviewed
    progress = get_percentage_progress(progress)

    print("Saving Annotation Progress details to csv file: " + str(args.progress_file))
    progress_df = pd.DataFrame.from_dict(progress, orient="index")
    progress_df.to_csv(args.progress_file)