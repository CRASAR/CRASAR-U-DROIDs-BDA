import labelbox
import argparse
import pandas as pd


def get_annotator_stats(project_id, labels, annotators, task):

    for i, label in enumerate(labels):
        label_details = labels[i]["projects"][project_id]["labels"][0]

        # Seconds to label tile
        seconds_to_label_tile = label_details["performance_details"]["seconds_to_create"]

        # Priority for tiles
        priority = labels[i]["projects"][project_id]["project_details"]["priority"]
        
        # Number of annotations excluding road lines
        polygon_details = label_details["annotations"]["objects"]
        annotations_num = 0
        for poly in polygon_details:
            if poly["name"] != "Road Line":
                annotations_num += 1

        annotator_email = label_details["label_details"]["created_by"]

        if annotator_email not in annotators.keys():
            annotators[annotator_email] = {"seconds": seconds_to_label_tile, "tiles": 1, "polygons": annotations_num}
        else:
            annotators[annotator_email]["seconds"] += seconds_to_label_tile
            annotators[annotator_email]["tiles"] += 1
            annotators[annotator_email]["polygons"] += annotations_num
    

        # Task performace details
        task[str(priority)]["seconds"] += seconds_to_label_tile
        task[str(priority)]["tiles"] += 1
        task[str(priority)]["annotations"] += annotations_num

    return annotators

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='annotator statistics', description='This program computes the annotators performance.')
    parser.add_argument('--api_key_file', type=str, help='The path to the api key file.')
    parser.add_argument('--project_key', type=str, help='The labelbox project key.')
    parser.add_argument('--annotators_file', type=str, help='The path to csv file where the annotator details will be saved.')
    parser.add_argument('--task_file', type=str, help='The path to csv file where the task detail will be saved.')
    args = parser.parse_args()

    #Load the API Key
    f = open(args.api_key_file)
    API_KEY = f.readlines()[0].replace(" ", "").replace("\n", "").replace("\r", "")
    f.close()

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

    filters={
        "workflow_status": "Done"
    }

    # Export Done Labels
    print("Exporting Done Labels from Project ....\n")
    export_labels = project.export_v2(params=params, filters=filters)
    export_labels.wait_till_done()
    if export_labels.errors:
        print(export_labels.errors)
    export_json = export_labels.result
    annotator_info = {}
    task_info = {"1": {"seconds": 0, "tiles": 0, "annotations": 0}, 
                 "2": {"seconds": 0, "tiles": 0, "annotations": 0}, 
                 "3": {"seconds": 0, "tiles": 0, "annotations": 0}, 
                 "4": {"seconds": 0, "tiles": 0, "annotations": 0}, 
                 "5": {"seconds": 0, "tiles": 0, "annotations": 0}}
    annotator_info = get_annotator_stats(PROJECT_ID, export_json, annotator_info, task_info)

    # Export InReview Labels
    filters={
        "workflow_status": "InReview"
    }
    print("Exporting InReview Labels from Project ....\n")
    export_labels = project.export_v2(params=params, filters=filters)
    export_labels.wait_till_done()
    if export_labels.errors:
        print(export_labels.errors)
    export_json = export_labels.result
    annotator_info = get_annotator_stats(PROJECT_ID, export_json, annotator_info, task_info)

    # Export InRework Labels
    filters={
        "workflow_status": "InRework"
    }
    print("Exporting InRework Labels from Project ....\n")
    export_labels = project.export_v2(params=params, filters=filters)
    export_labels.wait_till_done()
    if export_labels.errors:
        print(export_labels.errors)
    export_json = export_labels.result
    annotator_info = get_annotator_stats(PROJECT_ID, export_json, annotator_info, task_info)

    print("Saving Annotator Stats to csv file: " + str(args.annotators_file))
    annotator_df = pd.DataFrame.from_dict(annotator_info, orient="index")
    annotator_df.to_csv(args.annotators_file)

    print("Saving Task details to csv file: " + str(args.task_file))
    task_df = pd.DataFrame.from_dict(task_info, orient="index")
    task_df.to_csv(args.task_file)