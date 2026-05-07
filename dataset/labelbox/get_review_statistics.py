import json
import argparse
import math
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Get Review Statistics',
                                     description='This program counts the number of files that were either rejected, or reworked, during the review process.')
    parser.add_argument('--json_data_file',
                        type=str,
                        help='The data file downloaded from labelbox.')
    parser.add_argument('--project_id',
                        type=str,
                        help='The ID for the label box project review statistics to be loaded.')
    parser.add_argument('--review_update_buffer_seconds',
                        type=int,
                        help="The time between the time spent reviewing a label and updated vs approved for an approval to count as an update",
                        default=1)
    args = parser.parse_args()

    INITIAL_LABELING_TASK = "Initial labeling task"
    PROJECTS = "projects"

    with open(args.json_data_file, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]

    count_rejected = 0
    count_edited_in_review = 0
    count_edited_in_review_or_rejected = 0
    count_edited_in_review_and_rejected = 0
    count_accepted_without_edit = 0
    count_total = 0
    for sample in data:
        created_at = datetime.strptime(sample[PROJECTS][args.project_id]["labels"][0]["label_details"]["created_at"], '%Y-%m-%dT%H:%M:%S.%f+00:00')
        updated_at = datetime.strptime(sample[PROJECTS][args.project_id]["labels"][0]["label_details"]["updated_at"], '%Y-%m-%dT%H:%M:%S.%f+00:00')

        workflow_steps = sample[PROJECTS][args.project_id]["project_details"]["workflow_history"]

        actions = [step["action"] for step in workflow_steps]
        next_task_names = []
        for step in workflow_steps:
            if "next_task_name" in step.keys():
                next_task_names.append(step["next_task_name"])
            else:
                next_task_names.append(None)

        #Labelbox reverse indexes this list. Index 0 represents the most recent action
        last_action_time = datetime.strptime(workflow_steps[0]["created_at"], '%Y-%m-%dT%H:%M:%S.%f+00:00')

        seconds_to_review = sample[PROJECTS][args.project_id]["labels"][0]["performance_details"]["seconds_to_review"]

        rejected = False
        edited_in_review = False

        if len(actions) > 3 and not rejected:
            rejected = True

        if math.fabs((updated_at - last_action_time).total_seconds()) < seconds_to_review + args.review_update_buffer_seconds:
            edited_in_review = True

        count_total += 1
        count_rejected += 1 if rejected else 0
        count_edited_in_review += 1 if edited_in_review else 0
        count_edited_in_review_or_rejected += 1 if edited_in_review or rejected else 0
        count_edited_in_review_and_rejected += 1 if edited_in_review and rejected else 0
        count_accepted_without_edit += 1 if not (edited_in_review or rejected) else 0

    print("count_rejected", count_rejected)
    print("count_edited_in_review", count_edited_in_review)
    print("count_edited_in_review_or_rejected", count_edited_in_review_or_rejected)
    print("count_accepted_without_edit", count_accepted_without_edit)
    print("count_edited_in_review_and_rejected", count_edited_in_review_and_rejected)
    print("count_total", count_total)
