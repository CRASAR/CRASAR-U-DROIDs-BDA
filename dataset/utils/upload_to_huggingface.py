from huggingface_hub import HfApi

print("Initializing API")
token = ""
api = HfApi(token=token)

print("Starting upload")
api.upload_folder(
    folder_path="H:/tmp/gold/CRASAR-U-DROIDs",
    repo_id="CRASAR/CRASAR-U-DROIDs",
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
)