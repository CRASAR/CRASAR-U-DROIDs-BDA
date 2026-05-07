import os
import json
import argparse
from dataset.utils.view_utils import get_view_id
from dataset.constants import ORTHO_EVENT, TEST_EVENTS


def add_annotations(annotations, boundary_name, test_views, train_views, file_name):
    view_id = get_view_id()

    current_event = ORTHO_EVENT[boundary_name]
    if current_event in TEST_EVENTS:
        test_views[view_id] = []
        for annotation in annotations:
            annotation["boundary"] = boundary_name
            annotation["filename"] = file_name
            test_views[view_id].append(annotation)
    else:
        train_views[view_id] = []
        for annotation in annotations:
            annotation["boundary"] = boundary_name
            annotation["filename"] = file_name
            train_views[view_id].append(annotation)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="combine_annotations",
        description="This program combines all BDA annotations across boundaries into one file by view ID.",
    )
    parser.add_argument(
        "--suas_annotations_path_map",
        type=str,
        required=True,
        help="The path to the annotations file path map.",
    )
    parser.add_argument(
        "--satellite_annotations_folder",
        type=str,
        help="Path to satellite annotations folder.",
        default=None,
    )
    parser.add_argument(
        "--crewed_annotations_folder",
        type=str,
        help="Path to crewed annotations folder.",
        default=None,
    )
    parser.add_argument(
        "--output_combined_annotations",
        type=str,
        required=True,
        help="Path to the output combined annotations file.",
    )

    args = parser.parse_args()

    print("Loading suas Annotation File Path Mapping...")
    with open(args.suas_annotations_path_map, "r") as f:
        annotation_path_map = json.load(f)

    all_views_train = {}
    all_views_test = {}

    suas_test_views = {}
    suas_train_views = {}

    crewed_test_views = {}
    crewed_train_views = {}

    satellite_test_views = {}
    satellite_train_views = {}

    print("Processing annotations...")
    for geotiff_path, annotation_path in annotation_path_map.items():
        boundary_name = os.path.basename(geotiff_path)

        # Load SUAS annotations
        filename = os.path.basename(annotation_path)
        with open(annotation_path, "r") as f:
            suas_annotations = json.load(f)
        add_annotations(
            suas_annotations, boundary_name, suas_test_views, suas_train_views, filename
        )

        # Load satellite annotations
        if args.satellite_annotations_folder:
            for root, dirs, files in os.walk(args.satellite_annotations_folder):
                for file in files:
                    if boundary_name in file:
                        with open(os.path.join(root, file), "r") as f:
                            satellite_annotations = json.load(f)
                        add_annotations(
                            satellite_annotations,
                            boundary_name,
                            satellite_test_views,
                            satellite_train_views,
                            file,
                        )

        # Load crewed annotations
        if args.crewed_annotations_folder:
            for root, dirs, files in os.walk(args.crewed_annotations_folder):
                for file in files:
                    if boundary_name in file:
                        with open(os.path.join(root, file), "r") as f:
                            crewed_annotations = json.load(f)
                        add_annotations(
                            crewed_annotations,
                            boundary_name,
                            crewed_test_views,
                            crewed_train_views,
                            file,
                        )

    print(
        f"Total views collected: {len(suas_test_views) + len(suas_train_views)+ len(crewed_train_views)+ len(crewed_test_views) + len(satellite_train_views) + len(satellite_test_views)}"
    )
    print(
        "\t Train Views: ",
        len(suas_train_views) + len(crewed_train_views) + len(satellite_train_views),
    )
    print("\t\t sUAS Views: ", len(suas_train_views))
    print("\t\t Crewed Views: ", len(crewed_train_views))
    print("\t\t Satellite Views: ", len(satellite_train_views))
    print(
        "\t Test Views: ",
        len(suas_test_views) + len(crewed_test_views) + len(satellite_test_views),
    )
    print("\t\t sUAS Views: ", len(suas_test_views))
    print("\t\t Crewed Views: ", len(crewed_test_views))
    print("\t\t Satellite Views: ", len(satellite_test_views))

    all_views_train = dict(
        suas_train_views | crewed_train_views | satellite_train_views
    )
    all_views_test = dict(suas_test_views | crewed_test_views | satellite_test_views)
    print(
        f"Total views combined (Train + Test): {len(all_views_test) + len(all_views_train)}"
    )
    print(f"\tTotal views combined (Train): {len(all_views_train)}")
    print(f"\tTotal views combined (Test): {len(all_views_test)}")

    os.makedirs(args.output_combined_annotations, exist_ok=True)
    with open(
        os.path.join(args.output_combined_annotations, "suas_train.json"), "w"
    ) as f:
        json.dump(suas_train_views, f, indent=2)
    with open(
        os.path.join(args.output_combined_annotations, "suas_test.json"), "w"
    ) as f:
        json.dump(suas_test_views, f, indent=2)

    with open(
        os.path.join(args.output_combined_annotations, "crewed_train.json"), "w"
    ) as f:
        json.dump(crewed_train_views, f, indent=2)
    with open(
        os.path.join(args.output_combined_annotations, "crewed_test.json"), "w"
    ) as f:
        json.dump(crewed_test_views, f, indent=2)

    with open(
        os.path.join(args.output_combined_annotations, "satellite_train.json"), "w"
    ) as f:
        json.dump(satellite_train_views, f, indent=2)
    with open(
        os.path.join(args.output_combined_annotations, "satellite_test.json"), "w"
    ) as f:
        json.dump(satellite_test_views, f, indent=2)

    with open(os.path.join(args.output_combined_annotations, "train.json"), "w") as f:
        json.dump(all_views_train, f, indent=2)
    with open(os.path.join(args.output_combined_annotations, "test.json"), "w") as f:
        json.dump(all_views_test, f, indent=2)

    print(f"Done. Output saved to {args.output_combined_annotations}")
