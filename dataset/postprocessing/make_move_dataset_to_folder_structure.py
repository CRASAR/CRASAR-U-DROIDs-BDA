import os

import json
import shutil
import argparse

from dataset.constants import (
    DATASET_BASE_NAME,
    IMAGERY_FOLDER_NAME,
    UAS_FOLDER_NAME,
    ANNOTATIONS_FOLDER_NAME,
    BDA_FOLDER_NAME,
    RDA_FOLDER_NAME,
    BDA_ADJ_FOLDER_NAME,
    RDA_ADJ_FOLDER_NAME,
    TRAIN_FOLDER_NAME,
    TEST_FOLDER_NAME,
    ORTHO_EVENT,
    TRAIN_EVENTS,
    TEST_EVENTS,
    SATELLITE_FOLDER_NAME,
    CREWED_FOLDER_NAME,
)

VIEW_FOLDER_NAME = {
    "suas": UAS_FOLDER_NAME,
    "crewed": CREWED_FOLDER_NAME,
    "satellite": SATELLITE_FOLDER_NAME,
}

EXCLUDED_ORTHOS = [
    "1001-Harlem-Heights.geo.tif_20220929a_RGB.geo.tif",
    "1001-Iona-Point.geo.tif_20220930d_RGB.geo.tif",
    "1001-McGregor-College-Pkwy-South.3.geo.tif_20220930d_RGB.geo.tif",
    "1002-Boca-Grande.1.geo.tif_20221001b_RGB.geo.tif",
    "1001-Kennedy-Green-Mobile-Homes.geo.tif_10400100703E5500-visual.tif.geo.tif",
    "10132018-MexicoBeach.geo.tif_103001008699D200.tif.geo.tif",
    "10132018-MexicoBeach.geo.tif_1030010087C1A800.tif.geo.tif",
    "10132018-MexicoBeach.geo.tif_105001001292E300.tif.geo.tif",
    "10142018-MexicoBeach.geo.tif_103001008699D200.tif.geo.tif",
    "10142018-MexicoBeach.geo.tif_1030010087C1A800.tif.geo.tif"
]


def check_is_train(source_ortho):
    ortho_title = os.path.split(source_ortho)[1]
    event = ORTHO_EVENT[ortho_title]
    if event in TRAIN_EVENTS and (not event in TEST_EVENTS):
        return True
    if event in TEST_EVENTS and (not event in TRAIN_EVENTS):
        return False
    raise Exception(
        'Error: Event "'
        + str(event)
        + '" is not in either the train or test event list.'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="make_move_dataset_to_folder_structure",
        description="This program generates a folder structure and moves the annotations/data into the right places.",
    )
    parser.add_argument(
        "--ortho_dir_path",
        type=str,
        help="The path to the ortho path map file",
    )
    parser.add_argument(
        "--bda_annotations_path_map",
        type=str,
        help="The to the bda annotations path map file",
    )
    parser.add_argument(
        "--rda_annotations_path_map",
        type=str,
        help="The to the bda annotations path map file",
        default=None,
    )
    parser.add_argument(
        "--bda_adj_annotations_path_map",
        type=str,
        help="The to the adj annotations path map file",
    )
    parser.add_argument(
        "--rda_adj_annotations_path_map",
        type=str,
        help="The to the bda annotations path map file",
        default=None,
    )
    parser.add_argument(
        "--output_folder_location",
        type=str,
        help="The path to where the output folders and files should be written",
    )
    parser.add_argument(
        "--view_source", type=str, help="View source (crewed, satellite, uas)"
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Whether directory should be restructured",
    )
    args = parser.parse_args()

    train_prefix = os.path.join(
        args.output_folder_location, DATASET_BASE_NAME, TRAIN_FOLDER_NAME
    )
    test_prefix = os.path.join(
        args.output_folder_location, DATASET_BASE_NAME, TEST_FOLDER_NAME
    )

    print("Creating Folder Structure")
    assert len(DATASET_BASE_NAME) > 1

    base_dir_path = os.path.join(args.output_folder_location, DATASET_BASE_NAME)
    if os.path.exists(base_dir_path) and args.restart:
        shutil.rmtree(base_dir_path)
        os.mkdir(base_dir_path)

    for prefix in [train_prefix, test_prefix]:
        if not os.path.exists(prefix):
            os.mkdir(prefix)
            os.mkdir(os.path.join(prefix, IMAGERY_FOLDER_NAME))
            os.mkdir(os.path.join(prefix, ANNOTATIONS_FOLDER_NAME))

        if os.path.exists(
            os.path.join(
                prefix, IMAGERY_FOLDER_NAME, VIEW_FOLDER_NAME[args.view_source]
            )
        ):
            shutil.rmtree(
                os.path.join(
                    prefix, IMAGERY_FOLDER_NAME, VIEW_FOLDER_NAME[args.view_source]
                )
            )
        if os.path.exists(
            os.path.join(
                prefix, ANNOTATIONS_FOLDER_NAME, VIEW_FOLDER_NAME[args.view_source]
            )
        ):
            shutil.rmtree(
                os.path.join(
                    prefix, ANNOTATIONS_FOLDER_NAME, VIEW_FOLDER_NAME[args.view_source]
                )
            )

        os.mkdir(
            os.path.join(
                prefix, IMAGERY_FOLDER_NAME, VIEW_FOLDER_NAME[args.view_source]
            )
        )
        os.mkdir(
            os.path.join(
                prefix, ANNOTATIONS_FOLDER_NAME, VIEW_FOLDER_NAME[args.view_source]
            )
        )
        os.mkdir(
            os.path.join(
                prefix,
                ANNOTATIONS_FOLDER_NAME,
                VIEW_FOLDER_NAME[args.view_source],
                BDA_FOLDER_NAME,
            )
        )
        os.mkdir(
            os.path.join(
                prefix,
                ANNOTATIONS_FOLDER_NAME,
                VIEW_FOLDER_NAME[args.view_source],
                BDA_ADJ_FOLDER_NAME,
            )
        )

        if args.rda_annotations_path_map is not None:
            os.mkdir(
                os.path.join(
                    prefix,
                    ANNOTATIONS_FOLDER_NAME,
                    VIEW_FOLDER_NAME[args.view_source],
                    RDA_FOLDER_NAME,
                )
            )
            os.mkdir(
                os.path.join(
                    prefix,
                    ANNOTATIONS_FOLDER_NAME,
                    VIEW_FOLDER_NAME[args.view_source],
                    RDA_ADJ_FOLDER_NAME,
                )
            )

    print("Parsing BDA Annotations Path Map...")
    f = open(args.bda_annotations_path_map, "r")
    bda_annotations_path_map = json.loads(f.read())
    f.close()

    f = open(args.ortho_dir_path, "r")
    ortho_path_map = json.loads(f.read())
    f.close()

    if args.rda_annotations_path_map is not None:
        print("Parsing RDA Annotations Path Map...")
        f = open(args.rda_annotations_path_map, "r")
        rda_annotations_path_map = json.loads(f.read())
        f.close()

    print("Parsing BDA ADJ Annotations Path Map...")
    f = open(args.bda_adj_annotations_path_map, "r")
    bda_adjustments_path_map = json.loads(f.read())
    f.close()

    print(bda_adjustments_path_map)

    if args.rda_adj_annotations_path_map is not None:
        print("Parsing RDA ADJ Annotations Path Map...")
        f = open(args.rda_adj_annotations_path_map, "r")
        rda_adjustments_path_map = json.loads(f.read())
        f.close()

    print("Starting ortho and BDA annotations migration...")
    for k, annotations_file in bda_annotations_path_map.items():
        source_ortho = k.replace(".json", "")
        ortho_path = ortho_path_map[source_ortho]

        path_prefix = None
        if check_is_train(source_ortho):
            path_prefix = train_prefix
        else:
            path_prefix = test_prefix

        ortho_name = os.path.split(ortho_path)[-1]
        annotations_name = None
        adjustments_name = None
        valid_sample = False
        try:
            annotations_name = os.path.split(annotations_file)[-1]
            if ortho_path in bda_adjustments_path_map.keys():
                adjustments_name = os.path.split(bda_adjustments_path_map[ortho_path])[-1]
                adjustments = bda_adjustments_path_map[ortho_path]
            else:
                adjustments_name = os.path.split(bda_adjustments_path_map[source_ortho])[-1]
                adjustments = bda_adjustments_path_map[source_ortho]
            valid_sample = True
        except KeyError:
            print(
                "Skipping",
                source_ortho,
                "becasue the complete data payload could not be found (PathMap KeyError)",
            )
        if ortho_name in EXCLUDED_ORTHOS:
            valid_sample = False
            print("Skipping", ortho_name, "since it was excluded...")

        if valid_sample:
            ortho_dst = os.path.join(
                path_prefix,
                IMAGERY_FOLDER_NAME,
                VIEW_FOLDER_NAME[args.view_source],
                ortho_name,
            )
            annotations_bda_dst = os.path.join(
                path_prefix,
                ANNOTATIONS_FOLDER_NAME,
                VIEW_FOLDER_NAME[args.view_source],
                BDA_FOLDER_NAME,
                annotations_name,
            )
            annotations_adj_dst = os.path.join(
                path_prefix,
                ANNOTATIONS_FOLDER_NAME,
                VIEW_FOLDER_NAME[args.view_source],
                BDA_ADJ_FOLDER_NAME,
                adjustments_name,
            )

            print("\tCopying data for", ortho_name)
            shutil.copyfile(ortho_path, ortho_dst)
            shutil.copyfile(annotations_file, annotations_bda_dst)
            shutil.copyfile(adjustments, annotations_adj_dst)

    if args.rda_annotations_path_map is not None:
        print("Starting RDA annotations migration...")
        for source_ortho, annotations_file in rda_annotations_path_map.items():

            ortho_path = ortho_path_map[source_ortho]

            path_prefix = None
            if check_is_train(source_ortho):
                path_prefix = train_prefix
            else:
                path_prefix = test_prefix

            ortho_name = os.path.split(ortho_path)[-1]
            annotations_name = None
            adjustments_name = None
            valid_sample = False
            try:
                annotations_name = os.path.split(annotations_file)[-1]
                if ortho_path in rda_adjustments_path_map.keys():
                    adjustments_name = os.path.split(
                        rda_adjustments_path_map[ortho_path]
                    )[-1]
                    rda_adjustments = rda_adjustments_path_map[ortho_path]
                else:
                    adjustments_name = os.path.split(
                        rda_adjustments_path_map[source_ortho]
                    )[-1]
                    rda_adjustments = rda_adjustments_path_map[source_ortho]
                valid_sample = True
            except KeyError:
                print(
                    "Skipping",
                    source_ortho,
                    "becasue the complete data payload could not be found (PathMap KeyError)",
                )

            if valid_sample:
                annotations_rda_dst = os.path.join(
                    path_prefix,
                    ANNOTATIONS_FOLDER_NAME,
                    VIEW_FOLDER_NAME[args.view_source],
                    RDA_FOLDER_NAME,
                    annotations_name,
                )
                annotations_adj_dst = os.path.join(
                    path_prefix,
                    ANNOTATIONS_FOLDER_NAME,
                    VIEW_FOLDER_NAME[args.view_source],
                    RDA_ADJ_FOLDER_NAME,
                    adjustments_name,
                )

                print("\tCopying data for", ortho_name)
                shutil.copyfile(annotations_file, annotations_rda_dst)
                shutil.copyfile(
                    rda_adjustments, annotations_adj_dst
                )
