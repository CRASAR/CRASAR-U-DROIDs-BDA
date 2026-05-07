import os
import json
import argparse
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.special import rel_entr
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.proportion import proportions_ztest

from dataset.constants import (
    NO_DAMAGE,
    MINOR_DAMAGE,
    MAJOR_DAMAGE,
    DESTROYED,
    UNCLASSIFIED,
    OBSCURED,
    BDA_DAMAGE_CLASSES,
)

OBSCURED = "Obscured"

DAYS_AFTER_SUAS_ORTHO = "days_after_suas_ortho"
MASK_SOURCE = "mask_source"
PRE_OR_POST_EVENT = "pre_or_post_event"
FILENAME = "filename"
IMAGERY_SOURCE_VEHICLE = "imagery_source_vehicle"


def cohens_d(group1, group2):
    diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt(
        (
            (len(group1) - 1) * np.std(group1, ddof=1) ** 2
            + (len(group2) - 1) * np.std(group2, ddof=1) ** 2
        )
        / (len(group1) + len(group2) - 2)
    )
    return diff / pooled_std


def compute_odds_ratio(suas_labels, sat_labels, ignore_sat_obscured):
    suas_label_counts = defaultdict(lambda: 0)
    sat_label_counts = defaultdict(lambda: 0)
    for suas_label, sat_label in zip(suas_labels.values(), sat_labels.values()):
        if (ignore_sat_obscured and sat_label != OBSCURED) or not ignore_sat_obscured:
            suas_label_counts[suas_label] += 1
            sat_label_counts[sat_label] += 1

    n_observations = [len(suas_labels), len(sat_labels)]

    odds_ratios = {}
    for label in [NO_DAMAGE, MINOR_DAMAGE, MAJOR_DAMAGE, DESTROYED, UNCLASSIFIED]:
        table = [
            [suas_label_counts[label], sat_label_counts[label]],
            [
                n_observations[0] - suas_label_counts[label],
                n_observations[1] - sat_label_counts[label],
            ],
        ]
        odds_ratios[label] = (table[0][0] * table[1][1]) / (table[0][1] * table[1][0])
    return odds_ratios


def remove_obscured_labels(labeled_data):
    result = []
    for data in labeled_data:
        result.append([])
        for building in data:
            if building["label"] != OBSCURED and building["label"] != OBSCURED.lower():
                result[-1].append(building)
    return result


def chi_squared_test(suas_labels, sat_labels, ignore_sat_obscured):
    suas_label_counts = defaultdict(lambda: 0)
    sat_label_counts = defaultdict(lambda: 0)
    for suas_label, sat_label in zip(suas_labels.values(), sat_labels.values()):
        if (ignore_sat_obscured and sat_label != OBSCURED) or not ignore_sat_obscured:
            suas_label_counts[suas_label] += 1
            sat_label_counts[sat_label] += 1

    data = []
    for label in set(list(suas_label_counts.keys()) + list(sat_label_counts.keys())):
        data.append([suas_label_counts[label], sat_label_counts[label]])

    _, p, _, _ = chi2_contingency(data)

    return p


def z_test_per_label(suas_labels, sat_labels, ignore_sat_obscured):
    p_values = {}

    suas_label_counts = defaultdict(lambda: 0)
    sat_label_counts = defaultdict(lambda: 0)
    for suas_label, sat_label in zip(suas_labels.values(), sat_labels.values()):
        if (ignore_sat_obscured and sat_label != OBSCURED) or not ignore_sat_obscured:
            suas_label_counts[suas_label] += 1
            sat_label_counts[sat_label] += 1

    n_observations = [len(suas_labels), len(sat_labels)]

    for label in [NO_DAMAGE, MINOR_DAMAGE, MAJOR_DAMAGE, DESTROYED, UNCLASSIFIED]:
        data = [suas_label_counts[label], sat_label_counts[label]]
        p_values[label] = proportions_ztest(data, n_observations)[1]
    return p_values


def get_swapped_unclassified_label(base_label, new_label, ignore_lone_unclassified):
    if base_label is None:
        return new_label
    if (
        new_label == UNCLASSIFIED
        and base_label != UNCLASSIFIED
        and ignore_lone_unclassified
    ):
        return base_label
    return new_label


def compute_paired_difference_views(
    temporal_views, field, ignore_unclassified=True, ignore_obscured=True
):
    sat_view_1 = []
    sat_view_2 = []

    for (day1, view1), (day2, view2) in combinations(temporal_views.items(), 2):
        buildings_day2 = {building["id"]: building for building in view2}
        delta_days = np.abs(day1 - day2)
        for building1 in view1:
            building_id = building1["id"]
            if building_id in buildings_day2:
                value = None
                valid = False

                if field == "days":
                    value = delta_days
                    valid = True
                else:
                    try:
                        field1 = building1["view_properties"][field]
                        field2 = buildings_day2[building_id]["view_properties"][field]
                        value = np.abs(field1 - field2)
                        valid = True
                    except KeyError:
                        pass
                if valid:
                    label1 = building1["label"]
                    label2 = buildings_day2[building_id]["label"]
                    if not (
                        (
                            (label1 == UNCLASSIFIED or label2 == UNCLASSIFIED)
                            and ignore_unclassified
                        )
                        or (
                            (label1 == OBSCURED or label2 == OBSCURED)
                            and ignore_obscured
                        )
                    ):
                        sat_view_1.append([label1, value])
                        sat_view_2.append([label2, value])

    agree_dist = []
    disagree_dist = []
    for i, (label1, field) in enumerate(sat_view_1):
        if label1 == sat_view_2[i][0]:
            agree_dist.append(field)
        else:
            disagree_dist.append(field)
    return agree_dist, disagree_dist


def group_buildings_temporally(satellite_data, multiview_info):
    group_temporal = {}

    for data in satellite_data:
        for building in data:
            source_filename_label = building[FILENAME].split("\\")[-1].replace(".json", "")
            days_after_suas_ortho = multiview_info[
                multiview_info[FILENAME] == source_filename_label
            ][DAYS_AFTER_SUAS_ORTHO].iloc[0]
            if days_after_suas_ortho not in group_temporal.keys():
                group_temporal[days_after_suas_ortho] = []
            group_temporal[days_after_suas_ortho].append(building)

    return group_temporal


def get_best_oracle_label_for_building(
    suas_data, satellite_data, valid_ids, ignore_lone_unclassified=True
):
    sUAS_labels = {}

    for data in suas_data:
        for building in data:
            if building["id"] in valid_ids:
                sUAS_labels[building["id"]] = building["label"]

    satellite_labels = defaultdict(lambda: None)
    for data in satellite_data:
        for building in data:
            if building["id"] in valid_ids:
                new_label = get_swapped_unclassified_label(
                    satellite_labels[building["id"]],
                    building["label"],
                    ignore_lone_unclassified,
                )
                if satellite_labels[building["id"]] is None:
                    satellite_labels[building["id"]] = new_label
                elif new_label == sUAS_labels[building["id"]]:
                    satellite_labels[building["id"]] = new_label

    return sUAS_labels, satellite_labels


def get_best_antioracle_label_for_building(
    suas_data, satellite_data, valid_ids, ignore_lone_unclassified=True
):
    sUAS_labels = {}

    for data in suas_data:
        for building in data:
            if building["id"] in valid_ids:
                sUAS_labels[building["id"]] = building["label"]

    satellite_labels = defaultdict(lambda: None)
    for data in satellite_data:
        for building in data:
            if building["id"] in valid_ids:
                new_label = get_swapped_unclassified_label(
                    satellite_labels[building["id"]],
                    building["label"],
                    ignore_lone_unclassified,
                )
                if satellite_labels[building["id"]] is None:
                    satellite_labels[building["id"]] = new_label
                elif new_label != sUAS_labels[building["id"]]:
                    satellite_labels[building["id"]] = new_label

    return sUAS_labels, satellite_labels


def get_best_temporal_label_for_building(
    suas_data,
    satellite_data,
    valid_ids,
    multiview_info,
    sort_strategy="abs",
    ignore_lone_unclassified=True,
):
    sUAS_labels = {}

    for data in suas_data:
        for building in data:
            if building["id"] in valid_ids:
                sUAS_labels[building["id"]] = building["label"]

    satellite_labels = defaultdict(lambda: None)
    for data in satellite_data:
        for building in data:
            if building["id"] in valid_ids:
                if satellite_labels[building["id"]]:
                    new_label = get_swapped_unclassified_label(
                        satellite_labels[building["id"]][0],
                        building["label"],
                        ignore_lone_unclassified,
                    )
                new_label = building["label"]
                source_filename_label = building[FILENAME].split("\\")[-1].replace(".json", "")
                days_after_suas_ortho = multiview_info[
                    multiview_info[FILENAME] == source_filename_label
                ][DAYS_AFTER_SUAS_ORTHO].iloc[0]
                if sort_strategy == "abs":
                    days_after_suas_ortho = np.abs(days_after_suas_ortho)
                elif sort_strategy == "real":
                    pass
                if satellite_labels[building["id"]] is None:
                    satellite_labels[building["id"]] = [
                        new_label,
                        days_after_suas_ortho,
                    ]
                elif satellite_labels[building["id"]][1] > days_after_suas_ortho:
                    satellite_labels[building["id"]] = [
                        new_label,
                        days_after_suas_ortho,
                    ]

    resulting_sat_labels = {}
    for bld_id, value in satellite_labels.items():
        if not value is None:
            label, _ = value
            resulting_sat_labels[bld_id] = label

    return sUAS_labels, resulting_sat_labels


def get_intersecting_ids(annotations_data1, annotations_data2):
    bda_1_ids = []
    bda_2_ids = []
    # Count Damage Labels for sUAS data format
    for data in annotations_data1:
        for building in data:
            bda_1_ids.append(building["id"])

    # Count Damage Labels for Satellite data format
    for data in annotations_data2:
        for building in data:
            bda_2_ids.append(building["id"])

    valid_ids = list(set(bda_1_ids) & set(bda_2_ids))
    return valid_ids


def get_probability_of_disagreement(
    suas_id2labels, sat_id2labels, ignore_sat_obscured=False
):
    count_agree = 0
    count_disagree = 0
    for bld_id in suas_id2labels.keys():
        if (
            ignore_sat_obscured and sat_id2labels[bld_id] != OBSCURED
        ) or not ignore_sat_obscured:
            if suas_id2labels[bld_id] == sat_id2labels[bld_id]:
                count_agree += 1
            else:
                count_disagree += 1
    return count_disagree / (count_agree + count_disagree)


def compute_kl_divergence(damagecounts1, damagecounts2):
    bda1_dist = np.array(list(damagecounts1.values()), dtype=np.float64)
    bda2_dist = np.array(list(damagecounts2.values()), dtype=np.float64)
    bda1_dist /= np.sum(bda1_dist)
    bda2_dist /= np.sum(bda2_dist)

    # Compute KL Divergence
    return sum(rel_entr(bda1_dist, bda2_dist)), sum(rel_entr(bda2_dist, bda1_dist))


def compute_transistion_matrix(
    suas_id2labels, sat_id2labels, plot_folder, prefix="", ignore_obscured=True
):

    suas_labels = [suas_id2labels[bld_id] for bld_id in sat_id2labels]
    sat_labels = [sat_id2labels[bld_id] for bld_id in sat_id2labels]

    labels = BDA_DAMAGE_CLASSES[:]
    try:
        labels.remove(OBSCURED)
    except ValueError:
        pass

    transition_matrix = confusion_matrix(
        y_true=suas_labels, y_pred=sat_labels, labels=labels
    )

    conf_matrix = np.array(transition_matrix)
    _, ax = plt.subplots(figsize=(9, 9))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    plt.xticks(range(conf_matrix.shape[1]), labels, fontsize=13)
    plt.yticks(range(conf_matrix.shape[0]), labels, fontsize=13)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
            )
            plt.xlabel("Satellite", fontsize=15)
            plt.ylabel("Drone", fontsize=15)
    plt.title(
        prefix + " vs. Drone Confusion Matrix | N=" + str(len(sat_labels)), fontsize=18
    )
    print(
        "Saving Confusion Matrix at ",
        str(
            os.path.join(
                plot_folder, prefix + "suas_vs_satellite_transition_matrix.png"
            )
        ),
    )
    plt.savefig(
        os.path.join(plot_folder, prefix + "suas_vs_satellite_transition_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )


def compute_class_balances(
    suas_id2labels,
    sat_id2labels,
    plot_folder,
    valid_ids,
    file_prefix="",
    ignore_sat_obscured=False,
):
    bda1_damage_labels = {
        NO_DAMAGE: 0,
        MINOR_DAMAGE: 0,
        MAJOR_DAMAGE: 0,
        DESTROYED: 0,
        UNCLASSIFIED: 0,
        OBSCURED: 0,
    }
    bda2_damage_labels = {
        NO_DAMAGE: 0,
        MINOR_DAMAGE: 0,
        MAJOR_DAMAGE: 0,
        DESTROYED: 0,
        UNCLASSIFIED: 0,
        OBSCURED: 0,
    }

    ids_to_consider = []
    for bld_id in sat_id2labels.keys():
        if (
            ignore_sat_obscured and sat_id2labels[bld_id] != OBSCURED
        ) or not ignore_sat_obscured:
            ids_to_consider.append(bld_id)

    for bld_id in ids_to_consider:
        bda1_damage_labels[suas_id2labels[bld_id]] += 1
        bda2_damage_labels[sat_id2labels[bld_id]] += 1

    # Plot Damage Class Balances
    categories = list(bda1_damage_labels.keys())
    bda1_counts = list(bda1_damage_labels.values())
    bda2_counts = list(bda2_damage_labels.values())

    x = np.arange(len(categories))
    bar_width = 0.25

    _, ax = plt.subplots(figsize=(12, 8))
    bda1_bars = ax.bar(
        x - bar_width / 2, bda1_counts, bar_width, label="sUAS", color="b"
    )
    bda2_bars = ax.bar(
        x + bar_width / 2, bda2_counts, bar_width, label="Satellite", color="g"
    )

    ax.set_xlabel("Building Damage Classes", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.set_title(
        file_prefix + " Building Damage Class Counts: Drone vs Satellite", fontsize=16
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, rotation=45)
    ax.legend(fontsize=12)

    for bars in [bda1_bars, bda2_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
            )

    print(
        "Saving Class Balance Figure at ",
        str(
            os.path.join(
                plot_folder, file_prefix + "suas_vs_satellite_bda_class_balances.png"
            )
        ),
    )

    plt.savefig(
        os.path.join(
            plot_folder, file_prefix + "suas_vs_satellite_bda_class_balances.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    return bda1_damage_labels, bda2_damage_labels


def compute_mulistrategy_class_balances(
    suas_damage_counts, sat_multistrategy_damage_counts, plot_folder, valid_ids
):
    all_bars = {"Drone": suas_damage_counts}
    for strategy_name in sat_multistrategy_damage_counts.keys():
        all_bars[strategy_name] = sat_multistrategy_damage_counts[strategy_name]
        all_bars[strategy_name].pop("Obscured", None)

    all_bars["Drone"].pop("Obscured", None)
    categories = all_bars["Drone"].keys()

    num_buildings = sum(all_bars["Drone"][label] for label in all_bars["Drone"].keys())

    x = np.arange(len(categories))
    multicol_area = 1.0
    bar_width = multicol_area / (len(categories) + 1)

    _, ax = plt.subplots(figsize=(24, 8))
    all_plotted_bars = []
    hatches = [".", "/", "x", "|", "*", "-", "+"]
    for i, (strat_label, damage_label_counts) in enumerate(all_bars.items()):
        if i == 0:
            all_plotted_bars.append(
                ax.bar(
                    x + bar_width * i,
                    list(damage_label_counts.values()),
                    bar_width,
                    label=strat_label,
                    color="black",
                    edgecolor="white",
                    hatch=hatches[i],
                )
            )
        else:
            all_plotted_bars.append(
                ax.bar(
                    x + bar_width * i,
                    list(damage_label_counts.values()),
                    bar_width,
                    label=strat_label,
                    color="lightgrey",
                    edgecolor="black",
                    hatch=hatches[i],
                )
            )

    ax.set_xlabel("Building Damage Classes", fontsize=22)
    ax.set_ylabel("Counts", fontsize=20)
    ax.set_title(
        "Building Damage Class Counts: Drone vs Satellite ($N_{Buildings}$="
        + str(num_buildings)
        + ")",
        fontsize=30,
    )
    ax.set_xticks(x + (bar_width * (len(categories) - 2)) / 2)
    ax.set_xticklabels(categories, fontsize=18, rotation=0)
    ax.tick_params(axis="y", which="major", labelsize=16)
    ax.legend(fontsize=20)

    max_bar = 0
    for bars in all_plotted_bars:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
            )

            if height > max_bar:
                max_bar = height

    ax.set_ylim(0, max_bar * 1.2)

    print(
        "Saving Class Balance Figure at ",
        str(
            os.path.join(plot_folder, "suas_vs_satellite_strat_bda_class_balances.png")
        ),
    )
    plt.savefig(
        os.path.join(
            plot_folder, "suas_vs_satellite_multistrat_bda_class_balances.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )


def count_labels(labeled_polygons, label_to_count):
    count = 0
    for p in labeled_polygons:
        if p["label"] == label_to_count:
            count += 1
    return count


def plot_sat_views_per_building_histogram(
    satellite_data, valid_ids, plot_folder, file_prefix=""
):
    views_per_building = defaultdict(lambda: 0)
    for data in satellite_data:
        for building in data:
            if building["id"] in valid_ids:
                views_per_building[building["id"]] += 1
    x = list(views_per_building.values())

    tick_positions = [1.5, 2.5, 3.5, 4.5]

    _, ax = plt.subplots(figsize=(12, 8))
    n, _, _ = plt.hist(x, bins=[1, 2, 3, 4, 5], color="grey")
    ax.set_xlabel("Number of Views", fontsize=20)
    ax.set_ylabel("Number of Buildings", fontsize=20)
    ax.set_title(
        "Histogram of Satellite Views per Building\n($N_{Buildings}$="
        + str(len(x))
        + " | $N_{Views}$="
        + str(int(sum(x)))
        + ")",
        fontsize=25,
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([1, 2, 3, 4])
    ax.tick_params(axis="both", which="major", labelsize=16)

    for x, val in zip(tick_positions, n):
        ax.annotate(
            f"{int(val)}",
            xy=(x, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    print(
        "Saving view histogram figure at ",
        str(os.path.join(plot_folder, file_prefix + "view_histogram.png")),
    )
    plt.savefig(
        os.path.join(
            plot_folder, file_prefix + "satellite_view_counts_per_building.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )


def plot_sat_view_properites_per_building_histogram(
    satellite_data, valid_ids, view_property, plot_folder, file_prefix=""
):
    properties_of_views = []
    for data in satellite_data:
        for building in data:
            if building["id"] in valid_ids and "view_properties" in building.keys():
                properties_of_views.append(building["view_properties"][view_property])

    _, ax = plt.subplots(figsize=(12, 8))
    plt.hist(properties_of_views, bins=100)
    ax.set_xlabel(view_property, fontsize=14)
    ax.set_ylabel("Number of Views", fontsize=14)
    ax.set_title(
        "Histogram of "
        + view_property
        + " per Building View ($N_{Views}$="
        + str(len(properties_of_views))
        + ")",
        fontsize=16,
    )

    print(
        "Saving view histogram figure at ",
        str(os.path.join(plot_folder, view_property + "view_histogram.png")),
    )
    plt.savefig(
        os.path.join(plot_folder, view_property + "view_properties.png"),
        dpi=300,
        bbox_inches="tight",
    )


def get_coincident_buildings_per_ortho(valid_ids, suas_data):
    file_to_coincident_count = {}
    for filename, data in suas_data.items():
        file_to_coincident_count[filename] = 0
        for building in data:
            if building["id"] in valid_ids:
                file_to_coincident_count[filename] += 1
    return file_to_coincident_count


def get_satellite_building_counts(satellite_data, valid_ids):
    view_count = 0
    view_count_with_properties = 0
    for data in satellite_data:
        for building in data:
            if building["id"] in valid_ids:
                view_count += 1
                if "view_properties" in building.keys():
                    view_count_with_properties += 1
    return view_count, view_count_with_properties


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="compute_satellite_BDA_dataset_stats",
        description="This program computes high level statistics about the Satellite BDA annotations.",
    )
    parser.add_argument(
        "--satellite_annotations_path_map",
        type=str,
        help="The path to the satellite annotations file path map.",
    )
    parser.add_argument(
        "--drone_annotations_path_map",
        type=str,
        help="The path to the suas annotations file path map.",
    )
    parser.add_argument(
        "--output_stats_folder_path",
        type=str,
        help="The path to the output statistics file.",
    )
    parser.add_argument(
        "--multiview_stats_file_path",
        type=str,
        help="The path to the multiview information.",
    )
    args = parser.parse_args()
    try:
        os.makedirs(args.output_stats_folder_path)
    except FileExistsError as e:
        pass

    f = open(args.satellite_annotations_path_map)
    data = f.read()
    sat_annotations_path_map = json.loads(data)
    f.close()

    ortho_label_counts = {}

    for geotif_path, annotation_path in sat_annotations_path_map.items():

        ortho_local_title = os.path.split(geotif_path)[-1]

        if "090403-Lancaster-Canyon-Gate.geo.tif" in geotif_path or \
           "20230830-SteinhatcheeRiver.geo.tif" in geotif_path or \
           "20230831-Jena-SteinhatcheeRiverSouth.geo.tif" in geotif_path:
           annotation_path = None

        if annotation_path is not None:
            # Load the annotations
            print("Loading the Satellite BDA annotations from:", annotation_path)

            f = open(annotation_path, "r")
            annotations_data = json.loads(f.read())
            f.close()

            ortho_label_counts[ortho_local_title] = {
                "1 - no damage": count_labels(annotations_data, NO_DAMAGE),
                "2 - minor damage": count_labels(annotations_data, MINOR_DAMAGE),
                "3 - major damage": count_labels(annotations_data, MAJOR_DAMAGE),
                "4 - destroyed": count_labels(annotations_data, DESTROYED),
                "5 - un-classified": count_labels(annotations_data, UNCLASSIFIED),
                "6 - obscured": count_labels(annotations_data, OBSCURED),
                "7 - total": len(annotations_data),
            }

            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["1 - no damage"],
                "polygons with the no damage label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["2 - minor damage"],
                "polygons with the minor damage label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["3 - major damage"],
                "polygons with the major damage label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["4 - destroyed"],
                "polygons with the destroyed label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["5 - un-classified"],
                "polygons with the un-classified label",
            )
            print(
                "\tFound",
                ortho_label_counts[ortho_local_title]["6 - obscured"],
                "polygons with the obscured label",
            )

        else:
            print("Warning! Could not find Annotations Path for ", str(geotif_path))

    print("Saving summary statistics file...")
    stats = pd.DataFrame(ortho_label_counts)
    stats.transpose().to_csv(
        os.path.join(args.output_stats_folder_path, "satellite_bda_damage_counts.csv")
    )
    print("Done...")

    if args.drone_annotations_path_map is not None:
        drone_annotations_path_map = json.load(open(args.drone_annotations_path_map))

        # Read multiview csv for metadata information
        multiview_df = pd.read_csv(args.multiview_stats_file_path)

        print("Computing sUAS vs Satellite Statistics....")

        suas_data = {}
        satellite_data = []

        for geotif_path, drone_annotation_path in drone_annotations_path_map.items():
            # Load the annotations
            print("Loading the sUAS BDA annotations from:", drone_annotation_path)
            f = open(drone_annotation_path, "r")
            drone_annotations_data = json.loads(f.read())
            f.close()

            suas_data[geotif_path] = drone_annotations_data

        for geotif_path, sat_annotation_path in sat_annotations_path_map.items():
            print("Loading the Satellite BDA annotations from:", sat_annotation_path)
            try:
                f = open(sat_annotation_path, "r")
                closest_annotation_path_annotations_data = json.loads(f.read())
                f.close()

                satellite_data.append(closest_annotation_path_annotations_data)
            except TypeError as e:
                print(
                    "Skipping",
                    geotif_path,
                    "->",
                    sat_annotation_path,
                    "because of",
                    type(e),
                )

        non_obscured_sat_data = remove_obscured_labels(satellite_data)

        valid_ids = get_intersecting_ids(
            list(suas_data.values()), non_obscured_sat_data
        )

        print("Coincident Counts:")
        print(get_coincident_buildings_per_ortho(valid_ids, suas_data))

        print(
            "\n\nFound",
            len(valid_ids),
            "Buildings with at least one sUAS and Satellite view",
        )

        closest_suas_labels_abs, closest_sat_labels_abs = (
            get_best_temporal_label_for_building(
                list(suas_data.values()),
                non_obscured_sat_data,
                valid_ids,
                multiview_df,
                "abs",
            )
        )
        closest_suas_labels_real, closest_sat_labels_real = (
            get_best_temporal_label_for_building(
                list(suas_data.values()),
                non_obscured_sat_data,
                valid_ids,
                multiview_df,
                "real",
            )
        )
        oracle_suas_labels, oracle_sat_labels = get_best_oracle_label_for_building(
            list(suas_data.values()), non_obscured_sat_data, valid_ids
        )
        antioracle_suas_labels, antioracle_sat_labels = (
            get_best_antioracle_label_for_building(
                list(suas_data.values()), non_obscured_sat_data, valid_ids
            )
        )

        # With the closet post-disaster Satellite orthos to the sUAS orthos...
        # Compute Class Balances between sUAS and Satellite
        suas_class_counts, anti_oracle_class_counts = compute_class_balances(
            antioracle_suas_labels,
            antioracle_sat_labels,
            args.output_stats_folder_path,
            valid_ids,
            "Anti-Oracle",
        )
        suas_class_counts, oracle_class_counts = compute_class_balances(
            oracle_suas_labels,
            oracle_sat_labels,
            args.output_stats_folder_path,
            valid_ids,
            "Oracle",
        )
        suas_class_counts, closest_to_disaster_class_counts = compute_class_balances(
            closest_suas_labels_real,
            closest_sat_labels_real,
            args.output_stats_folder_path,
            valid_ids,
            "Closest to Disaster Temporally",
        )
        suas_class_counts, closest_to_suas_class_counts = compute_class_balances(
            closest_suas_labels_abs,
            closest_sat_labels_abs,
            args.output_stats_folder_path,
            valid_ids,
            "Closest to Drone Temporally",
        )

        suas_class_counts_ignore_obscured, anti_oracle_class_counts_ignore_obscured = (
            compute_class_balances(
                antioracle_suas_labels,
                antioracle_sat_labels,
                args.output_stats_folder_path,
                valid_ids,
                "Anti-Oracle (Ignore Obscured)",
                True,
            )
        )
        suas_class_counts_ignore_obscured, oracle_class_counts_ignore_obscured = (
            compute_class_balances(
                closest_suas_labels_real,
                closest_sat_labels_real,
                args.output_stats_folder_path,
                valid_ids,
                "Closest to Disaster Temporally (Ignore Obscured)",
                True,
            )
        )
        (
            suas_class_counts_ignore_obscured,
            closest_to_disaster_class_counts_ignore_obscured,
        ) = compute_class_balances(
            closest_suas_labels_abs,
            closest_sat_labels_abs,
            args.output_stats_folder_path,
            valid_ids,
            "Closest to Drone Temporally (Ignore Obscured)",
            True,
        )
        (
            suas_class_counts_ignore_obscured,
            closest_to_suas_class_counts_ignore_obscured,
        ) = compute_class_balances(
            oracle_suas_labels,
            oracle_sat_labels,
            args.output_stats_folder_path,
            valid_ids,
            "Oracle (Ignore Obscured)",
            True,
        )

        compute_mulistrategy_class_balances(
            suas_class_counts_ignore_obscured,
            {
                "Satellite Anti-Oracle": anti_oracle_class_counts_ignore_obscured,
                "Satellite Oracle": oracle_class_counts_ignore_obscured,
                "Satellite Closest to Disaster": closest_to_disaster_class_counts_ignore_obscured,
                "Satellite Closest to Drone": closest_to_suas_class_counts_ignore_obscured,
            },
            args.output_stats_folder_path,
            valid_ids,
        )

        print("\n\nAre the sUAS and Satellite distributions different? (Is p < 0.001?)")
        print(
            "sUAS and Closest to Disaster:",
            chi_squared_test(closest_suas_labels_abs, closest_sat_labels_abs, False),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(closest_suas_labels_abs, closest_sat_labels_abs, False),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(closest_suas_labels_abs, closest_sat_labels_abs, False),
        )
        print(
            "sUAS and Closest to sUAS:",
            chi_squared_test(closest_suas_labels_real, closest_sat_labels_real, False),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(closest_suas_labels_real, closest_sat_labels_real, False),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(
                closest_suas_labels_real, closest_sat_labels_real, False
            ),
        )
        print(
            "sUAS and Oracle:",
            chi_squared_test(oracle_suas_labels, oracle_sat_labels, False),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(oracle_suas_labels, oracle_sat_labels, False),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(oracle_suas_labels, oracle_sat_labels, False),
        )
        print(
            "sUAS and Anti-Oracle:",
            chi_squared_test(antioracle_suas_labels, antioracle_sat_labels, False),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(antioracle_suas_labels, antioracle_sat_labels, False),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(antioracle_suas_labels, antioracle_sat_labels, False),
        )

        print(
            "sUAS and Closest to Disaster (Ignore Obscured):",
            chi_squared_test(closest_suas_labels_abs, closest_sat_labels_abs, True),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(closest_suas_labels_abs, closest_sat_labels_abs, True),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(closest_suas_labels_abs, closest_sat_labels_abs, True),
        )
        print(
            "sUAS and Closest to sUAS (Ignore Obscured):",
            chi_squared_test(closest_suas_labels_real, closest_sat_labels_real, True),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(closest_suas_labels_real, closest_sat_labels_real, True),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(closest_suas_labels_real, closest_sat_labels_real, True),
        )
        print(
            "sUAS and Oracle (Ignore Obscured):",
            chi_squared_test(oracle_suas_labels, oracle_sat_labels, True),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(oracle_suas_labels, oracle_sat_labels, True),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(oracle_suas_labels, oracle_sat_labels, True),
        )
        print(
            "sUAS and Anti-Oracle (Ignore Obscured):",
            chi_squared_test(antioracle_suas_labels, antioracle_sat_labels, True),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(antioracle_suas_labels, antioracle_sat_labels, True),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(antioracle_suas_labels, antioracle_sat_labels, True),
        )

        print(
            "\n\nAre the Oracle and Anti-Oracle Distributions Different? (Is p < 0.001?)"
        )
        print(
            "Oracle and Anti-Oracle:",
            chi_squared_test(oracle_sat_labels, antioracle_sat_labels, False),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(oracle_sat_labels, antioracle_sat_labels, False),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(oracle_sat_labels, antioracle_sat_labels, False),
        )
        print(
            "Oracle and Anti-Oracle (Ignore Obscured):",
            chi_squared_test(oracle_sat_labels, antioracle_sat_labels, True),
        )
        print(
            "\tP-Value By Label:    ",
            z_test_per_label(oracle_sat_labels, antioracle_sat_labels, True),
        )
        print(
            "\tEffect Size By Label:",
            compute_odds_ratio(oracle_sat_labels, antioracle_sat_labels, True),
        )

        print(
            "\n\nAre any view selection strategies different signficantly from one another"
        )
        strats = {
            "oracle": oracle_sat_labels,
            "anti_oracle": antioracle_sat_labels,
            "closest to disaster": closest_sat_labels_real,
            "closest to drone": closest_sat_labels_abs,
        }
        for key_1, source_1 in strats.items():
            for key_2, source_2 in strats.items():
                print("\t", key_1, key_2, chi_squared_test(source_1, source_2, True))

        oracle_disagreement_prob = get_probability_of_disagreement(
            oracle_suas_labels, oracle_sat_labels
        )
        antioracle_disagreement_prob = get_probability_of_disagreement(
            antioracle_suas_labels, antioracle_sat_labels
        )
        closest_to_suas_disagreement_prob = get_probability_of_disagreement(
            closest_suas_labels_abs, closest_sat_labels_abs
        )
        closest_to_disaster_disagreement_prob = get_probability_of_disagreement(
            closest_suas_labels_real, closest_sat_labels_real
        )

        print("\n\nProbablity of Disagreement Calculations")
        print(
            "\tclosest_to_disaster_disagreement_prob",
            closest_to_disaster_disagreement_prob,
        )
        print("\tclosest_to_suas_disagreement_prob", closest_to_suas_disagreement_prob)
        print("\toracle_disagreement_prob", oracle_disagreement_prob)
        print("\tantioracle_disagreement_prob", antioracle_disagreement_prob)

        oracle_disagreement_prob_ignore_obscured = get_probability_of_disagreement(
            oracle_suas_labels, oracle_sat_labels, True
        )
        antioracle_disagreement_prob_ignore_obscured = get_probability_of_disagreement(
            antioracle_suas_labels, antioracle_sat_labels, True
        )
        closest_to_suas_disagreement_prob_ignore_obscured = (
            get_probability_of_disagreement(
                closest_suas_labels_abs, closest_sat_labels_abs, True
            )
        )
        closest_to_disaster_disagreement_prob_ignore_obscured = (
            get_probability_of_disagreement(
                closest_suas_labels_real, closest_sat_labels_real, True
            )
        )

        print(
            "\tclosest_to_disaster_disagreement_prob_ignore_obscured",
            closest_to_disaster_disagreement_prob_ignore_obscured,
        )
        print(
            "\tclosest_to_suas_disagreement_prob_ignore_obscured",
            closest_to_suas_disagreement_prob_ignore_obscured,
        )
        print(
            "\toracle_disagreement_prob_ignore_obscured",
            oracle_disagreement_prob_ignore_obscured,
        )
        print(
            "\tantioracle_disagreement_prob_ignore_obscured",
            antioracle_disagreement_prob_ignore_obscured,
        )

        # Compute Transisition Matrix for the change in labels between labels (y-axis -> sUAs label, x-axis -> satellite label)
        print("\n\nGenerating Oracle Transition Matrix...")
        compute_transistion_matrix(
            oracle_suas_labels,
            oracle_sat_labels,
            args.output_stats_folder_path,
            "Satellite Oracle",
            True,
        )
        compute_transistion_matrix(
            antioracle_suas_labels,
            antioracle_sat_labels,
            args.output_stats_folder_path,
            "Satellite Anti-Oracle",
            True,
        )
        compute_transistion_matrix(
            closest_suas_labels_abs,
            closest_sat_labels_abs,
            args.output_stats_folder_path,
            "Satellite Closest to Drone",
            True,
        )
        compute_transistion_matrix(
            closest_suas_labels_real,
            closest_sat_labels_real,
            args.output_stats_folder_path,
            "Satellite Closest to Disaster ",
            True,
        )

        # KL Divergence between the distribution between sUAS and Satellite
        print("\n\nKL Divergence")
        kl_divergence1, kl_divergence2 = compute_kl_divergence(
            suas_class_counts_ignore_obscured, anti_oracle_class_counts_ignore_obscured
        )
        print(
            "\tComputed KL Divergence (suas || anti-oracle satellite): ",
            kl_divergence1,
            "nats",
        )
        print(
            "\tComputed KL Divergence (anti-oracle satellite || suas): ",
            kl_divergence2,
            "nats",
        )
        kl_divergence1, kl_divergence2 = compute_kl_divergence(
            suas_class_counts_ignore_obscured, oracle_class_counts_ignore_obscured
        )
        print(
            "\tComputed KL Divergence (suas || oracle satellite): ",
            kl_divergence1,
            "nats",
        )
        print(
            "\tComputed KL Divergence (oracle satellite || suas): ",
            kl_divergence2,
            "nats",
        )
        kl_divergence1, kl_divergence2 = compute_kl_divergence(
            suas_class_counts_ignore_obscured,
            closest_to_disaster_class_counts_ignore_obscured,
        )
        print(
            "\tComputed KL Divergence (suas || Closest to Disaster satellite): ",
            kl_divergence1,
            "nats",
        )
        print(
            "\tComputed KL Divergence (Closest to Disaster satellite || suas): ",
            kl_divergence2,
            "nats",
        )
        kl_divergence1, kl_divergence2 = compute_kl_divergence(
            suas_class_counts_ignore_obscured,
            closest_to_suas_class_counts_ignore_obscured,
        )
        print(
            "\tComputed KL Divergence (suas || Closest to sUAS satellite): ",
            kl_divergence1,
            "nats",
        )
        print(
            "\tComputed KL Divergence (Closest to sUAS satellite || suas): ",
            kl_divergence2,
            "nats",
        )

        print("\n\nCount of satellite views...")
        view_count, view_count_with_properties = get_satellite_building_counts(
            non_obscured_sat_data, valid_ids
        )
        print("\tTotal count views", view_count)
        print("\tCount of views with meta", view_count_with_properties)

        # With only the sat orthos, compute the change with (time, view)
        suas_days_sat_data = group_buildings_temporally(
            non_obscured_sat_data, multiview_df
        )
        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "days"
        )
        disagree_rate = len(disagree_dist) / (len(disagree_dist) + len(agree_dist))
        view_count = len(disagree_dist) + len(agree_dist)
        print(
            "\n\nProbability of Disagreement between time/view satellite",
            disagree_rate,
            "N=",
            view_count,
            "N_disagree=",
            len(disagree_dist),
            "N_agree=",
            len(agree_dist),
        )

        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "days"
        )
        print("\tConsider date...")
        print(
            "\t\tIs change in date predictive of label changes (P-Value < 0.001)? P-Value =",
            ttest_ind(agree_dist, disagree_dist).pvalue,
        )
        print(
            "\t\t                                                             Effect Size =",
            cohens_d(agree_dist, disagree_dist),
        )
        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "off_nadir"
        )
        print("\tConsider off_nadir...")
        print(
            "\t\tIs change in off_nadir predictive of label changes (P-Value < 0.001)? P-Value =",
            ttest_ind(agree_dist, disagree_dist).pvalue,
        )
        print(
            "\t\t                                                                  Effect Size =",
            cohens_d(agree_dist, disagree_dist),
        )
        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "azimuth"
        )
        print("\tConsider azimuth...")
        print(
            "\t\tIs change in azimuth predictive of label changes (P-Value < 0.001)? P-Value =",
            ttest_ind(agree_dist, disagree_dist).pvalue,
        )
        print(
            "\t\t                                                                Effect Size =",
            cohens_d(agree_dist, disagree_dist),
        )
        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "incidence_angle"
        )
        print("\tConsider incidence_angle...")
        print(
            "\t\tIs change in incidence_angle predictive of label changes (P-Value < 0.001)? P-Value =",
            ttest_ind(agree_dist, disagree_dist).pvalue,
        )
        print(
            "\t\t                                                                        Effect Size =",
            cohens_d(agree_dist, disagree_dist),
        )
        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "sun_azimuth"
        )
        print("\tConsider sun_azimuth...")
        print(
            "\t\tIs change in sun_azimuth predictive of label changes (P-Value < 0.001)? P-Value =",
            ttest_ind(agree_dist, disagree_dist).pvalue,
        )
        print(
            "\t\t                                                                    Effect Size =",
            cohens_d(agree_dist, disagree_dist),
        )
        agree_dist, disagree_dist = compute_paired_difference_views(
            suas_days_sat_data, "sun_elevation"
        )
        print("\tConsider sun_elevation...")
        print(
            "\t\tIs change in sun_elevation predictive of label changes (P-Value < 0.001)? P-Value =",
            ttest_ind(agree_dist, disagree_dist).pvalue,
        )
        print(
            "\t\t                                                                      Effect Size =",
            cohens_d(agree_dist, disagree_dist),
        )

        print("\n\nPlotting Satellite States...")
        plot_sat_views_per_building_histogram(
            non_obscured_sat_data, valid_ids, args.output_stats_folder_path
        )
        plot_sat_view_properites_per_building_histogram(
            non_obscured_sat_data, valid_ids, "off_nadir", args.output_stats_folder_path
        )
        plot_sat_view_properites_per_building_histogram(
            non_obscured_sat_data, valid_ids, "azimuth", args.output_stats_folder_path
        )
        plot_sat_view_properites_per_building_histogram(
            non_obscured_sat_data,
            valid_ids,
            "incidence_angle",
            args.output_stats_folder_path,
        )
        plot_sat_view_properites_per_building_histogram(
            non_obscured_sat_data,
            valid_ids,
            "sun_azimuth",
            args.output_stats_folder_path,
        )
        plot_sat_view_properites_per_building_histogram(
            non_obscured_sat_data,
            valid_ids,
            "sun_elevation",
            args.output_stats_folder_path,
        )
