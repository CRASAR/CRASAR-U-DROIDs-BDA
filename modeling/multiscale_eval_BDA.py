import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset.constants import (
    NO_DAMAGE,
    MINOR_DAMAGE,
    MAJOR_DAMAGE,
    DESTROYED,
    UNCLASSIFIED,
    BDA_DAMAGE_CLASSES,
)

CLASS_COLORS = {
    NO_DAMAGE: "green",
    MINOR_DAMAGE: "yellow",
    MAJOR_DAMAGE: "orange",
    DESTROYED: "red",
    UNCLASSIFIED: "purple",
}


# pylint: disable=unsubscriptable-object,redefined-outer-name,redefined-builtin
def multiscale_plot(metrics_files, out_folder):
    f1_data = []
    accuracy_data = []
    precision_data = []
    recall_data = []
    auc_roc_data = []
    x_values = []

    for file in metrics_files:
        print("Reading metrics from: ", file)
        file_name = os.path.split(file)[1]
        x_values.append(float(file_name.split("_", 1)[0]))
        with open(file, "r") as f:
            data = json.load(f)

        f1_scores = data["metrics"]["F1"]["class_level"]
        accuracy_scores = data["metrics"]["Accuracy"]["class_level"]
        precision_scores = data["metrics"]["Precision"]["class_level"]
        recall_scores = data["metrics"]["Recall"]["class_level"]
        auc_roc_scores = data["metrics"]["AUC_ROC"]["class_level"]

        f1_data.append(f1_scores)
        accuracy_data.append(accuracy_scores)
        precision_data.append(precision_scores)
        recall_data.append(recall_scores)
        auc_roc_data.append(auc_roc_scores)

    f1_df = pd.DataFrame(f1_data, index=x_values)
    accuracy_df = pd.DataFrame(accuracy_data, index=x_values)
    precision_df = pd.DataFrame(precision_data, index=x_values)
    recall_df = pd.DataFrame(recall_data, index=x_values)
    auc_roc_df = pd.DataFrame(auc_roc_data, index=x_values)

    f1_df = f1_df.sort_index()
    accuracy_df = accuracy_df.sort_index()
    precision_df = precision_df.sort_index()
    recall_df = recall_df.sort_index()
    auc_roc_df = auc_roc_df.sort_index()

    metric_names = ["F1", "Accuracy", "AUC_ROC", "Precision", "Recall"]
    metric_dfs = [f1_df, accuracy_df, auc_roc_df, precision_df, recall_df]

    for metric, metric_df in zip(metric_names, metric_dfs):
        plt.figure(figsize=(10, 5))
        for class_name in metric_df.columns:
            color = CLASS_COLORS[class_name]
            plt.plot(
                metric_df.index,
                metric_df[class_name],
                label=f"{class_name}",
                color=color,
            )
        plt.xlabel("Downscale factor")
        plt.ylabel(metric)
        plt.legend()
        plt.title(f"{metric} Scores Across Scales")
        plt.savefig(os.path.join(out_folder, f"{metric.lower()}_scores.png"))

    print("Done.")


def gsd_plot(file, out_folder):
    print("Plotting GSD Plots for all predictions...")
    gsd_data = {}
    gsd_ranges = []

    print("Reading metrics from:", file)
    with open(file, "r") as f:
        data = json.load(f)

    for gsd_range, gsd_metrics in data["gsd_metrics"].items():
        if gsd_range not in gsd_ranges:
            gsd_ranges.append(gsd_range)
            gsd_data[gsd_range] = {
                "F1": {},
                "Accuracy": {},
                "Precision": {},
                "Recall": {},
                "AUC_ROC": {},
            }
        for metric_name in ["F1", "Accuracy", "Precision", "Recall", "AUC_ROC"]:
            if metric_name in gsd_metrics:
                if isinstance(gsd_metrics[metric_name].get("class_level", {}), dict):
                    for cls, value in gsd_metrics[metric_name]["class_level"].items():
                        gsd_data[gsd_range][metric_name].setdefault(cls, []).append(
                            value
                        )
                else:
                    for cls in BDA_DAMAGE_CLASSES:
                        gsd_data[gsd_range][metric_name].setdefault(cls, []).append(0)

    gsd_ranges = sorted(gsd_ranges, key=lambda x: float(x.split("-")[0]))
    gsd_x_labels = [f"{gsd}" for gsd in gsd_ranges]

    for metric_name in ["F1", "Accuracy", "Precision", "Recall", "AUC_ROC"]:
        plt.figure(figsize=(12, 6))
        for cls in sorted(BDA_DAMAGE_CLASSES):
            metric_values = [
                np.mean(gsd_data[gsd][metric_name].get(cls, [0])) for gsd in gsd_ranges
            ]
            color = CLASS_COLORS[cls]
            plt.plot(gsd_x_labels, metric_values, label=f"{cls}", color=color)

        plt.xlabel("GSD Range (m)")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} Across GSD Ranges (m/px) (Per Class)")
        plt.xticks(rotation=45)
        plt.legend(title="Classes", loc="best")
        plt.tight_layout()

        save_path = os.path.join(out_folder, f"{metric_name.lower()}_gsd_scores.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot: {save_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the line graphs accross the multiple metrics files."
    )
    parser.add_argument("--metrics_folder", type=str, help="Path to the metrics folder")
    parser.add_argument("--out_path", type=str, help="The output path for the plots.")
    parser.add_argument(
        "--gsd_path", type=str, help="Path to all gsd eval.", default=None
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
        print("Created the directory to store the output: " + str(args.out_path))

    if not os.path.exists(args.gsd_path):
        os.makedirs(args.gsd_path, exist_ok=True)
        print("Created the directory to store the output: " + str(args.gsd_path))

    metrics_files = []
    for root, dir, files in os.walk(args.metrics_folder):
        for file in files:
            metrics_files.append(os.path.join(root, file))

    multiscale_plot(metrics_files, args.out_path)

    if args.gsd_path is not None:
        gsd_plot(args.gsd_path, args.out_path)
