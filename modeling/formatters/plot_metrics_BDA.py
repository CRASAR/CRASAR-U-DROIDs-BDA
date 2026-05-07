import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from modeling.constants import BDA_METRICS, PLOT_BAR_OFFSET_FACTOR

def get_keys_per_stratification(model_metrics_data, stratification):
    return list(model_metrics_data[stratification].keys())
def get_stratifications(model_metrics_data):
    stratifications = list(model_metrics_data.keys())
    stratifications.remove("model_name")
    return stratifications

def get_model_names_bundles_and_stratification_keys(model_names, model_metrics_data, stratification):
    tmp_bundle = []
    tmp_model_names = []
    tmp_stratification_keys = []
    for d, name in zip(model_metrics_data, model_names):
        if stratification in d:
            tmp_stratification_keys.extend(list(d[stratification].keys()))
            tmp_bundle.append(d[stratification])
            tmp_model_names.append(name)
    return tmp_model_names, tmp_bundle, tmp_stratification_keys

def generate_confusion_matrix_plot(model_metrics_data, model_names, plot_folder=None, return_np=False):
    if plot_folder is None and not return_np:
        raise ValueError("Must define either a folder path where the confusion matrix will be written, or return a numpy array.")
    for k, data in enumerate(model_metrics_data):
        if "Confusion_Matrix" in data["metrics"]:
            conf_matrix = np.array(data["metrics"]["Confusion_Matrix"]["matrix"])
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.matshow(conf_matrix, cmap=plt.colormaps.get_cmap("Blues"), alpha=0.3)
            plt.xticks(range(conf_matrix.shape[1]), data["metrics"]["Confusion_Matrix"]["class_labels"])
            plt.yticks(range(conf_matrix.shape[0]), data["metrics"]["Confusion_Matrix"]["class_labels"])
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            ax = plt.gca()
            ax.yaxis.set_label_position("right")
            current_step = data["step"] if "step" in data else None
            plt.title(str(model_names[k]) + ' Confusion Matrix | N='+str(data["samples"]["total"]) + \
                ("" if current_step is None else " | Step=" + str(current_step)), fontsize=18)

            # Save plot
            if plot_folder:
                path = os.path.join(plot_folder, "Confusion_Matrix_" + str(model_names[k]) + ".png")
                print("Saving Plot: " + str(path))
                plt.savefig(path)
                plt.clf()
                plt.close('all')
            if return_np:
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close('all')
                return data
    return None

def generate_class_level_plots(model_metrics_data, model_names, plots_folder, hide_numeric, rounding=2):
    model_count = len(model_metrics_data)
    models_str = "_".join(model_names)

    # Generate a plot for each class level metric
    for metric in BDA_METRICS:
        plt.figure(figsize=(8.5,8.5))

        xlabels = []
        max_xs = 0
        max_y = -10e99
        min_y = 10e99

        # For every model we want to include in this plot
        for i, data in enumerate(model_metrics_data):
            model = model_names[i]
            class_counts = data["samples"]["actual_class_counts"]
            xlabels = [label + "\nn=" + str(class_counts[label]) for label in data["metrics"][metric]["class_level"].keys()]
            xs = np.arange(len(xlabels))
            ys = list(data["metrics"][metric]["class_level"].values())

            # Calcuate the positional offsets for each of the bars
            width = (1-PLOT_BAR_OFFSET_FACTOR)/model_count
            offset = (i*width) - width*(model_count/2) + width/2

            max_xs = max(max_xs, len(xs))
            max_y = max(max_y, *ys)
            min_y = min(min_y, *ys)

            # Plot the data
            plt.bar(xs+offset, ys, width, label=model)

        # For every model we want to include with this plot (but we are doing this again having computed the maxes and mins)
        if not hide_numeric:
            for i, data in enumerate(model_metrics_data):


                ys = list(data["metrics"][metric]["class_level"].values())
                labels = list(data["metrics"][metric]["class_level"].keys())

                # Calcuate the positional offsets for each of the bars
                width = (1-PLOT_BAR_OFFSET_FACTOR)/model_count
                offset = (i*width) - width*(model_count/2) + width/2

                # Plot the numeric data
                for x, y, label in zip(xs, ys, labels):
                    t = str(np.around(y, rounding))
                    if "." in t:
                        a, b = t.split(".")
                        t = a + "." + b[:rounding]
                    plt.text(x+offset , y + 1.2*0.015, t, horizontalalignment="center", size="x-small")

        plt.xticks(np.arange(max_xs), xlabels, rotation=45)
        plt.xlabel("BDA Labels")
        plt.title(metric + " For Each Damage Class | N="+str(data["samples"]["total"]))

        plt.ylabel(metric)
        plt.legend()
        plt.grid(alpha=0.2, axis='y')
        plt.ylim((0.0, 1.2))

        path = os.path.join(plots_folder, str(metric) + "_" + models_str + ".png")
        print("Saving Plot: " + str(path))

        # Save plot
        plt.savefig(path)
        plt.clf()
        plt.close('all')

def generate_metric_plots(model_metrics_data, model_names, plots_folder, macro_or_micro, hide_numeric, rounding=2):
    models_str = "_".join(model_names)

    width = 0.8

    # Generate a plot for each class level metric
    for metric in BDA_METRICS:
        plt.figure(figsize=(8.5,8.5))

        valid = False

        # For every model we want to include in this plot
        for i, data in enumerate(model_metrics_data):

            if macro_or_micro in data["metrics"][metric].keys():
                model = model_names[i]
                xs = np.array([i])
                ys = [data["metrics"][metric][macro_or_micro]]

                # Plot the data
                plt.bar(xs, ys, width, label="Model #" + str(i+1) + " | " + model)
                valid = True

        # For every model we want to include with this plot (but we are doing this again having computed the maxes and mins)
        if not hide_numeric:
            for i, data in enumerate(model_metrics_data):
                if macro_or_micro in data["metrics"][metric].keys():
                    y = data["metrics"][metric][macro_or_micro]
                    x = i

                    t = str(np.around(y, rounding))
                    if "." in t:
                        a, b = t.split(".")
                        t =  a + "." + b[:rounding]
                    plt.text(x, y + 1.2*0.015, t, horizontalalignment="center", size="x-small")

        if valid:
            plt.xticks(np.arange(len(model_names)), ["Model #" + str(i+1) for i in range(0, len(model_names))])
            plt.xlabel("")
            plt.title(macro_or_micro + " " + metric + " For Each Damage Class | N="+str(data["samples"]["total"]))

            plt.ylabel(metric)
            plt.legend()
            plt.grid(alpha=0.2, axis='y')
            plt.ylim((0, 1.2))
            plt.xlim((-width*1.1, len(model_names)-1+width*1.1))

            path = os.path.join(plots_folder, str(metric) + "_" + macro_or_micro + "_" + models_str + ".png")
            print("Saving Plot: " + str(path))

            # Save plot
            plt.savefig(path)
            plt.clf()
            plt.close('all')

def generate_plot_files(model_metrics_data, model_names, plots_folder, hide_numeric, round_sigfigs):
    generate_metric_plots(model_metrics_data, model_names, plots_folder, "micro", hide_numeric, round_sigfigs)
    generate_metric_plots(model_metrics_data, model_names, plots_folder, "macro", hide_numeric, round_sigfigs)
    generate_class_level_plots(model_metrics_data, model_names, plots_folder, hide_numeric, round_sigfigs)
    generate_confusion_matrix_plot(model_metrics_data, model_names, plots_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the metrics associated with the passed metrics files')
    parser.add_argument('--metrics_files', metavar='N', type=str, nargs='+', help='The paths to the metrics files that you want to plot')
    parser.add_argument('--plots_folder', type=str, help='The path to where the plots should be saved.')
    parser.add_argument('--round', type=int, help="The number of digits to round to in the plot.", default=2)
    parser.add_argument('--hide_numeric', action="store_true")
    args = parser.parse_args()

    # Create the folder where these plots will land
    os.makedirs(args.plots_folder, exist_ok=True)

    # Get metric data
    passed_model_metrics_data = []
    passed_model_names = []
    metric_grouping_keys = []
    for file in args.metrics_files:
        with open(file, "r") as f:
            read_metrics_data = f.read()
        passed_model_metrics_data.append(json.loads(read_metrics_data))
        passed_model_names.append(" ".join(passed_model_metrics_data[-1]["model_name"]))
        metric_grouping_keys.extend(get_stratifications(json.loads(read_metrics_data)))

    metric_grouping_keys = list(set(metric_grouping_keys))

    for key in metric_grouping_keys:
        names, bundles, stratification_keys = get_model_names_bundles_and_stratification_keys(passed_model_names, passed_model_metrics_data, key)

        if key == "global_metrics":
            metrics_subfolder_path = os.path.join(args.plots_folder, key)
            os.makedirs(metrics_subfolder_path, exist_ok=True)
            generate_plot_files(bundles, names, metrics_subfolder_path, args.hide_numeric, args.round)
        else:
            metrics_subfolder_path = os.path.join(args.plots_folder, key)
            os.makedirs(metrics_subfolder_path, exist_ok=True)
            for stratification_key in stratification_keys:
                if stratification_key != "NaN":
                    stratification_metrics_subfolder_path = os.path.join(args.plots_folder, key, stratification_key)
                    os.makedirs(stratification_metrics_subfolder_path, exist_ok=True)
                    sub_names, stratification_bundles, _ = get_model_names_bundles_and_stratification_keys(names, bundles, stratification_key)
                    generate_plot_files(stratification_bundles, sub_names, stratification_metrics_subfolder_path, args.hide_numeric, args.round)

    print("Done.")
