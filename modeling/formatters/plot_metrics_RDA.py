# Modified from https://github.tamu.edu/hrail/RTAPS
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

current = os.path.dirname(os.path.realpath(__file__))
modeling_path = os.path.dirname(current)
dataset_path = os.path.dirname(modeling_path)
sys.path.append(modeling_path)
sys.path.append(dataset_path)

from modeling.constants import RDA_METRICS, PLOT_BAR_OFFSET_FACTOR

def generate_confusion_matrix_plot(model_metrics_data, model_names, round_digits=3, plot_folder=None, return_np=False, key="Confusion_Matrix"):
    if(plot_folder is None and return_np == False):
        raise Exception("Must define either a folder path where the confusion matrix will be written, or return a numpy array.")
    for k, data in enumerate(model_metrics_data):
        if(key in data["metrics"]):
            conf_matrix = np.around(np.array(data["metrics"][key]["matrix"]), round_digits)
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
            plt.xticks(range(conf_matrix.shape[1]), data["metrics"][key]["class_labels"])
            plt.yticks(range(conf_matrix.shape[0]), data["metrics"][key]["class_labels"])
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
            
            plt.xlabel('Predictions')
            plt.ylabel('Actuals')
            plt.title(str(model_names[k]) + ' Confusion Matrix | N='+str(data["samples"]["total"]), fontsize=18)
            
            # Save plot
            if(plot_folder):
                path = os.path.join(plot_folder, key + "_" + str(model_names[k]) + ".png")
                print("Saving Plot: " + str(path))
                plt.savefig(path)
                plt.clf()
            if(return_np):
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                return data

def generate_class_level_plots(model_metrics_data, model_names, plots_folder, hide_numeric, round=2):
    model_count = len(model_metrics_data)
    models_str = "_".join([m for m in model_names])

    # Generate a plot for each class level metric
    for metric in RDA_METRICS:
        fig = plt.figure(figsize=(8.5,8.5))

        xlabels = []
        max_xs = 0
        max_y = -10e99
        min_y = 10e99

        # For every model we want to include in this plot
        for i, data in enumerate(model_metrics_data):
            model = model_names[i]
            class_counts = data["samples"]["class_level"]
            xlabels = [label + "\nn=" + str(str(np.around(class_counts[label], round))) for label in data["metrics"][metric]["class_level"].keys()]
            xs = np.arange(len(xlabels))
            ys = list(data["metrics"][metric]["class_level"].values())

            # Calcuate the positional offsets for each of the bars
            width = (1-PLOT_BAR_OFFSET_FACTOR)/model_count
            offset = (i*width) - width*(model_count/2) + width/2

            max_xs = max(max_xs, len(xs))
            max_y = max(max_y, max(ys))
            min_y = min(min_y, min(ys))

            # Plot the data
            plt.bar(xs+offset, ys, width, label=model)

        # For every model we want to include with this plot (but we are doing this again having computed the maxes and mins)
        if(not hide_numeric):
            for i, data in enumerate(model_metrics_data):


                ys = list(data["metrics"][metric]["class_level"].values())
                labels = list(data["metrics"][metric]["class_level"].keys())

                # Calcuate the positional offsets for each of the bars
                width = (1-PLOT_BAR_OFFSET_FACTOR)/model_count
                offset = (i*width) - width*(model_count/2) + width/2
                
                # Plot the numeric data
                for x, y, label in zip(xs, ys, labels):
                    t = str(np.around(y, round))
                    if("." in t):
                        a, b = t.split(".")
                        t = a + "." + b[:round]
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

def generate_metric_plots(model_metrics_data, model_names, plots_folder, macro_or_micro, hide_numeric, round=2):
    model_count = len(model_metrics_data)
    models_str = "_".join([m for m in model_names])

    width = 0.8

    # Generate a plot for each class level metric
    for metric in RDA_METRICS:
        fig = plt.figure(figsize=(8.5,8.5))

        valid = False

        # For every model we want to include in this plot
        for i, data in enumerate(model_metrics_data):

            if(macro_or_micro in data["metrics"][metric].keys()):
                model = model_names[i]
                xs = np.array([i])
                ys = [data["metrics"][metric][macro_or_micro]]

                # Plot the data
                plt.bar(xs, ys, width, label="Model #" + str(i+1) + " | " + model)
                valid = True

        # For every model we want to include with this plot (but we are doing this again having computed the maxes and mins)
        if(not hide_numeric):
            for i, data in enumerate(model_metrics_data):
                if(macro_or_micro in data["metrics"][metric].keys()):
                    y = data["metrics"][metric][macro_or_micro]
                    x = i
                
                    t = str(np.around(y, round))
                    if("." in t):
                        a, b = t.split(".")
                        t =  a + "." + b[:round]
                    plt.text(x, y + 1.2*0.015, t, horizontalalignment="center", size="x-small")

        if(valid):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the metrics associated with the passed metrics files')
    parser.add_argument('--metrics_files', metavar='N', type=str, nargs='+', help='The paths to the metrics files that you want to plot')
    parser.add_argument('--plots_folder', type=str, help='The path to where the plots should be saved.')
    parser.add_argument('--round', type=int, help="The number of digits to round to in the plot.", default=3)
    parser.add_argument('--hide_numeric', action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.plots_folder):
        os.makedirs(args.plots_folder, exist_ok=True)
        print("Created the directory to store the plots: " + str(args.plots_folder))

    # Get metric data
    model_metrics_data = []
    model_names = []
    for file in args.metrics_files:
        f = open(file, "r")
        data = f.read()
        f.close()
        model_metrics_data.append(json.loads(data))
        model_names.append(model_metrics_data[-1]["model_name"])

    generate_metric_plots(model_metrics_data, model_names, args.plots_folder, "macro", args.hide_numeric, args.round)
    generate_class_level_plots(model_metrics_data, model_names, args.plots_folder, args.hide_numeric, args.round)
    generate_confusion_matrix_plot(model_metrics_data, model_names, args.round, args.plots_folder, key="Confusion_Matrix_pixels")
    generate_confusion_matrix_plot(model_metrics_data, model_names, args.round, args.plots_folder, key="Confusion_Matrix_km")

    print("Done.")