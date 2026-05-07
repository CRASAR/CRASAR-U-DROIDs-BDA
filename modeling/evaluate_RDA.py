import os
import json
import argparse

import numpy as np
from shapely import LineString

from modeling.Orthomosaic import OrthomosaicFactory
from modeling.Spatial import MultiLabeledRoadLineFactory, LabeledRoadLine, MultiLabeledRoadLine
from modeling.utils.shape_utils import convert_coords_to_shapely
from modeling.utils.hyperparameters import parse_hyperparameters
from modeling.DataMap import Labels2IdxMap


def load_multi_labeled_road_lines_from_preds(preds_data, parent_road_lines):
    results = []
    for parent_road_line_id, labeled_road_lines in preds_data.items():
        objectified_labeled_road_lines = []
        parent_road_line = parent_road_lines[parent_road_line_id]
        for labeled_road_line in labeled_road_lines:
            objectified_labeled_road_lines.append(LabeledRoadLine(label=labeled_road_line["label"],
                                                      confidence=labeled_road_line["confidence"],
                                                      geometry_source=labeled_road_line["geometry_source"],
                                                      pixel_geom=convert_coords_to_shapely(labeled_road_line["pixels"], LineString, lambda x:(x["x"], x["y"])),
                                                      parent_road_line_identifier=labeled_road_line["parent_road_line_id"],
                                                      adjusted=labeled_road_line,
                                                      adjustment_subfield=labeled_road_line["adjustment_subfield"]))

        results.append(MultiLabeledRoadLine(identifier=parent_road_line.getId(),
                                            label=parent_road_line.getLabel(),
                                            geometry_source=parent_road_line.getGeometrySource(),
                                            labeled_road_lines=objectified_labeled_road_lines,
                                            pixel_geom=parent_road_line.getGeometry("pixels"),
                                            epsg_4326_geom=parent_road_line.getGeometry("EPSG:4326"),
                                            adjusted=parent_road_line.isAdjusted(),
                                            adjustment_subfield=parent_road_line.getAdjustmentSubfield()))
    return results

def compute_confusion_matrix_for_road_pair(gt_multilabeled_road_line,
                                           pred_multilabeled_road_line,
                                           confusion_matrix,
                                           dataset_label_to_idx_map,
                                           model_idx_to_label_map,
                                           default_label,
                                           scale,
                                           random_baseline=False,
                                           label_map=None):
    total_intersected_length = 0
    #Look through all the predicted spans we have
    for pred_sub_span in pred_multilabeled_road_line.get_labeled_sub_lines():
        pred_line = pred_sub_span.getGeometry("relative", gt_multilabeled_road_line)
        #And compare them to the ground truth spans...
        if random_baseline:
            predicted_label = np.random.choice(list(confusion_matrix.keys()))
        elif pred_sub_span.getLabel() == label_map.getBackgroundClass()[0]:
            predicted_label = default_label
        else:
            predicted_label = pred_sub_span.getLabel()

        for gt_sub_span in gt_multilabeled_road_line.get_labeled_sub_lines():
            gt_line = gt_sub_span.getGeometry("relative", gt_multilabeled_road_line)

            intersected_length = pred_line.intersection(gt_line).length*scale
            total_intersected_length += intersected_length

            confusion_matrix_gt_label = model_idx_to_label_map.getLabels(dataset_label_to_idx_map.getIndex(gt_sub_span.getLabel()))[0]
            confusion_matrix[confusion_matrix_gt_label][predicted_label] += intersected_length

        #If there was any section of the line left over, that means that it didn't intersect with any annotation polygons
        #This means that the spans belonged to road lines. So we need to annotate them as such.
        if total_intersected_length < pred_line.length*scale:
            remaining_annotation_length = (pred_line.length*scale) - total_intersected_length
            #If we actually predicted the background class for some reason, then make it the default label
            confusion_matrix[default_label][predicted_label] += remaining_annotation_length


    return confusion_matrix

def compute_instance_metrics(confusion_matrix, key):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for ground_truth_label in confusion_matrix.keys():
        for predicted_label in confusion_matrix[ground_truth_label].keys():
            if ground_truth_label == key and predicted_label == key:
                tp += confusion_matrix[ground_truth_label][predicted_label]
            if ground_truth_label == key and predicted_label != key:
                fn += confusion_matrix[ground_truth_label][predicted_label]
            if ground_truth_label != key and predicted_label == key:
                fp += confusion_matrix[ground_truth_label][predicted_label]
            # pylint: disable-next=consider-using-in
            if ground_truth_label != key and predicted_label != key:
                tn += confusion_matrix[ground_truth_label][predicted_label]
    recall_denom = 1 if (tp+tn) == 0 else (tp+tn)
    recall = tp/recall_denom
    precision_denom = 1 if (tp+fp) == 0 else (tp+fp)
    precision = tp/precision_denom
    f1_denom = 1 if (2*tp+fp+fn) == 0 else (2*tp+fp+fn)
    f1 = (2*tp)/f1_denom
    accuracy_denom = 1 if (tn+tp+fn+fp) == 0 else (tn+tp+fn+fp)
    accuracy = (tn+tp)/accuracy_denom
    iou_denom = 1 if (tp+fn+fp) == 0 else (tp+fn+fp)
    iou = tp/iou_denom

    return {"recall":recall, "precision":precision, "f1": f1, "accuracy":accuracy, "iou":iou}

def get_metrics_bundle(pred_multilabeled_road_lines,
                       gt_multilabeled_road_lines,
                       default_label,
                       dataset_label_map,
                       output_label_map,
                       road_lines_to_gsd,
                       model_name,
                       is_random=False):
    #Get the labels that are going to be used...
    confusion_matrix_pixels = {}
    confusion_matrix_km = {}

    #Get the labels for the confusion matrix
    conf_matrix_labels = list(output_label_map.getAllLabels())
    conf_matrix_labels.remove(output_label_map.getBackgroundClass()[0])

    #TODO: Make debug orthos based on the multilabel road lines.
    total_km_gsd_calculated = 0
    total_segments = 0
    for line in pred_multilabeled_road_lines:
        total_km_gsd_calculated += ((road_lines_to_gsd[line.getId()]/100)/1000) * line.getGeometry("pixels").length
        total_segments += len(line.get_labeled_sub_lines())

    for gt_label in conf_matrix_labels:
        confusion_matrix_pixels[gt_label] = {}
        confusion_matrix_km[gt_label] = {}
        for pred_label in conf_matrix_labels:
            confusion_matrix_pixels[gt_label][pred_label] = 0
            confusion_matrix_km[gt_label][pred_label] = 0

    #For every road we have...
    for line in pred_multilabeled_road_lines:
        confusion_matrix_pixels = compute_confusion_matrix_for_road_pair(gt_multilabeled_road_lines[line.getId()],
                                                                         line,
                                                                         confusion_matrix_pixels,
                                                                         dataset_label_map,
                                                                         output_label_map,
                                                                         default_label=default_label,
                                                                         scale=line.getGeometry("pixels").length,
                                                                         random_baseline=is_random,
                                                                         label_map=output_label_map)
        confusion_matrix_km = compute_confusion_matrix_for_road_pair(gt_multilabeled_road_lines[line.getId()],
                                                                     line,
                                                                     confusion_matrix_km,
                                                                     dataset_label_map,
                                                                     output_label_map,
                                                                     default_label=default_label,
                                                                     scale=((road_lines_to_gsd[line.getId()]/100)/1000) * line.getGeometry("pixels").length,
                                                                     random_baseline=is_random,
                                                                     label_map=output_label_map)

    confusion_matrix_pixels_list = []
    confusion_matrix_km_list = []
    class_counts_km = {}
    for gt_label in conf_matrix_labels:
        confusion_matrix_pixels_list.append([])
        confusion_matrix_km_list.append([])
        class_counts_km[gt_label] = 0
        for pred_label in conf_matrix_labels:
            confusion_matrix_pixels_list[-1].append(confusion_matrix_pixels[gt_label][pred_label])
            confusion_matrix_km_list[-1].append(confusion_matrix_km[gt_label][pred_label])
            class_counts_km[gt_label] += confusion_matrix_km[gt_label][pred_label]

    rda_f1 = {}
    rda_accuracy = {}
    rda_precision = {}
    rda_recall = {}
    rda_iou = {}
    for label in conf_matrix_labels:
        d = compute_instance_metrics(confusion_matrix_km, label)
        rda_f1[label] = d["f1"]
        rda_accuracy[label] = d["accuracy"]
        rda_precision[label] = d["precision"]
        rda_recall[label] = d["recall"]
        rda_iou[label] = d["iou"]

    metrics = {
        "model_name":model_name,
        "samples": {
            "total":total_km_gsd_calculated,
            "class_level": class_counts_km,
            "total_predicted_segments":total_segments
            },
        "metrics": {
            "F1": {
                "class_level":rda_f1,
                "macro":sum(rda_f1.values())/len(conf_matrix_labels),
                },
            "Accuracy": {
                "class_level":rda_accuracy
                },
            "Precision": {
                "class_level":rda_precision,
                "macro":sum(rda_precision.values())/len(conf_matrix_labels),
                },
            "Recall": {
                "class_level":rda_recall,
                "macro":sum(rda_recall.values())/len(conf_matrix_labels),
                },
            "IoU": {
                "class_level":rda_iou,
                "macro":sum(rda_iou.values())/len(conf_matrix_labels),
                },
            "Confusion_Matrix_pixels" : {
                "matrix": confusion_matrix_pixels_list,
                "class_labels": conf_matrix_labels
                },
            "Confusion_Matrix_km" : {
                "matrix": confusion_matrix_km_list,
                "class_labels": conf_matrix_labels
                }
            }
        }
    return metrics

def get_ground_truth_multilabeled_road_lines(orthomosaics, adjusted):
    result = {}
    for gt_orthomosaic in orthomosaics:
        road_lines = gt_orthomosaic.get_road_lines(adjusted=adjusted)
        annotation_polygons = gt_orthomosaic.get_road_line_annotation_polygons()
        road_lines_with_multiple_labels = MultiLabeledRoadLineFactory(road_lines, annotation_polygons)
        for road_line_with_multiple_labels in road_lines_with_multiple_labels:
            result[road_line_with_multiple_labels.getId()] = road_line_with_multiple_labels
    return result

def get_road_line_to_gsd_map(orthomosaics, adjusted):
    result = {}
    for gt_orthomosaic in orthomosaics:
        road_lines = gt_orthomosaic.get_road_lines(adjusted=adjusted)
        annotation_polygons = gt_orthomosaic.get_road_line_annotation_polygons()
        road_lines_with_multiple_labels = MultiLabeledRoadLineFactory(road_lines, annotation_polygons)
        for road_line_with_multiple_labels in road_lines_with_multiple_labels:
            result[road_line_with_multiple_labels.getId()] = gt_orthomosaic.get_gsd()[0]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model.')
    parser.add_argument('--road_lines_folder', type=str, help='Path to buildings labels folder')
    parser.add_argument('--road_adjustments_folder', type=str, help='Path to buildings labels folder')
    parser.add_argument('--preds_path', type=str, help='The path to file that contains the model predicitons.')
    parser.add_argument('--metrics_file', type=str, help='The path to the file where the metrics will be stored.')
    parser.add_argument('--hyperparameters_file', type=str, help='The path to the file where the hyperparameter are stored.')
    parser.add_argument('--ortho_stats_file', type=str, help="The path to the statistics.csv file included with the dataset.")
    parser.add_argument('--random_baseline', action="store_true",
                                             help="When this flag is set, the input predictions will be overridden and the random baseline will be used.")
    args = parser.parse_args()

    metrics_file = os.path.dirname(args.metrics_file)
    if not os.path.exists(metrics_file):
        os.makedirs(metrics_file, exist_ok=True)
        print("Created the directory to store the output: " + str(metrics_file))

    #Initialize the model
    print("Reading hyperparameters...")
    hyperparameters = parse_hyperparameters(args.hyperparameters_file)

    parsed_default_label = hyperparameters["channel_maps"]["default_label"]
    parsed_output_label_map = Labels2IdxMap(hyperparameters["channel_maps"]["output_class_2_idx_map"],
                                            hyperparameters["channel_maps"]["background_class_idx"])
    parsed_dataset_label_map = Labels2IdxMap(hyperparameters["channel_maps"]["input_dataset_class_2_idx_map"],
                                             hyperparameters["channel_maps"]["background_class_idx"])

    #Read the data from all the different polygons in the folder containing the labels
    print("Loading Orthomosaics...")
    gt_orthomosaics = OrthomosaicFactory(rda_annotation_folder=args.road_lines_folder,
                                         rda_adj_annotation_folder=args.road_adjustments_folder,
                                         statistics_file_path=args.ortho_stats_file)
    print("Loaded", len(gt_orthomosaics), "orthomosaics...")
    print("Done")

    #Read the data from the predictions file
    print("Loading Predictions...")
    with open(args.preds_path, "r") as f:
        preds = json.loads(f.read())
    print("Done")

    #Load all of the ground truth multilabeled roadlines
    gt_mlrs = get_ground_truth_multilabeled_road_lines(gt_orthomosaics, not args.road_adjustments_folder is None)
    local_road_lines_to_gsd = get_road_line_to_gsd_map(gt_orthomosaics, not args.road_adjustments_folder is None)

    #Convert the predictions into multilabeled roadlines
    pred_mlrs = load_multi_labeled_road_lines_from_preds(preds["preds"], gt_mlrs)

    #Compute all the metrics that we are about and store them in a dict
    metrics_bundle = get_metrics_bundle(pred_mlrs,
                                        gt_mlrs,
                                        parsed_default_label,
                                        parsed_dataset_label_map,
                                        parsed_output_label_map,
                                        local_road_lines_to_gsd,
                                        preds["model_name"],
                                        args.random_baseline)

    #Convert the dict to json
    metrics_json = json.dumps(metrics_bundle)

    #Write the metrics to a file
    with open(args.metrics_file, "w") as f:
        f.write(metrics_json)
    print("Done.")
