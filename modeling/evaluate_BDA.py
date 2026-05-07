import os
import copy
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils import resample

from modeling.utils.hyperparameters import parse_hyperparameters, add_hyperparameters_files_to_parse_args
from modeling.Models.OrthoInferenceWrapper import joint_file_pred_key, parse_pred_key

def parse_reportable_metrics(metrics, metric_paths_str):
    result = []
    for path in metric_paths_str.split(","):
        result.append(parse_reportable_metric(metrics, path))
    return result

def parse_reportable_metric(metrics, metric_path):
    metric_tmp = copy.deepcopy(metrics)
    result = {}
    metric_path_aug = copy.deepcopy(metric_path)
    metric_keys = metric_path.split(">")
    for i, key in enumerate(metric_keys):
        if key == "*":
            sub_result = {}
            for matched_key in metric_tmp.keys():
                sub_keys =">".join(metric_keys[i:]).replace("*", matched_key, 1)
                sub_result = sub_result | parse_reportable_metric(metric_tmp, sub_keys)
            metric_tmp = sub_result
            metric_path_aug = metric_path.split("*")[0]
            break
        else:
            metric_tmp = metric_tmp[key]
    result[metric_path_aug] = metric_tmp
    return result

def parse_actual_labels(buildings_labels_folder, channel_map):
    output_channel_2_label = {idx:label for label, idx in channel_map["channel_maps"]["output_class_2_idx_map"].items()}
    input_label_2_output_channel = channel_map["channel_maps"]["input_dataset_class_2_idx_map"]

    # Read the data from all the different polygons in the folder containing the labels
    actuals = {}
    for root, _, files in os.walk(buildings_labels_folder):
        for file in files:
            ortho_filename = file.replace(".json", "")
            with open(os.path.join(root, file), "r") as f:
                labeled_polygons = json.load(f)
            for labeled_polygon in labeled_polygons:
                actual_label = output_channel_2_label[input_label_2_output_channel[labeled_polygon["label"]]]
                actuals[joint_file_pred_key(ortho_filename, labeled_polygon["view_id"])] = actual_label
    return actuals

def parse_model_name_and_predicted_labels(preds_path):
    with open(preds_path, "r") as f:
        preds_data = json.load(f)
        return preds_data["model_name"], preds_data["preds"]

def get_label_indicator_dict(label, labels):
    result = {l:0 for l in labels}
    result[label] = 1.0
    return result

class AlignedPredsActuals:
    def __init__(self, unaligned_preds, unaligned_actuals, orthomosaic_stats, channel_map, verbose=True):
        # Store the passed objects
        self._unaligned_preds = unaligned_preds
        self._unaligned_actuals = unaligned_actuals
        self._orthomosaic_stats = orthomosaic_stats
        self._channel_map = channel_map
        self._verbose = verbose
        self._deterministic_ordering, rejected_preds_count = self._get_aligned_pred_ids_on_field_value(None, None)
        if self._verbose and rejected_preds_count > 0:
            print("WARNING: Failed to match", rejected_preds_count, "predictions with actual labels. This could be due to running a partial evaluation.")

    def get_labels(self):
        result = list(self._channel_map["channel_maps"]["output_class_2_idx_map"].keys())
        result.remove("background")
        return result
        
    def get_preds_multilabel_dict(self):
        return self._get_fixed_ordered_data(self._unaligned_preds, lambda x:get_label_indicator_dict(x["label"], self.get_labels()))
        
    def get_actuals_multilabel_dict(self):
        return self._get_fixed_ordered_data(self._unaligned_actuals, lambda x:get_label_indicator_dict(x, self.get_labels()))
        
    def get_preds_labels(self):
        return self._get_fixed_ordered_data(self._unaligned_preds, lambda x:x["label"])
        
    def get_actuals_labels(self):
        return self._get_fixed_ordered_data(self._unaligned_actuals, lambda x:x)
        
    def _get_fixed_ordered_data(self, data, func):
        result = []
        for pred_id in self._deterministic_ordering:
            valid = False
            try:
                result.append(func(data[pred_id]))
                valid = True
            except KeyError:
                pass
            try:
                if not valid:
                    ortho_name, building_id, _, _ = parse_pred_key(pred_id)
                    general_key = joint_file_pred_key(ortho_name, building_id)
                    result.append(func(data[general_key]))
                    valid = True
            except KeyError:
                pass
            if self._verbose and not valid:
                print("WARNING: Found prediction for a building that does not appear in actuals.")
        return result
        
    def _get_subset_dict(self, data, pred_ids, gsd_pred_ids=True):
        if gsd_pred_ids:
            return {pred_id:data[pred_id] for pred_id in pred_ids}
        pred_ids_without_gsds = []
        for pred_id in pred_ids:
            ortho_name, building_id, _, _ = parse_pred_key(pred_id)
            pred_ids_without_gsds.append(joint_file_pred_key(ortho_name, building_id))
        return {pred_id:data[pred_id] for pred_id in pred_ids_without_gsds}

    def _get_aligned_pred_ids_on_field_value(self, field_name, field_value):
        valid_pred_ids = []
        rejected_preds_count = 0
        found_field_value = None
        for pred_id in self._unaligned_preds:
            ortho_name, building_id, gsd_x, gsd_y = parse_pred_key(pred_id)
            if field_name == "gsd_x":
                found_field_value = gsd_x
            elif field_name == "gsd_y":
                found_field_value = gsd_y
            elif not field_name is None:
                found_field_value = self._orthomosaic_stats.loc[ortho_name][field_name]
            valid_field = field_name is None or field_value is None or found_field_value == field_value
            valid_actual = joint_file_pred_key(ortho_name, building_id) in self._unaligned_actuals
            if valid_field and valid_actual:
                valid_pred_ids.append(pred_id)
            else:
                rejected_preds_count += 1
        return valid_pred_ids, rejected_preds_count
        
    def get_predicted_values(self, field):
        result = []
        for pred_id in self._unaligned_preds:
            ortho_name, _, gsd_x, gsd_y = parse_pred_key(pred_id)
            if field == "gsd_x":
                result.append(gsd_x)
            elif field == "gsd_y":
                result.append(gsd_y)
            else:
                result.append(self._orthomosaic_stats.loc[ortho_name][field])
        return list(set(result))
        
    def subset(self, field_name=None, field_value=None):
        subset_pred_ids, _ = self._get_aligned_pred_ids_on_field_value(field_name, field_value)
        subset_preds = self._get_subset_dict(self._unaligned_preds, subset_pred_ids, gsd_pred_ids=True)
        subset_actuals = self._get_subset_dict(self._unaligned_actuals, subset_pred_ids, gsd_pred_ids=False)
        return AlignedPredsActuals(subset_preds, subset_actuals, self._orthomosaic_stats, self._channel_map)

    def resample(self, n_samples, replace=True, stratify=None):
        # We resample the deterministic ordering instead of dict keys to avoid duplicate key overwrites.
        resampled_ordering = resample(self._deterministic_ordering, replace=replace, stratify=stratify, n_samples=n_samples)
        
        # Create a shallow copy and override the ordering for this resampled instance.
        resampled_bundle = copy.copy(self)
        resampled_bundle._deterministic_ordering = resampled_ordering
        resampled_bundle._verbose = False
        return resampled_bundle

    def get_actual_class_counts(self):
        result = {l:0 for l in self.get_labels()}
        # Load the actual labels from the dataset
        for label in self._unaligned_actuals.values():
            result[label] += 1
        return result
        
    def get_predicted_class_counts(self):
        result = {l:0 for l in self.get_labels()}
        # Load the predicted labels from the predictions
        for pred_data in self._unaligned_preds.values():
            result[pred_data["label"]] += 1
        return result
        
    def __len__(self):
        return len(self._deterministic_ordering)

def compute_AUCROC(aligned_preds_actuals_bundle):
    # Compute the AUC ROC
    bda_auc_roc = {}
    for label in aligned_preds_actuals_bundle.get_labels():
        try:
            is_actual_label = [float(a[label]) for a in aligned_preds_actuals_bundle.get_preds_multilabel_dict()]
            is_pred_label = [p[label] for p in aligned_preds_actuals_bundle.get_actuals_multilabel_dict()]
            bda_auc_roc[label] = roc_auc_score(is_actual_label, is_pred_label)
        except ValueError:
            print("Warning: Unable to compute AUC ROC for", label, " due to insufficient positive or negative examples.")
    return bda_auc_roc

def compute_measure(aligned_preds_actuals_bundle, measure_func, measure_type="class_level", zero_division=None):
    if measure_type == "class_level":
        bda_measure = {}
        for label in aligned_preds_actuals_bundle.get_labels():
            preds = [a[label] for a in aligned_preds_actuals_bundle.get_preds_multilabel_dict()]
            actuals = np.around([p[label] for p in aligned_preds_actuals_bundle.get_actuals_multilabel_dict()])
            if zero_division is None:
                bda_measure[label] = measure_func(preds, actuals)
            else:
                bda_measure[label] = measure_func(preds, actuals, zero_division=zero_division)
        return bda_measure
    if zero_division is None:
        return measure_func(aligned_preds_actuals_bundle.get_actuals_labels(),
                            aligned_preds_actuals_bundle.get_preds_labels(),
                            average=measure_type)
    return measure_func(aligned_preds_actuals_bundle.get_actuals_labels(),
                        aligned_preds_actuals_bundle.get_preds_labels(),
                        average=measure_type,
                        zero_division=zero_division)

def compute_f1(aligned_preds_actuals_bundle, measure_type):
    return compute_measure(aligned_preds_actuals_bundle, f1_score, measure_type, zero_division=np.nan)
def compute_precision(aligned_preds_actuals_bundle, measure_type):
    return compute_measure(aligned_preds_actuals_bundle, precision_score, measure_type, zero_division=np.nan)
def compute_recall(aligned_preds_actuals_bundle, measure_type):
    return compute_measure(aligned_preds_actuals_bundle, recall_score, measure_type, zero_division=np.nan)
def compute_accuracy(aligned_preds_actuals_bundle):
    return compute_measure(aligned_preds_actuals_bundle, accuracy_score, "class_level")
def compute_confusion_matrix(aligned_preds_actuals_bundle):
    return confusion_matrix(y_true=aligned_preds_actuals_bundle.get_actuals_labels(),
                            y_pred=aligned_preds_actuals_bundle.get_preds_labels(),
                            labels=aligned_preds_actuals_bundle.get_labels())

def compute_confidence_interval(aligned_preds_actuals_bundle, n_samples, confidence_level, metric_func):
    print("Computing confidence interval...")
    bootstrapped_scores = []
    
    N = len(aligned_preds_actuals_bundle)
    y_true = aligned_preds_actuals_bundle.get_actuals_labels()
    
    for i in range(n_samples): 
        # Perform stratified sampling with replacement
        resampled_aligned_bundle = aligned_preds_actuals_bundle.resample(replace=True, stratify=y_true, n_samples=N)
        
        # Compute the metric
        score = metric_func(resampled_aligned_bundle)
        bootstrapped_scores.append(score)
        
    # Calculate the Percentile Confidence Interval
    alpha = (1.0 - confidence_level) / 2.0
    
    # Find the upper and lower bounds based on the observed percentiles
    lower_bound = np.percentile(bootstrapped_scores, alpha * 100)
    upper_bound = np.percentile(bootstrapped_scores, (1.0 - alpha) * 100)
    print("Done")
    
    return {"lower_bound_"+str(confidence_level): lower_bound,
            "upper_bound_"+str(confidence_level): upper_bound}

def generate_metrics_payload(aligned_preds_actuals_bundle, do_confidence=False, n_samples=1000, confidence_level=0.95):
    return {
        "samples": {
            "total": len(aligned_preds_actuals_bundle),
            "actual_class_counts": aligned_preds_actuals_bundle.get_actual_class_counts(),
            "predicted_class_counts": aligned_preds_actuals_bundle.get_predicted_class_counts()
        },
        "metrics": {
            "AUC_ROC": {
                "class_level": compute_AUCROC(aligned_preds_actuals_bundle),
            },
            "F1": { 
                "class_level": compute_f1(aligned_preds_actuals_bundle, "class_level"),
                "macro": {"score": compute_f1(aligned_preds_actuals_bundle, "macro"),
                          "confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                                                                                                            n_samples,
                                                                                                            confidence_level,
                                                                                                            lambda x:compute_f1(x, "macro"))},
                "micro": {"score": compute_f1(aligned_preds_actuals_bundle, "micro")},
                          #"confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                          #                                                                                  n_samples,
                          #                                                                                  confidence_level,
                          #                                                                                  lambda x:compute_f1(x, "micro"))}
            },
            "Accuracy": {
                "class_level": compute_accuracy(aligned_preds_actuals_bundle),
            },
            "Precision": {
                "class_level": compute_precision(aligned_preds_actuals_bundle, "class_level"),
                "macro": {"score": compute_precision(aligned_preds_actuals_bundle, "macro")},
                          #"confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                          #                                                                                  n_samples,
                          #                                                                                  confidence_level,
                          #                                                                                  lambda x:compute_precision(x, "macro"))},
                "micro": {"score": compute_precision(aligned_preds_actuals_bundle, "micro")},
                          #"confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                          #                                                                                  n_samples,
                          #                                                                                  confidence_level,
                          #                                                                                  lambda x:compute_precision(x, "micro"))}
            },
            "Recall": {
                "class_level": compute_recall(aligned_preds_actuals_bundle, "class_level"),
                "macro": {"score": compute_recall(aligned_preds_actuals_bundle, "macro")},
                          #"confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                          #                                                                                  n_samples,
                          #                                                                                  confidence_level,
                          #                                                                                  lambda x:compute_recall(x, "macro"))},
                "micro": {"score": compute_recall(aligned_preds_actuals_bundle, "micro")},
                          #"confidence_interval": None if not do_confidence else compute_confidence_interval(aligned_preds_actuals_bundle,
                          #                                                                                  n_samples,
                          #                                                                                  confidence_level,
                          #                                                                                  lambda x:compute_recall(x, "micro"))}
            },
            "Confusion_Matrix": {
                "matrix": compute_confusion_matrix(aligned_preds_actuals_bundle).tolist(),
                "class_labels": aligned_preds_actuals_bundle.get_labels(),
            },
        }
    }

def compute_metrics_per_unique_label(aligned_preds_actuals_bundle, field, do_confidence=False, n_samples=1000, confidence_level=0.95):
    field_tmp = field
    if isinstance(field, str):
        field_tmp = [field]
    return recurse_compute_metrics_per_unique_label(aligned_preds_actuals_bundle, field_tmp, do_confidence, n_samples, confidence_level)

def recurse_compute_metrics_per_unique_label(aligned_preds_actuals_bundle, fields, do_confidence, n_samples, confidence_level):
    result = {}
    field = fields[0]
    for value in aligned_preds_actuals_bundle.get_predicted_values(field):
        bundle_subset = aligned_preds_actuals_bundle.subset(field, value)
        if len(fields) == 0:
            pass
        elif len(fields) == 1:
            if len(bundle_subset) > 0:
                result[value] = generate_metrics_payload(bundle_subset, do_confidence, n_samples, confidence_level)
        else:
            result[value] = recurse_compute_metrics_per_unique_label(bundle_subset, fields[1:], do_confidence, n_samples, confidence_level)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    add_hyperparameters_files_to_parse_args(parser,
                                            add_dataset_paths_file_path=True,
                                            add_data_source_config_parameters_file_path=True)
    parser.add_argument("--preds_paths", type=str, nargs="+", help="The path to file that contains the model predicitons.")
    parser.add_argument("--channels_hyperparameters_file_path", type=str, help="The path to the file that maps dataset labels to model outputs.")
    parser.add_argument("--metrics_file", type=str, help="The path to the file where the metrics will be stored.")
    parser.add_argument("--dataset_subset", type=str, default="test",  help="The key in the dataset path yaml to access.")
    parser.add_argument("--metrics_to_report", type=str, default="global_metrics>metrics>F1>macro,per_gsd_metrics>*>metrics>F1>macro",
        help="The metric that will be printed to the log for easier monitoring")
    parser.add_argument("--confidence_interval_N", type=int, default=1000,
        help="The number of samples to draw when bootstrapping confidence intervals.")
    parser.add_argument("--confidence_level", type=float, default=0.95,
        help="The confidence level for the produced confidence interval.")
    args = parser.parse_args()

    print("Parsing the channel parameters...")
    channel_parameters = parse_hyperparameters(args.channels_hyperparameters_file_path)

    print("Parsing data paths and evaluation parameters...")
    dataset_paths = parse_hyperparameters(args.dataset_paths_file_path, verbose=False)
    data_source_config_parameters = parse_hyperparameters(args.data_source_config_parameters_file_path, verbose=False)

    print("Parsing the orthomosaics statistics metadata file...")
    paresed_orthomosaic_stats = pd.read_csv(dataset_paths["statistics"], header=0, index_col="Orthomosaic")
    paresed_orthomosaic_stats["Orthomosaic"] = paresed_orthomosaic_stats.index

    print("Locating output metrics location...")
    metrics_dir = os.path.dirname(args.metrics_file)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)
        print("Created the directory to store the output: " + str(metrics_dir))

    print("Parsing passed predictions files...")
    all_preds = {}
    model_names = []
    for passed_preds_path in args.preds_paths:
        parsed_model_name, parsed_preds = parse_model_name_and_predicted_labels(passed_preds_path)
        all_preds = all_preds | parsed_preds
        model_names.append(parsed_model_name)

    print("Parsing ground truth labels files...")
    ground_truth_labels = {}
    for source in data_source_config_parameters[args.dataset_subset]:
        if data_source_config_parameters[args.dataset_subset][source] is True:
            # Parse the actual labels and add the labels to the combined dictionary
            ground_truth_labels = ground_truth_labels | parse_actual_labels(dataset_paths[args.dataset_subset][source]["bda"]["annotations_path"], channel_parameters)

    print("Aligning predictions and actuals...")
    parsed_aligned_preds_actuals_bundle = AlignedPredsActuals(all_preds, ground_truth_labels, paresed_orthomosaic_stats, channel_parameters)

    print("Computing metrics")
    metrics = {"model_name": list(set(model_names))}
    metrics = metrics | {"global_metrics": generate_metrics_payload(parsed_aligned_preds_actuals_bundle, True, args.confidence_interval_N, args.confidence_level)}
    metrics = metrics | {"per_mapper_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "Mapper")}
    metrics = metrics | {"per_ortho_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "Orthomosaic")}
    metrics = metrics | {"per_gsd_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "gsd_x")}
    metrics = metrics | {"per_source_gsd_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, ["Source", "gsd_x"])}
    metrics = metrics | {"per_collection_platform_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "Platform / Provider")}
    metrics = metrics | {"per_source_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "Source")}
    metrics = metrics | {"per_pre_or_post_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "Pre/Post Event")}
    metrics = metrics | {"per_event_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, "Event")}
    metrics = metrics | {"per_source_event_metrics": compute_metrics_per_unique_label(parsed_aligned_preds_actuals_bundle, ["Source", "Event"])}

    reported_metrics = parse_reportable_metrics(metrics, args.metrics_to_report)
    for key in reported_metrics:
        print(key)

    metrics_json = json.dumps(metrics, indent=4)
    with open(args.metrics_file, "w") as metrics_file_object:
        metrics_file_object.write(metrics_json)
    print("Done.")