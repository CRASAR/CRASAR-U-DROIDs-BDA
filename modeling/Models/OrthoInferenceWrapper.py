from collections import defaultdict
from dataset.constants import BDA_DAMAGE_CLASSES

def joint_file_pred_key(ortho_name, prediction_id, gsd_x=None, gsd_y=None):
    gsd_key = ""
    gsd_key = ("" if gsd_x is None else str(gsd_x)) + "_" + ("" if gsd_y is None else str(gsd_y))
    return gsd_key + "_" + ortho_name + "_" + prediction_id

def parse_pred_key(pred_key):
    splits = pred_key.split("_")
    gsd_x = splits[0]
    gsd_y = splits[1]
    prediction_id = splits[-1]
    ortho_name = "_".join(splits[2:-1])
    return ortho_name, prediction_id, gsd_x, gsd_y

# Perform fusion just by taking the majority class
def fuse_bda_tiled_inference(tiled_preds, class_labels=None):
    if class_labels is None:
        class_labels = BDA_DAMAGE_CLASSES
    fused_labels = {}

    missing_labels = []
    for prediction_id, inferences in tiled_preds.items():
        label_totals = defaultdict(lambda: 0)
        total = 0
        for inference in inferences:
            for label in class_labels:
                try:
                    val = inference["class_preds"][label]
                    label_totals[label] += val
                    total += val
                except KeyError:
                    missing_labels.append(label)

        # Prevent divide by zero
        aggregated_label = max(label_totals.items(), key=lambda x: x[1])[0]
        if total == 0:
            print("Warning! fuse_bda_tiled_inference found no labels to fuse.")
            total = 1

        fused_labels[prediction_id] = {
            "label": aggregated_label,
            "confidence": label_totals[aggregated_label] / total
        }
    if len(missing_labels) > 0:
        missing_labels = set(missing_labels)
        print("Warning! Found label(s)", missing_labels, "which was/were not found in class preds.")
    return fused_labels
