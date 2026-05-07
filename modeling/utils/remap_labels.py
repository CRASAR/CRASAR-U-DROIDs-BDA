import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--preds_path", type=str,
        help="The path to file that contains the model predicitons.")
    parser.add_argument("--target_label", type=str,
        help="The label value that needs to be changed.")
    parser.add_argument("--destination_label", type=str,
        help="The value that should be put in place of the target label.")
    parser.add_argument("--out_path", type=str,
        help="The path to file that contains the model predicitons.")
    args = parser.parse_args()

    with open(args.preds_path, "r") as f:
        preds_data = json.load(f)

    for pred in preds_data["preds"]:
        if preds_data["preds"][pred]["label"] == args.target_label:
            preds_data["preds"][pred]["label"] = args.destination_label

    with open(args.out_path, "w") as f:
        f.write(json.dumps(preds_data))
