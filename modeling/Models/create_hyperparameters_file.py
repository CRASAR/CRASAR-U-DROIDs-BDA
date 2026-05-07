import os
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Empty Hyperparameters File for model."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="The path to folder where the hyperparameters files should be saved to.",
        default="./",
    )
    parser.add_argument(
        "--channel_maps_file",
        type=str,
        help="Path to hyperparameters file for channel maps for given task.",
        default="./",
    )
    parser.add_argument(
        "--class_weights_file",
        type=str,
        help="Path to hyperparameters file for channel maps for given task.",
        default="./",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The task for the model. If RDA, specify the variations [-binary, -simple, -full] (e.g., RDA-full)",
        default="BDA",
    )
    args = parser.parse_args()

    # Read the yaml file for channel maps
    channel_maps = {}
    with open(args.channel_maps_file, "r") as file:
        channel_maps = yaml.safe_load(file)

    # Read the class weights file for channel maps
    class_weights = {}
    with open(args.class_weights_file, "r") as file:
        class_weights = yaml.safe_load(file)

    # Hyperparameters for model training
    hyperparameters = {
        "input": {
            "normalized_inputs": True,
            "channels": {"red": 0, "green": 1, "blue": 2, "mask": 3},
            "training_parameters": {
                "grad_accumulation": 4,
                "batch_size": 2,
                "samples_per_epoch": 40,
                "max_epochs": 100,
                "gamma": 2,
                "alpha": 0.25,
                "l1_reg": 0.000001,
                "log_images_every_n_steps": 10,
                "tile_x": 2048,
                "tile_y": 2048,
                "mask_x": 2048,
                "mask_y": 2048,
                "optimizer_parameters": {"learning_rate": 0.0005},
            "model_parameters": {},
            "loss_parameters": {
                "loss": "cross entropy",
            }
            },
        }
    }

    # Combine Dictionaries
    hyperparameters["channel_maps"] = channel_maps
    hyperparameters["input"]["training_parameters"][
        "output_class_weights"
    ] = class_weights

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # File path to save the hyperparameters YAML file
    output_file = os.path.join(args.out_path, args.task + "hyperparameters.yaml")

    # Write the hyperparameters to the YAML file
    with open(output_file, "w") as file:
        yaml.dump(hyperparameters, file, default_flow_style=False)

    print("Hyperparameters YAML file saved at " + args.out_path)
    print("WARNING: This hyparameters file is not sufficient to run any model. It merely provides the basics for it, please edit it according to model.")
