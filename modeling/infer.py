import os
import json
import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from modeling.Orthomosaic import MultisourceOrthomosaicFactory
from modeling.data_modules.TrainValPredictDataModule import TrainValPredictDataModule
from modeling.DataMap import Labels2IdxMap
from modeling.utils.hyperparameters import (
    parse_hyperparameters,
    add_hyperparameters_files_to_parse_args,
)
from modeling.utils.data_augmentations import get_inference_transforms
from modeling.utils.initialize_dataset import initialize_windowed_dataset
from modeling.Models.model_registry import parse_and_initialize_segmentation_model

local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(
    f"Running process {os.getpid()} on local rank {local_rank} and __name__ = {__name__}"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with Model.")
    add_hyperparameters_files_to_parse_args(
        parser,
        add_dataset_paths_file_path=True,
        add_data_source_config_parameters_file_path=True,
        add_channels_hyperparameters_file_path=True,
        add_infer_datagen_hyperparameters_yaml_path=True,
        add_model_hyperparameters_yaml_path=True,
    )
    parser.add_argument("--model_path", type=str, help="The path to trained model.")
    parser.add_argument(
        "--preds_path",
        type=str,
        help="The path to file where predicitons will be stored.",
    )
    parser.add_argument(
        "--data_gen_workers",
        type=int,
        help="The number of worker processes that will be used for data generation",
        default=12,
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="The floating point precision with which the model should use.",
        default="16-mixed",
    )
    parser.add_argument(
        "--accelerator",
        help="Which hardware accelerator should be used for model training (cpu, gpu, tpu, mps)",
        default="cpu",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        help='The precision to be used by CUDA tensor cores when available. Options are ("medium", "high")',
        default=None,
    )
    parser.add_argument(
        "--default_epsg_int",
        type=int,
        help="The integer used to identify the CRS of orthosmosaics loaded when their transforms are stored in a TFW file.",
        default=None,
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        help="scale factor to downsample/upsample the imagery",
        default=1.0,
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="The backend used to read imagery during validation",
        default="auto",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When set, use simplified parameters to initialize things faster.",
    )
    parser.add_argument(
        "--load_strict",
        action="store_true",
        help="When set, raise errors when there is are unknown parameter values in loaded models.",
    )
    args = parser.parse_args()

    if args.matmul_precision:
        print("Setting matmul precision to " + args.matmul_precision)
        torch.set_float32_matmul_precision(args.matmul_precision)

    print("Inferencing Model stored at " + str(args.model_path))

    # Create the location where the predictions are going to be stored
    preds_folder, _ = os.path.split(args.preds_path)
    if not os.path.exists(preds_folder):
        os.makedirs(preds_folder, exist_ok=True)
        print("Created the directory to store the output: " + str(preds_folder))

    # Initialize the model
    print("Reading hyperparameters...")
    model_hyperparameters = parse_hyperparameters(args.model_hyperparameters_yaml_path)
    channel_parameters = parse_hyperparameters(args.channels_hyperparameters_file_path)
    datagen_hyperparameters = parse_hyperparameters(
        args.infer_datagen_hyperparameters_yaml_path
    )
    dataset_paths = parse_hyperparameters(args.dataset_paths_file_path, verbose=False)
    data_source_config_parameters = parse_hyperparameters(
        args.data_source_config_parameters_file_path, verbose=False
    )

    print("Creating index maps")
    input_dataset_label_map = Labels2IdxMap(
        channel_parameters["channel_maps"]["input_dataset_class_2_idx_map"],
        channel_parameters["channel_maps"]["background_class_idx"],
    )

    # Initialize the model
    print("Initializing Inference Model...")
    print(
        "\tTask:",
        model_hyperparameters["task"],
        "Model Type:",
        model_hyperparameters["model_type"],
    )

    if args.model_path:
        print("Loading model from:", args.model_path)
        model = parse_and_initialize_segmentation_model(
            channel_parameters,
            model_hyperparameters,
            None,
            model_path=args.model_path,
            strict_load=args.load_strict,
        )
    else:
        model = parse_and_initialize_segmentation_model(
            channel_parameters, model_hyperparameters, None
        )
    print("\tDone")
    print("Model Loaded...\n\n")
    print(ModelSummary(model, max_depth=1))
    print("\n\n\n")

    # Get the orthomsaics that will be used for inference
    # pylint: disable=duplicate-code
    print("Initializing Prediction Data Module")
    inference_orthomosaics = MultisourceOrthomosaicFactory(
        dataset_paths_dict=dataset_paths,
        data_source_config_parameters=data_source_config_parameters,
        model_hyperparameters=model_hyperparameters,
        boundary_folder=None,
        train_validation_test="test",
        limit=(1 if args.debug else None),
        backend=args.backend,
        required_channels=[3, 4],
    )
    # pylint: enable=duplicate-code

    dataset = initialize_windowed_dataset(
        inference_orthomosaics,
        channel_parameters,
        model_hyperparameters,
        datagen_hyperparameters,
        get_inference_transforms(),
    )
    print("\tDone")

    print("Initializing Prediction Data Module")
    dm = TrainValPredictDataModule(
        predict_dataset=dataset,
        num_workers=1 if args.debug else args.data_gen_workers,
        predict_batch_size=model_hyperparameters["validation"]["validation_parameters"][
            "batch_size"
        ],
    )

    trainer = Trainer(
        precision=args.precision, accelerator=args.accelerator, num_sanity_val_steps=0
    )

    # Predict the model.
    print("Starting Inference")
    trainer.predict(model, dm)

    # Write the predictions to a file
    if local_rank == 0:
        print("Writing predictions...")
        preds = model.get_predicted_labels()
        with open(args.preds_path, "w") as f:
            f.write(json.dumps({"model_name": model.getName(), "preds": preds}))
        f.close()
        print("Done")
