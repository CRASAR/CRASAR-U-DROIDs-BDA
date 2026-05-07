# Modeling
This readme provides documentation and instructions on how to create, train, infer, and evaluate Models. 

## Create New Models

### Adding Models to the repo

Consider following these steps to add a new model to the repository. 
1. Create a new folder under this directory /src/modeling/Models/ for the new Model
2. If you have a backbone (e.g., ViT), please add it under this directory /src/modeling/Models/Backbones/
2. Add the model to the /src/modeling/Models/model_registry.py under the STR2MODELCLASS mapping

### Generate Hyperparameters files

The following script can be used to generate a template hyperparameters files.

```rb
/src/modeling/Models/create_hyperparameters_file.py --out_path path/to/output --channel_maps_file path/to/channel_maps_file --class_weights_file path/to/class_weight_file --task <Task Name (e.g., BDA, RDA, BDAADJ)>
```
- channel_maps_files are located at /src/modeling/Models/Hyperparameters/channel_maps/
- class_weights_files are located at /src/modeling/Models/Hyperparameters/class_weights/

NOTE: The generated template hyperparameters files is not sufficient to train/infer the models. It is only to generate a base hyperparameters file to modify further.

## Train Existing Models

The following script can be used to train an existing model.

```rb
/src/modeling/train.py 
```

Training batch scripts are under the directory /scripts/bat_scripts/modeling/<ModelName>/

## Infer with Existing Models

The following script can be used to infer with a trained model.

```rb
/src/modeling/infer.py 
```

Inference batch scripts are under the directory /scripts/bat_scripts/inference/<ModelName>/

## Evaluate Models

Based on the task for the model, these are the scripts to run evaluation. 
- BDA: /src/modeling/evaluate_RDA.py
- RDA: /src/modeling/evaluate_RDA.py