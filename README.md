PestSpace

PestSpace is a deep learning project for plant disease classification, supporting both single-plant and multi-plant scenarios. 
The project uses Hydra configuration manager to control experiments and to easily switch between models, datasets, and hyperparameters.
All model parameters, training settings, and dataset paths are defined in configuration file config.yaml located in the conf/ folder.


##Repository Structure

PestSpace/
├── conf
│   ├── config.yaml                      # configure run parameters
│   └── model
│       ├── EfficientnetB0.yaml          # EfficientNetB0 model parameters
│       ├── Resnet18.yaml                # ResNet18 model parameters
│       └── multiEfficientnetB0.yaml     # EfficientNetB0 model parameters
├── data
│   ├── data.py                          # data for single plant
│   ├── multidata.py                     # data for multiple plants
│   └── multidata_valPS.py               # data for multiple plants with validation only on PestSpace
├── models
│   ├── EfficientnetB0.py                # EfficientNetB0 model for single plant
│   ├── Resnet18.py                      # ResNet18 model for single plant
│   └── multiEfficientnetB0.py           # EfficientNetB0 model for multiple plants
├── environment.yml                      # dependencies
├── train.py                             # runs training process
└── README.md
##

Workflow

- Create the Conda environment using environment.yml and activate it.
- Configure the experiment by editing conf/config.yaml, including selecting the model, dataset, and training parameters.
- Configure model specific parameters in conf/model/model.yaml.
- In train.py, import the desired dataset loader depending on the experiment (e.g., from data.data import get_data_loaders or from data.       multidata import get_data_loaders).
- Run training from the project root using python train.py.


Author

Fred Väärtnõu
