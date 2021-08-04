# Latent Structure Model
Model to construct material state transfer graphs from wet lab protocols by learning intermediate latent structures as described in [Learning Latent Structures for Cross Action Phrase Relations in Wet Lab Protocols](https://aclanthology.org/2021.acl-long.525.pdf). 

![main figure](https://github.com/chaitanya2334/lsm/blob/master/lsm.jpg?raw=true)

# Setup
---
## Requirements
- python 3.7 (tested with 3.7.6)
- pytorch (tested with 1.7)
- transformers (tested with 3.1.0)
- matplotlib (tested with 3.1.3)
- hydra core (tested with 1.0.3)
- numpy (tested with 1.18.1)
- pandas (tested with 1.0.1)
- pytorch-lightning (tested with 1.0.4)
- scikit-learn (tested with 0.22.2)
- scipy (tested with 1.5.4)
- tensorboard (tested with 2.4.0)
- tqdm (tested with 4.43.0)


### Install all requirements
```
conda env create --file=environment.yaml
```
### Load conda env
```
conda activate lsm
```

## Convert Data
The wlp-mstg-dataset is provided as a submodule in this repository. However, is present in the standoff format. 
This must be converted into json format, before running the code.
```
python -m src.preprocessing.to_json
```

## Train and test the model

The configuration file `src/configs/igcn.yaml` contains all the hyperparameters used for the best model.
```
python -m src.trainers.igcn_trainer
```
