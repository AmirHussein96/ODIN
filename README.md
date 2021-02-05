# ODIN

This repository contains the code used for the paper titled "Domain Adaptation with Representation Learning and Nonlinear Relation for Time Series" by Hussein A., Hajj H. [comming soon] 

## High level approach description

![Alt text](images/high_level.png?raw=true "proposed_approach")

## Model

![Alt text](images/DA_AE_soft.png?raw=true "proposed_approach")

## Requirements

- python version ` 3.6.12 `
- To create anaconda environment run `conda env create -f environment.yml`

## Main Folders Description

- CHBMIT and FB: Raw dataset folders. 
- CHBMIT_cache and FB_cache: Prepared data folders.
- models/: Model source code.
- utils/: Helping modules to load and prepare the data.


## Quick start

1. Download the PAR/HARR datasets
    - PAR: https://sensor.informatik.uni-mannheim.de/#dataset
    - HARR: http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition#:~:text=The%20Heterogeneity%20Dataset%20for%20Human,%2C%20feature%20extraction%2C%20etc.
2. Run ```main.py``` in one of the following modes: 
    - `cr_user`: cross user
    - `cr_device`: cross device 
    - `cr_user_device`: cross user and cross device

```
python main.py --dataset PAR --mode cr_user --path "path/to/dataset"
```



## Sample of reconstructed signals from test set after adaptation

- Run ```inspect_AE.py``` to generate sample figures of advesarial examples 
![Alt text](images/Figure_1.png?raw=true "rec1")
![Alt text](images/Figure_2.png?raw=true "rec2")


## Results

![Alt text](images/shifts.PNG?raw=true "shifts")

![Alt text](images/shifts_dann.PNG?raw=true "shifts_dann")

![Alt text](images/shifts_odin.PNG?raw=true "shifts_odin")

## Contacts

- [Amir Hussein](https://github.com/AmirHussein96) anh21@mail.aub.edu 


## Paper:
[comming soon]

