# ODIN

This repository contains the code used for the paper titled "Domain Adaptation with Representation Learning and Nonlinear Relation for Time Series" by Hussein A., Hajj H. [comming soon] 

## Requirements

- python version ` 3.6.12 `
- To create anaconda environment run `conda env create -f environment.yml`

## Main Folders Description

- CHBMIT and FB: Raw dataset folders. 
- CHBMIT_cache and FB_cache: Prepared data folders.
- models/: Model source code.
- utils/: Helping modules to load and prepare the data.


## Quick start

1. Download the two datasets (CHBMIT and FB) and move them into their folders.
    - CHBMIT: http://physionet.org/physiobank/database/chbmit/
    - FB: http://epilepsy.uni-freiburg.de.
2. Run ```main.py```
```
python main.py --mode without_AE --dataset CHBMIT
```

## Model

![Alt text](images/proposed_approach.PNG?raw=true "proposed_approach")


## Generate a Sample of Adversarial Examples

- Run ```inspect_AE.py``` to generate sample figures of advesarial examples 
![Alt text](images/AE_noise.png?raw=true "AE")


## Results

![Alt text](images/tsne.PNG?raw=true "tsne")

## Contacts

- [Amir Hussein](https://github.com/AmirHussein96) anh21@mail.aub.edu 


## Paper:
[comming soon]

