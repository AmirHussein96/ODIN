# ODIN

This repository contains the code used for the paper titled [Domain Adaptation with Representation Learning and Nonlinear Relation for Time Series" by Hussein A., Hajj H](https://dl.acm.org/doi/10.1145/3502905).

## High level approach description

![Alt text](images/high_level.png?raw=true "proposed_approach")

## Model

![Alt text](images/odin_stage2.png?raw=true "proposed_approach")

## Requirements

- python version ` 3.6.12 `
- To create anaconda environment run `conda env create -f environment.yml`


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


## Domain adaptation toy  example

Toy examples for the limitations of domain adaptation with hard parameter sharing and how domain adaptation with soft parameter sharing overcomes these limitations [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1APQdDNW4zwRemgWM1mpTsjgb4-rcKVT9?usp=sharing)

![Alt text](images/shifts.PNG?raw=true "shifts")

![Alt text](images/shifts_dann.PNG?raw=true "shifts_dann")

![Alt text](images/shifts_odin.PNG?raw=true "shifts_odin")



## Contacts

- [Amir Hussein](https://github.com/AmirHussein96) anh21@mail.aub.edu 


## Cite Paper:
```
@article{hussein2022domain,
  title={Domain Adaptation with Representation Learning and Nonlinear Relation for Time Series},
  author={Hussein, Amir and Hajj, Hazem},
  journal={ACM Transactions on Internet of Things},
  volume={3},
  number={2},
  pages={1--26},
  year={2022},
  publisher={ACM New York, NY}
}
```
