# [ECE549/SP21/Final Project] Plumonary Nodules Classification
Sources code for the final project of ECE549/SP21 class on Plumonary Nodules classification

## Authors:
* **Hieu Hoang**: hthieu@illinois.edu
* **Anh Leu**: anhleu@illinois.edu

## Download dataset
We used the pre-processing scripts for cropping nodules, follow the same pipeline as the original Deep-Lung paper. The
preprocessed LIDIC/IDRI dataset is available at: https://drive.google.com/file/d/1KbklchWZegDlR4ZvLxnwQdzhpJ46YeMH/view?usp=sharing

## Our implementation
Our implementation is encoupled inside the NAS-Lung folder.

* Folder `code/` contains our training utils (for logging, tensorboard, etc.), evaluation scripts, data-preprocessing script, config loader.
* Folder `configs/` contains the training configurations, all hyper-parameters that we used for all experiments.
* Folder `eval_result/` contains all the validation evaluations generated automatically.
* File `losses/loss_multiview.py` contains the implementation of the multi-views consistency loss function.
* Folder `models/` contains all the implmentation of deep architectures, files `dpn3d.py` (taken from the Deep-Lung), `cnn_res.py`  `net_sphere.py` are from the authors of NAS-Lung and Deep-Lung (for reproducing results purposes) with small modifications. The rest are our NAS-Lung with multi-views consistency branch and 3D SE-ResNet re-implementation.   
* Folder `notebooks/` contains the ipynb files for all visualization, data expoloration mentioned in the report.
* Folder `radiomics_feat/` contains the script for runing `pyradiomics`
 library on the LIDC/IDRI dataset, based on the mask generated from the LIDC2DICOM repo.
* File `train.py` also be modified (and simplified comparing to the original implementation) to match with the training utils.
* Regarding to the LIDC2DICOM repo, we used all the functions provided since the main purpose is just for data-conversion. To run this repo, we also exploring DOCKER for the first time since this repo requires other third-party softwares that do not available on our computing system.
## References
This project is built up on the following Github repositories:
* NAS-Lung: https://github.com/fei-hdu/NAS-Lung
* Deep-Lung: https://github.com/wentaozhu/DeepLung
* LIDC2DICOM: https://github.com/QIICR/lidc2dicom

