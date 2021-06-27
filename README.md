
# Installation Requirement

- python 3.6
- pytorch 1.1.0 （`conda install pytorch torchvision torchaudio`)

```
pip install -r requirements.txt
```

# Run The Demo
To try the demo, run `streamlit run main.py --server.maxUploadSize 1`

One-click run version (including the installation, but might not work properly): `bash run.sh`

# Files
```
├── main.py
├── README.md
├── make_naive.py
├── auto_localization.py
├── harmonize.py
├── baiduAPI.py
├── requirements.txt
├── run.sh
├── Con_Sin_GAN
│   ├── ConSinGAN
│   ├── evaluate_model.py
│   └── main_train.py
├── Images
│   ├── default_human
│   ├── default_paintings
│   └── tmp
├── TrainedModels
│   ├── nightsky
│   ├── oil_building
│   ├── oil_tree
│   ├── pen_tree
│   ├── scream
│   ├── some_painting
│   ├── street
│   └── watercolor_building
└── MySegmentation
    ├── dataset.py
    ├── gen_mask.py
    ├── loadmodel.py
    ├── log
    ├── models
    ├── padding_data.py
    ├── pics
    ├── print_log
    ├── README.md
    └── segmentation_train.py

```
## Con_Sin_GAN
Con_Sin_GAN is the directory for ConSinGAN implementations obtained from https://github.com/tohinz/ConSinGAN and modified by us for this project.

## Images

This directory saves the default painting images and human images, as well as some temporary images.

## TrainedModels

Models in this directory corresponds to the default painting images.
## MySegmentation

MySegmentation provides a FPN model implemented by ourselves. (But in our final demo, we use BaiduAPI for this project due to its better performance.)
### 1. Train Model:

#### Dataset:

​	Download [Supervisely Person Dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets).

​	Use `./gen_mask.py` to extract label from `.json` file.

​	Use `./padding_data.py` to get images with same size.	

#### Train:

```
nohup python -u  segmentation_train.py  &> ./log/CVFPN.out&
```

#### Log:

Use `./print_log/printlog.ipynb` to print the log.

### 2. Use Model:

Use `python loadmodel.py` to get the segmentation result. 