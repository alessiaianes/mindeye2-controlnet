#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

# pip install --upgrade pip
# C:/Users/User/Desktop/coding/mindeye2-hidiff/fmri/Scripts/python.exe -m pip install --upgrade pip

# python3.11 -m venv fmri
python -m venv fmri
# source fmri/bin/activate
source fmri/Scripts/activate
C:/Users/User/Desktop/coding/mindeye2-hidiff/fmri/Scripts/python.exe -m pip install --upgrade pip


pip install numpy matplotlib jupyter jupyterlab_nvdashboard jupyterlab tqdm scikit-image accelerate webdataset pandas einops ftfy regex kornia h5py open_clip_torch torchvision torch transformers xformers torchmetrics diffusers wandb omegaconf pytorch-lightning sentence-transformers evaluate nltk rouge_score umap-learn deepspeed
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install dalle2-pytorch
