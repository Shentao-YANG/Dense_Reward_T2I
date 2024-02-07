#!/bin/bash

conda init bash
source ~/.bashrc

# Create a conda environment and install required modules.
conda env create -f environment.yaml
conda activate rldiff
