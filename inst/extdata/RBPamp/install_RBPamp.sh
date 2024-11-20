#!/bin/bash

#to run this script in command line: bash -i install_RBPamp.sh
#run commands
conda env create -f RBPamp.yml -n RBPamp
conda activate RBPamp
conda install -c anaconda cython
python setup.py build
python -m pip install .