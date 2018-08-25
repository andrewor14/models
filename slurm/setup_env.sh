#!/bin/bash

module load cudnn/cuda-9.1/7.1.2
module load anaconda3/5.0.1

# Note: This makes sure we're running tensorflow with GPU support
# This is commented out because we only need to do this once
#pip3 uninstall -y tensorflow
#pip3 uninstall -y tensorflow-gpu
#pip3 install --user tensorflow-gpu

python test_gpu_support.py

