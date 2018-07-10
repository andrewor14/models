#!/bin/bash

TF_MODELS=/home/andrewor/models
HOROVOD=/home/andrewor/horovod
DATASET_DIR=/tigress/andrewor/dataset/cifar10-dataset
MODEL_DIR=/tigress/andrewor/logs/resnet_cifar10_model

module load cudnn/cuda-8.0/7.0
module load anaconda3/5.0.1
module load openmpi/cuda-8.0/gcc/3.0.0/64
pip3 install --user horovod
export PYTHONPATH="$PYTHONPATH:$TF_MODELS"

cd $HOROVOD/examples
mpirun -np 4 \
  -H localhost:4 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python tensorflow_mnist.py

