#!/bin/bash

TF_MODELS=/home/andrewor/models
DATASET_DIR=/tigress/andrewor/dataset/cifar10-dataset
MODEL_DIR=/tigress/andrewor/logs/resnet_cifar10_model

module load cudnn/cuda-9.1/7.1.2
module load anaconda3/5.0.1
export PYTHONPATH="$PYTHONPATH:$TF_MODELS"
cd $TF_MODELS/official/resnet
python cifar10_main.py --data_dir=$DATASET_DIR \
                       --model_dir=$MODEL_DIR \
                       --num_gpus=4 \
                       --train_epochs=1000

