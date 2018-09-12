#!/bin/bash

TF_MODELS=/home/andrewor/models
DATASET_DIR=/tigress/andrewor/dataset/cifar10-dataset
MODEL_DIR=/tigress/andrewor/logs/resnet_cifar10_model_"$1"

export PYTHONPATH="$PYTHONPATH:$TF_MODELS"
cd $TF_MODELS/official/resnet
python cifar10_main.py --data_dir="$DATASET_DIR" \
                       --model_dir="$MODEL_DIR" \
                       --num_gpus=4 \
                       --train_epochs=1000 \
                       --resnet_size=32 \
                       --epochs_between_evals=3

