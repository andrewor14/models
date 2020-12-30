#!/bin/bash

export RANK="${RANK:=0}"
export JOB_NAME="testing-$(date +%m_%d_%y_%s%3N)-rank-$RANK"
export NUM_GPUS="${NUM_GPUS:=1}"
export DATA_DIR="${DATA_DIR:=/root/dev/dataset/cifar10/cifar-10-batches-bin}"
export BATCH_SIZE="${BATCH_SIZE:=128}"
export NUM_STEPS="${NUM_STEPS:=10}"
export TRAIN_DIR="${TRAIN_DIR:=/root/dev/train_data/${JOB_NAME}}"
export DISTRIBUTION_STRATEGY="${DISTRIBUTION_STRATEGY:=mirrored}"
export LOG_FILE="${LOG_FILE:=/root/dev/logs/${JOB_NAME}.log}"

mkdir -p "$TRAIN_DIR"

python3 /root/dev/models/official/benchmark/models/resnet_cifar_main.py\
  --num_gpus="$NUM_GPUS"\
  --data_dir="$DATA_DIR"\
  --batch_size="$BATCH_SIZE"\
  --train_steps="$NUM_STEPS"\
  --model_dir="$TRAIN_DIR"\
  --log_steps="1"\
  --skip_eval="true"\
  --distribution_strategy="$DISTRIBUTION_STRATEGY"\
  >> "$LOG_FILE" 2>&1

