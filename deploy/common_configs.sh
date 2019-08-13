#!/bin/bash

# ==========================================================
#  A central collection of hard-coded environment variables
# ==========================================================

export ENVIRONMENT="$(hostname | awk -F '[.-]' '{print $1}' | sed 's/[0-9]//g')"
export DEFAULT_NUM_NODES="4"
export DEFAULT_NUM_PARAMETER_SERVERS="1"
export DEFAULT_NUM_WORKERS_PER_NODE="1"

if [[ "$ENVIRONMENT" = "tiger" ]]; then
  export ENVIRONMENT="tigergpu"
fi

if [[ "$ENVIRONMENT" = "tigergpu" ]]; then
  export LOG_DIR="/home/andrewor/logs"
  export MODELS_DIR="/home/andrewor/models"
  export IMAGENET_DATA_DIR="/tigress/andrewor/dataset/imagenet-dataset"
  export CIFAR10_DATA_DIR="/tigress/andrewor/dataset/cifar10-dataset/cifar-10-batches-bin"
  export WMT_DATA_DIR="" # TODO: fill this in
  export BASE_TRAIN_DIR="/tigress/andrewor/train_logs/"
  export BASE_EVAL_DIR="/tigress/andrewor/eval_logs/"
  export PYTHON_COMMAND="python3"
  export MPI_HOME="/home/andrewor/lib/openmpi"
elif [[ "$ENVIRONMENT" = "visiongpu" ]]; then
  export LOG_DIR="/home/andrewor/workspace/logs"
  export MODELS_DIR="/home/andrewor/workspace/models"
  export IMAGENET_DATA_DIR="" # TODO: fill this in
  export WMT_DATA_DIR="" # TODO: fill this in
  export CIFAR10_DATA_DIR="/home/andrewor/workspace/dataset/cifar10/cifar-10-batches-bin"
  export BASE_TRAIN_DIR="/home/andrewor/workspace/train_data"
  export BASE_EVAL_DIR="/home/andrewor/workspace/eval_data"
  export PYTHON_COMMAND="python3"
elif [[ "$ENVIRONMENT" == "ns" ]]; then
  export LOG_DIR="/home/andrewor/logs"
  export MODELS_DIR="/home/andrewor/models"
  export IMAGENET_DATA_DIR="/home/andrewor/dataset/imagenet"
  export CIFAR10_DATA_DIR="/home/andrewor/dataset/cifar10/cifar-10-batches-bin"
  export WMT_DATA_DIR="/home/andrewor/dataset/transformer"
  export BASE_TRAIN_DIR="/home/andrewor/train_data"
  export BASE_EVAL_DIR="/home/andrewor/eval_data"
  export PYTHON_COMMAND="/usr/licensed/anaconda3/5.2.0/bin/python3.6"
  # No GPUs on this cluster
  export BYPASS_GPU_TEST="true"
  export MPI_HOME="/home/andrewor/lib/openmpi"
elif [[ "$ENVIRONMENT" = "snsgpu" ]]; then
  export LOG_DIR="/home/andrew/Documents/dev/logs"
  export MODELS_DIR="/home/andrew/Documents/dev/models"
  export IMAGENET_DATA_DIR="" # TODO: fill this in
  export WMT_DATA_DIR="/home/andrew/Documents/dev/dataset/transformer"
  export CIFAR10_DATA_DIR="/home/andrew/Documents/dev/dataset/cifar10/cifar-10-batches-bin"
  export BASE_TRAIN_DIR="/home/andrew/Documents/dev/train_data"
  export BASE_EVAL_DIR="/home/andrew/Documents/dev/eval_data"
  export PYTHON_COMMAND="python3"
  export DEFAULT_NUM_WORKERS_PER_NODE="2"
elif [[ -n "$IN_DOCKER_CONTAINER" ]]; then
  export LOG_DIR="/root/dev/logs"
  export MODELS_DIR="/root/dev/models"
  export IMAGENET_DATA_DIR="/root/dev/dataset/imagenet"
  export CIFAR10_DATA_DIR="/root/dev/dataset/cifar10/cifar-10-batches-bin"
  export WMT_DATA_DIR="/root/dev/dataset/transformer"
  export BASE_TRAIN_DIR="/root/dev/train_data"
  export BASE_EVAL_DIR="/root/dev/eval_data"
  export PYTHON_COMMAND="python3"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  # It's OK to run MPI as root in a container
  export OMPI_ALLOW_RUN_AS_ROOT=1
  export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
else
  echo "ERROR: Unknown environment '$ENVIRONMENT'"
  exit 1
fi

# Helper function to get a human readable timestamp
function get_submit_timestamp() {
  date +%m_%d_%y_%s%3N
}

