#!/bin/bash

# ==========================================================
#  A central collection of hard-coded environment variables
# ==========================================================

export ENVIRONMENT="$(hostname | awk -F '[.-]' '{print $1}' | sed 's/[0-9]//g')"
export SLURM_EMAIL="andrewor@princeton.edu"
export DEFAULT_NUM_NODES="4"
export DEFAULT_NUM_PARAMETER_SERVERS="1"

if [[ "$ENVIRONMENT" = "tiger" ]]; then
  export ENVIRONMENT="tigergpu"
fi

if [[ "$ENVIRONMENT" = "tigergpu" ]]; then
  export LOG_DIR="/home/andrewor/logs"
  export MODELS_DIR="/home/andrewor/models"
  export IMAGENET_DATA_DIR="/tigress/andrewor/dataset/imagenet-dataset"
  export CIFAR10_DATA_DIR="/tigress/andrewor/dataset/cifar10-dataset/cifar-10-batches-bin"
  export BASE_TRAIN_DIR="/tigress/andrewor/train_logs/"
  export BASE_EVAL_DIR="/tigress/andrewor/eval_logs/"
  export TF_PKG="/home/andrewor/tensorflow_pkg/tensorflow-1.10.1-cp36-cp36m-linux_x86_64.whl"
  export PYTHON_COMMAND="python3"
  export DEFAULT_NUM_GPUS_PER_WORKER="4"
  export DEFAULT_NUM_GPUS_PER_NODE="4"
  export DEFAULT_NUM_CPUS_PER_NODE="28"
  export DEFAULT_MEMORY_PER_NODE="64G"
  export MPI_HOME="/home/andrewor/lib/openmpi"
elif [[ "$ENVIRONMENT" = "visiongpu" ]]; then
  export LOG_DIR="/home/andrewor/workspace/logs"
  export MODELS_DIR="/home/andrewor/workspace/models"
  export IMAGENET_DATA_DIR="" # TODO: fill this in
  export CIFAR10_DATA_DIR="/home/andrewor/workspace/dataset/cifar10/cifar-10-batches-bin"
  export BASE_TRAIN_DIR="/home/andrewor/workspace/train_data"
  export BASE_EVAL_DIR="/home/andrewor/workspace/eval_data"
  export TF_PKG="/home/andrewor/workspace/tensorflow_pkg/tensorflow-1.12.0rc1-cp35-cp35m-linux_x86_64.whl"
  export PYTHON_COMMAND="python3"
  export DEFAULT_NUM_GPUS_PER_WORKER="2"
  # No slurm on this machine, not used
  export DEFAULT_NUM_GPUS_PER_NODE=""
  export DEFAULT_NUM_CPUS_PER_NODE=""
  export DEFAULT_MEMORY_PER_NODE=""
elif [[ "$ENVIRONMENT" == "ns" ]]; then
  export LOG_DIR="/home/andrewor/logs"
  export MODELS_DIR="/home/andrewor/models"
  export IMAGENET_DATA_DIR="" # TODO: fill this in
  export CIFAR10_DATA_DIR="/home/andrewor/dataset/cifar10/cifar-10-batches-bin"
  export BASE_TRAIN_DIR="/home/andrewor/train_data"
  export BASE_EVAL_DIR="/home/andrewor/eval_data"
  export TF_PKG="" # TODO: fill this in
  export PYTHON_COMMAND="/usr/licensed/anaconda3/5.2.0/bin/python3.6"
  export DEFAULT_NUM_GPUS_PER_WORKER="0"
  export DEFAULT_NUM_GPUS_PER_NODE="0"
  export DEFAULT_NUM_CPUS_PER_NODE="16"
  export DEFAULT_MEMORY_PER_NODE="60G"
  # No GPUs on this cluster
  export BYPASS_GPU_TEST="true"
  export MPI_HOME="/home/andrewor/lib/openmpi"
elif [[ "$ENVIRONMENT" = "snsgpu" ]]; then
  export LOG_DIR="/home/andrew/Documents/dev/logs"
  export MODELS_DIR="/home/andrew/Documents/dev/models"
  export IMAGENET_DATA_DIR="" # TODO: fill this in
  export CIFAR10_DATA_DIR="/home/andrew/Documents/dev/dataset/cifar10/cifar-10-batches-bin"
  export BASE_TRAIN_DIR="/home/andrew/Documents/dev/train_data"
  export BASE_EVAL_DIR="/home/andrew/Documents/dev/eval_data"
  export TF_PKG="/home/andrew/Documents/dev/tensorflow_pkg/tensorflow-1.12.0rc1-cp36-cp36m-linux_x86_64.whl"
  export PYTHON_COMMAND="python3"
  export DEFAULT_NUM_GPUS_PER_WORKER="2"
  # No slurm on this machine, not used
  export DEFAULT_NUM_GPUS_PER_NODE=""
  export DEFAULT_NUM_CPUS_PER_NODE=""
  export DEFAULT_MEMORY_PER_NODE=""
else
  echo "ERROR: Unknown environment '$ENVIRONMENT'"
  exit 1
fi

# Helper function to get a human readable timestamp
function get_submit_timestamp() {
  date +%m_%d_%y_%s%3N
}

