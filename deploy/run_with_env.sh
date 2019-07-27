#!/bin/bash

# ========================================================
#  A wrapper around a script that sets up the environment
# ========================================================

if [[ "$#" -lt 1 ]]; then
  echo "Usage: run_with_env.sh <script_name> [<arg1>] [<arg2>] ..."
  exit 1
fi

# Set common configs
source common_configs.sh

export PYTHONPATH="$PYTHONPATH:$MODELS_DIR"

if [[ "$ENVIRONMENT" = "tigergpu" ]]; then
  MY_MPI_HOME="$MPI_HOME"
  module load anaconda3/5.2.0
  module load cudnn/cuda-9.2/7.3.1
  module load openmpi/gcc/3.0.0/64
  # Use our custom Open MPI library, which just points to the one we just loaded
  export MPI_HOME="$MY_MPI_HOME"
fi

# If we're not using GPUs, don't do the GPU test
NUM_GPUS_PER_WORKER="${NUM_GPUS_PER_WORKER:=$DEFAULT_NUM_GPUS_PER_WORKER}"
if [[ "$NUM_GPUS_PER_WORKER" != 0 ]] && [[ "$BYPASS_GPU_TEST" != "true" ]]; then
  if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "Must specify CUDA_VISIBLE_DEVICES if NUM_GPUS_PER_WORKER > 0"
    exit 1
  fi
  "$PYTHON_COMMAND" test_gpu_support.py
  if [[ "$?" -ne 0 ]]; then
    echo "GPU test failed. Exiting."
    exit 1
  fi
fi

bash "$@"

