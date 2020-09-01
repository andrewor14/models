#!/bin/bash

if [[ -z "$EXPERIMENT_MODE" ]]; then
  echo "Please set EXPERIMENT_MODE to one of 'local', '2nodes', '4nodes', 'all', or 'try':"
  echo "  local: run experiments across 1, 2, 4 and 8 GPUs on a single node"
  echo "  2nodes: run experiments across 16 GPUs on two nodes"
  echo "  4nodes: run experiments across 32 GPUs on four nodes"
  echo "  all: for each glue task, run 'local' first, then '2nodes', then '4nodes'"
  echo "  try: run a few steps locally on 8 GPUs, just to try things out"
  exit 1
fi

if [[ -n "$VIRTUAL_NODE_SIZE" ]]; then
  echo "Setting VIRTUAL_NODE_SIZE is not allowed"
  exit 1
fi

if [[ -n "$BATCH_SIZE" ]]; then
  echo "Setting BATCH_SIZE is not allowed"
  exit 1
fi

export MODEL="resnet"
export DATASET="imagenet"
export ENABLE_XLA="true"
export ENABLE_MONITOR_MEMORY="true"
export MPI_VERBOSE="false"
export MAX_NUM_GPUS="32"
export VIRTUAL_NODE_SIZE="256"
export BATCH_SIZE="8192"

# Helper function to run an experiment across a different number of GPUs
function run_it() {
  if [[ "$EXPERIMENT_MODE" == "try" ]]; then
    num_gpus_list="${NUM_GPUS:-8}"
  elif [[ "$EXPERIMENT_MODE" == "local" ]]; then
    num_gpus_list="${NUM_GPUS:-1 2 4 8}"
  elif [[ "$EXPERIMENT_MODE" == "2nodes" ]]; then
    num_gpus_list="16"
  elif [[ "$EXPERIMENT_MODE" == "4nodes" ]]; then
    num_gpus_list="32"
  elif [[ "$EXPERIMENT_MODE" == "all" ]]; then
    num_gpus_list="1 2 4 8 16 32"
  else
    echo "Error: unknown EXPERIMENT_MODE '$EXPERIMENT_MODE'"
    exit 1
  fi
  for num_gpus in $num_gpus_list; do
    if [[ "$num_gpus" == "32" ]]; then
      export NUM_NODES="4"
      export NUM_GPUS="8"
    elif [[ "$num_gpus" == "16" ]]; then
      export NUM_NODES="2"
      export NUM_GPUS="8"
    else
      export NUM_NODES="1"
      export NUM_GPUS="$num_gpus"
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 "$((num_gpus - 1))")"
    export NUM_VIRTUAL_NODES_PER_DEVICE="$((BATCH_SIZE / NUM_GPUS / NUM_NODES / VIRTUAL_NODE_SIZE))"
    export RUN_TAG="${BATCH_SIZE}bs_$((NUM_GPUS * NUM_NODES))gpu_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
    echo "Running experiment $RUN_TAG"
    ./run_distributed.sh
  done
}

# Optionally try running for only a few steps
if [[ "$EXPERIMENT_MODE" == "try" ]]; then
  unset NUM_EPOCHS
  export NUM_STEPS="10"
  export ENABLE_XLA="false"
  export MPI_VERBOSE="true"
else
  unset NUM_STEPS
  export NUM_EPOCHS="90"
fi

run_it

