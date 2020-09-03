#!/bin/bash

export MODEL="transformer"
export NUM_NODES="1"
export ENABLE_XLA="true"
export ENABLE_MONITOR_MEMORY="false"
export MPI_VERBOSE="false"
export MAX_NUM_GPUS="8"
export BASELINE="${BASELINE:=false}"

# Number of examples that can be processed on the GPU at a given time
# This should be 2048 for 2080 Ti and 4096 for V100
export VIRTUAL_NODE_SIZE="${VIRTUAL_NODE_SIZE:=4096}"

# Set the batch size such that the number of virtual nodes on the max GPU configuration is 1
DEFAULT_BATCH_SIZE="$((MAX_NUM_GPUS * VIRTUAL_NODE_SIZE))"
export BATCH_SIZE="${BATCH_SIZE:=$DEFAULT_BATCH_SIZE}"

if [[ "$EXPERIMENT_MODE" == "try" ]]; then
  num_gpus_list="${NUM_GPUS:-8}"
  export NUM_STEPS="10"
  export SKIP_EVAL="true"
  export ENABLE_XLA="false"
  export MPI_VERBOSE="true"
else
  num_gpus_list="${NUM_GPUS:-1 2 4 8}"
  export NUM_STEPS="100000"
  export STEPS_BETWEEN_EVALS="$((NUM_STEPS / 20))"
fi

for num_gpus in $num_gpus_list; do
  export NUM_GPUS="$num_gpus"
  export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 "$((num_gpus - 1))")"
  if [[ "$BASELINE" == "true" ]]; then
    export NUM_VIRTUAL_NODES_PER_DEVICE="1"
    export BATCH_SIZE="$((VIRTUAL_NODE_SIZE * NUM_GPUS * NUM_NODES))"
  else
    export NUM_VIRTUAL_NODES_PER_DEVICE="$((BATCH_SIZE / NUM_GPUS / VIRTUAL_NODE_SIZE))"
  fi
  export RUN_TAG="${BATCH_SIZE}bs_${NUM_GPUS}gpu_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
  echo "Running experiment $RUN_TAG"
  ./run_distributed.sh
done

