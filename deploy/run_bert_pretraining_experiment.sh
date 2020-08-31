#!/bin/bash

export MODEL="bert"
export BERT_TASK="pretraining"
export NUM_NODES="1"
export MPI_VERBOSE="false"
export ENABLE_XLA="false"
export ENABLE_MONITOR_MEMORY="true"
export BATCH_SIZE="${BATCH_SIZE:=32}"

# Number of examples that can be processed on the GPU at a given time
# This should be 2 for 2080 Ti and 4 for V100
export VIRTUAL_NODE_SIZE="${VIRTUAL_NODE_SIZE:=4}"

if [[ "$EXPERIMENT_MODE" == "try" ]]; then
  num_gpus_list="2"
  export NUM_STEPS="10"
  export MPI_VERBOSE="true"
else
  num_gpus_list="1 2 4 8"
  export NUM_STEPS="1000"
fi

for num_gpus in $num_gpus_list; do
  export NUM_GPUS="$num_gpus"
  export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 "$((num_gpus - 1))")"
  export NUM_VIRTUAL_NODES_PER_DEVICE="$((BATCH_SIZE / NUM_GPUS / VIRTUAL_NODE_SIZE))"
  export RUN_TAG="${BATCH_SIZE}bs_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
  echo "Running experiment $RUN_TAG"
  ./run_distributed.sh
done

