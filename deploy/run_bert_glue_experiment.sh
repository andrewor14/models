#!/bin/bash

if [[ -z "$EXPERIMENT_MODE" ]]; then
  echo "Please set EXPERIMENT_MODE to one of 'local', 'distributed', 'both', or 'try':"
  echo "  local: run experiments across 1, 2, 4 and 8 GPUs on a single node"
  echo "  distributed: run experiments across 8 GPUs on two machines"
  echo "  both: for each glue task, run 'local' first, then 'distributed'"
  echo "  try: run a few steps locally on 2 GPUs, just to try things out"
  exit 1
fi

export MODEL="bert"
export BERT_TASK="glue"
export MPI_VERBOSE="false"
export ENABLE_XLA="false"
export ENABLE_MONITOR_MEMORY="true"
export BATCH_SIZE="${BATCH_SIZE:=128}"
export PRETRAINED_DATA_DIR="${PRETRAINED_DATA_DIR:=/root/dev/dataset/bert/uncased_L-24_H-1024_A-16}"

# Number of examples that can be processed on the GPU at a given time
# This should be 2 for 2080 Ti and 4 for V100
export VIRTUAL_NODE_SIZE="${VIRTUAL_NODE_SIZE:=4}"

function run_it() {
  export GLUE_TASK="$1"
  if [[ "$EXPERIMENT_MODE" == "local" ]] || [[ "$EXPERIMENT_MODE" == "both" ]] || [[ "$EXPERIMENT_MODE" == "try" ]]; then
    if [[ "$EXPERIMENT_MODE" == "try" ]]; then
      num_gpus_list="2"
    else
      num_gpus_list="1 2 4 8"
    fi
    for num_gpus in $num_gpus_list; do
      export NUM_NODES="1"
      export NUM_GPUS="$num_gpus"
      export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 "$((num_gpus - 1))")"
      export NUM_VIRTUAL_NODES_PER_DEVICE="$((BATCH_SIZE / NUM_GPUS / VIRTUAL_NODE_SIZE))"
      export RUN_TAG="${GLUE_TASK}_${BATCH_SIZE}bs_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
      echo "Running experiment $RUN_TAG"
      ./run_distributed.sh
    done
  fi
  if [[ "$EXPEIRMENT_MODE" == "distributed" ]] || [[ "$EXPERIMENT_MODE" == "both" ]]; then
    if [[ "$BATCH_SIZE" == "128" ]]; then
      export NUM_NODES="2"
      export NUM_GPUS="8"
      export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
      export NUM_VIRTUAL_NODES_PER_DEVICE="1"
      export RUN_TAG="${GLUE_TASK}_${BATCH_SIZE}bs_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
      echo "Running experiment $RUN_TAG"
      ./run_distributed.sh
    else
      echo "Warning: skipping distributed mode for $GLUE_TASK because batch size was not 128 (was $BATCH_SIZE)"
    fi
  fi
}

# Optionally try running for only a few steps
if [[ "$EXPERIMENT_MODE" == "try" ]]; then
  unset NUM_EPOCHS
  export NUM_STEPS="10"
  export MPI_VERBOSE="true"
  export GLUE_TASK="${GLUE_TASK:=SST-2}"
  run_it "$GLUE_TASK"
  exit 0
fi

# SST-2
unset NUM_EPOCHS
export NUM_STEPS="1000"
run_it "SST-2"

# MNLI
unset NUM_EPOCHS
export NUM_STEPS="1000"
run_it "MNLI"

# CoLA
unset NUM_STEPS
export NUM_EPOCHS="3"
run_it "CoLA"

