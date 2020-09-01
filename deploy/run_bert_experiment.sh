#!/bin/bash

if [[ -z "$BERT_TASK" ]]; then
  echo "Please set BERT_TASK to one of 'glue' or 'pretraining'"
  exit 1
fi

if [[ -z "$EXPERIMENT_MODE" ]]; then
  echo "Please set EXPERIMENT_MODE to one of 'local', 'distributed', 'both', or 'try':"
  echo "  local: run experiments across 1, 2, 4 and 8 GPUs on a single node"
  echo "  distributed: run experiments across 16 GPUs on two machines"
  echo "  both: for each glue task, run 'local' first, then 'distributed'"
  echo "  try: run a few steps locally on 8 GPUs, just to try things out"
  exit 1
fi

export MODEL="bert"
export ENABLE_XLA="true"
export ENABLE_MONITOR_MEMORY="true"
export MPI_VERBOSE="false"
export MAX_NUM_GPUS="16"
export PRETRAINED_DATA_DIR="${PRETRAINED_DATA_DIR:=/root/dev/dataset/bert/uncased_L-24_H-1024_A-16}"
export BASELINE="${BASELINE:=false}"

# Number of examples that can be processed on the GPU at a given time
export VIRTUAL_NODE_SIZE="${VIRTUAL_NODE_SIZE:=2}"

# Set the batch size such that the number of virtual nodes on the max GPU configuration is 1
DEFAULT_BATCH_SIZE="$((MAX_NUM_GPUS * VIRTUAL_NODE_SIZE))"
export BATCH_SIZE="${BATCH_SIZE:=$DEFAULT_BATCH_SIZE}"

# Helper function to run an experiment across a different number of GPUs
function run_it() {
  if [[ "$EXPERIMENT_MODE" == "try" ]]; then
    num_gpus_list="${NUM_GPUS:-8}"
  elif [[ "$EXPERIMENT_MODE" == "local" ]]; then
    num_gpus_list="${NUM_GPUS:-1 2 4 8}"
  elif [[ "$EXPERIMENT_MODE" == "distributed" ]]; then
    num_gpus_list="16"
  elif [[ "$EXPERIMENT_MODE" == "both" ]]; then
    num_gpus_list="1 2 4 8 16"
  else
    echo "Error: unknown EXPERIMENT_MODE '$EXPERIMENT_MODE'"
    exit 1
  fi
  for num_gpus in $num_gpus_list; do
    if [[ "$num_gpus" == "16" ]]; then
      export NUM_NODES="2"
      export NUM_GPUS="8"
    else
      export NUM_NODES="1"
      export NUM_GPUS="$num_gpus"
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 "$((num_gpus - 1))")"
    if [[ "$BASELINE" == "true" ]]; then
      export NUM_VIRTUAL_NODES_PER_DEVICE="1"
      export BATCH_SIZE="$((VIRTUAL_NODE_SIZE * NUM_GPUS * NUM_NODES))"
    else
      export NUM_VIRTUAL_NODES_PER_DEVICE="$((BATCH_SIZE / NUM_GPUS / NUM_NODES / VIRTUAL_NODE_SIZE))"
    fi
    # Set run tag
    export RUN_TAG="${BATCH_SIZE}bs_$((NUM_GPUS * NUM_NODES))gpu_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
    if [[ "$BERT_TASK" == "glue" ]]; then
      export RUN_TAG="${GLUE_TASK}_${RUN_TAG}"
    fi
    echo "Running $BERT_TASK experiment $RUN_TAG"
    ./run_distributed.sh
  done
}

# Optionally try running for only a few steps
if [[ "$EXPERIMENT_MODE" == "try" ]]; then
  unset NUM_EPOCHS
  export NUM_STEPS="10"
  export ENABLE_XLA="false"
  export MPI_VERBOSE="true"
  if [[ "$BERT_TASK" == "glue" ]]; then
    export GLUE_TASK="${GLUE_TASK:=SST-2}"
  fi
  run_it
  exit 0
fi

# Run the experiment
if [[ "$BERT_TASK" == "glue" ]]; then
  export NUM_EPOCHS="50"
  export NUM_STEPS="100"
  glue_task_list="${GLUE_TASK_LIST:=SST-2 QNLI CoLA}"
  for glue_task in $glue_task_list; do
    export GLUE_TASK="$glue_task"
    run_it
  done
elif [[ "$BERT_TASK" == "pretraining" ]]; then
  unset NUM_EPOCHS
  export NUM_STEPS="1000"
  run_it
else
  echo "Error: unknown BERT_TASK '$BERT_TASK'"
  exit 1
fi

