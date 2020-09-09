#!/bin/bash

if [[ -z "$BERT_TASK" ]]; then
  echo "Please set BERT_TASK to one of 'glue' or 'pretraining'"
  exit 1
fi

export MODEL="bert"
export NUM_NODES="1"
export ENABLE_XLA="true"
export ENABLE_MONITOR_MEMORY="true"
export MPI_VERBOSE="false"
export MAX_NUM_GPUS="8"
export USER_NUM_GPUS="$NUM_GPUS"
export BASELINE="${BASELINE:=false}"

# Number of examples that can be processed on the GPU at a given time
export VIRTUAL_NODE_SIZE="${VIRTUAL_NODE_SIZE:=8}"

# Set the batch size such that the number of virtual nodes on the max GPU configuration is 1
DEFAULT_BATCH_SIZE="$((MAX_NUM_GPUS * VIRTUAL_NODE_SIZE))"
export BATCH_SIZE="${BATCH_SIZE:=$DEFAULT_BATCH_SIZE}"

# Helper function to run an experiment across a different number of GPUs
function run_it() {
  if [[ "$EXPERIMENT_MODE" == "try" ]]; then
    num_gpus_list="${USER_NUM_GPUS:-8}"
    unset NUM_EPOCHS
    export NUM_STEPS="10"
    export SKIP_EVAL="true"
    export ENABLE_XLA="false"
    export MPI_VERBOSE="true"
  else
    num_gpus_list="${USER_NUM_GPUS:-1 2 4 8}"
  fi
  for num_gpus in $num_gpus_list; do
    export NUM_GPUS="$num_gpus"
    export CUDA_VISIBLE_DEVICES="$(seq -s ',' 0 "$((num_gpus - 1))")"
    if [[ "$BASELINE" == "true" ]]; then
      export NUM_VIRTUAL_NODES_PER_DEVICE="1"
      export BATCH_SIZE="$((VIRTUAL_NODE_SIZE * NUM_GPUS * NUM_NODES))"
    else
      export NUM_VIRTUAL_NODES_PER_DEVICE="$((BATCH_SIZE / NUM_GPUS / NUM_NODES / VIRTUAL_NODE_SIZE))"
    fi
    export NUM_STEPS="$((NUM_EXAMPLES_PER_EPOCH / BATCH_SIZE))"
    # Set run tag
    export RUN_TAG="${BATCH_SIZE}bs_$((NUM_GPUS * NUM_NODES))gpu_${NUM_VIRTUAL_NODES_PER_DEVICE}vn"
    if [[ "$BERT_TASK" == "glue" ]]; then
      export RUN_TAG="${GLUE_TASK}_${RUN_TAG}"
    fi
    echo "Running $BERT_TASK experiment $RUN_TAG for $NUM_EPOCHS epochs and $NUM_STEPS steps per eopch"
    ./run_distributed.sh
  done
}

# Run the experiment
if [[ "$BERT_TASK" == "glue" ]]; then
  export NUM_EPOCHS="20"
  if [[ "$EXPERIMENT_MODE" == "true" ]]; then
    export GLUE_TASK="${GLUE_TASK:=SST-2}"
    run_it
    exit 0
  fi

  ## 1/10 of QNLI
  #export NUM_EXAMPLES_PER_EPOCH="10474"
  #export GLUE_TASK="QNLI"
  #run_it

  ## 1/10 of SST-2
  #export NUM_EXAMPLES_PER_EPOCH="6735"
  #export GLUE_TASK="SST-2"
  #run_it

  # CoLA
  export NUM_EPOCHS="50"
  export NUM_EXAMPLES_PER_EPOCH="8551"
  export GLUE_TASK="CoLA"
  run_it

  # MRPC
  export NUM_EPOCHS="50"
  export NUM_EXAMPLES_PER_EPOCH="3668"
  export GLUE_TASK="MRPC"
  run_it

  # RTE
  export NUM_EPOCHS="50"
  export NUM_EXAMPLES_PER_EPOCH="2490"
  export GLUE_TASK="RTE"
  run_it

  ## 1/100 of MNLI
  #export NUM_EXAMPLES_PER_EPOCH="3927"
  #export GLUE_TASK="MNLI"
  #run_it

  # 1/10 of QQP
  export NUM_EPOCHS="10"
  export NUM_EXAMPLES_PER_EPOCH="36385"
  export GLUE_TASK="QQP"
  run_it

elif [[ "$BERT_TASK" == "pretraining" ]]; then
  unset NUM_EPOCHS
  export NUM_STEPS="1000"
  run_it
else
  echo "Error: unknown BERT_TASK '$BERT_TASK'"
  exit 1
fi

