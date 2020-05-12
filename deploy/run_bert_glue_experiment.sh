#!/bin/bash

export NUM_TRIALS="5"
export BATCH_SIZE="64"

run_it() {
  export RUN_TAG="${NUM_VIRTUAL_NODES_PER_DEVICE}vn_${BATCH_SIZE}bs"
  for i in `seq 1 $NUM_TRIALS`; do
    echo "Running ${RUN_TAG}, iteration ${i}/${NUM_TRIALS}"
    ./run_bert_glue.sh;
  done
}

export CUDA_VISIBLE_DEVICES="0,1"
export NUM_VIRTUAL_NODES_PER_DEVICE="1"
run_it

export CUDA_VISIBLE_DEVICES="0"
export NUM_VIRTUAL_NODES_PER_DEVICE="2"
run_it

