#!/bin/bash

export NUM_TRIALS="3"
export BATCH_SIZE="128"
export NUM_EPOCHS="200"
export DATASET="cifar10"

run_it() {
  export RUN_TAG="${NUM_GPUS}gpu_${NUM_VIRTUAL_NODES_PER_DEVICE}vn_${BATCH_SIZE}bs"
  for i in `seq 1 $NUM_TRIALS`; do
    echo "Running ${RUN_TAG}, iteration ${i}/${NUM_TRIALS}"
    ./run_resnet.sh;
  done
}

export CUDA_VISIBLE_DEVICES="0"
export NUM_GPUS="1"
export NUM_VIRTUAL_NODES_PER_DEVICE="4"
run_it

export NUM_TRIALS="2"
export CUDA_VISIBLE_DEVICES="0,1"
export NUM_GPUS="2"
export NUM_VIRTUAL_NODES_PER_DEVICE="2"
run_it

export CUDA_VISIBLE_DEVICES="0"
export NUM_GPUS="1"
export NUM_VIRTUAL_NODES_PER_DEVICE="2"
run_it

export CUDA_VISIBLE_DEVICES="0,1"
export NUM_GPUS="2"
export NUM_VIRTUAL_NODES_PER_DEVICE="1"
run_it

