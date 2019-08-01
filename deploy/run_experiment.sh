#!/bin/bash

# ==================================================
#  A script for running experiments in the cluster
#
#  An experiment consists of multiple runs of a job
#  across different settings.
# ==================================================

# Set common configs
source common_configs.sh

# Configure these, see run_distributed.sh for an explanation for MODE
# These default settings will output [4, 8, 12, ... 88, 92, 96]
export MODE="${MODE:=static}"
export DATASET="${DATASET:=cifar10}"
export NUM_GPUS_INCREMENT="${NUM_GPUS_INCREMENT:=4}"
export MIN_GPUS="${MIN_GPUS:=4}"
export MAX_GPUS="${MAX_GPUS:=96}"

# Set default batch size based on dataset
if [[ "$DATASET" == "cifar10" ]]; then
  export BATCH_SIZE="${BATCH_SIZE:=128}"
elif [[ "$DATASET" == "imagenet" ]]; then
  export BATCH_SIZE="${BATCH_SIZE:=8192}"
else
  echo "ERROR: Unknown dataset '$DATASET'"
  exit 1
fi

# Don't touch these
export USE_KERAS="true"
export USE_HOROVOD="true"
export NUM_PARAMETER_SERVERS="0"
export MPI_SILENCE_OUTPUT="true"

# Run the experiment
export NUM_WORKERS_LIST=`seq $MIN_GPUS $NUM_GPUS_INCREMENT $MAX_GPUS`

echo "==========================================================="
echo " Running experiment '$MODE'"
echo "   dataset = $DATASET"
echo "   batch size = $BATCH_SIZE"
echo "   num workers = "$NUM_WORKERS_LIST
echo "==========================================================="

for NUM_WORKERS in $NUM_WORKERS_LIST; do
  echo " * Running with $NUM_WORKERS workers"
  export NUM_WORKERS
  ./run_distributed.sh
done

