#!/bin/bash

# Set common configs
source common_configs.sh

# Set run configs
export SCRIPT_NAME="$MODELS_DIR/official/resnet/test_mpi/test_mpi.sh"
export RUN_TAG="test_mpi"
export USE_HOROVOD="true"

# If we're just trying to attach to an existing cluster, just launch 1 worker
if [[ -n "$AUTOSCALING_MASTER_HOST_PORT" ]]; then
  export NUM_NODES=1
  export RUN_TAG="$RUN_TAG-added"
else
  export NUM_NODES=4
fi

./run_slurm.sh

