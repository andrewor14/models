#!/bin/bash

# ============================================================
#  Entry point for submitting a job through MPI
#
#  The caller should set the following environment variables:
#  LAUNCH_SCRIPT_NAME, NUM_NODES, NUM_WORKERS,
#  and NUM_PARAMETER_SERVERS
# ============================================================

# Set common configs
source common_configs.sh

# Run configs
export RUN_PATH="$MODELS_DIR/deploy/run_with_env.sh"
export LAUNCH_SCRIPT_NAME="${LAUNCH_SCRIPT_NAME:=run_tensorflow.sh}"
if [[ -z "$JOB_NAME" ]]; then
  SUBMIT_TIMESTAMP="$(get_submit_timestamp)"
  RUN_TAG="${RUN_TAG:=models}"
  export JOB_NAME="${RUN_TAG}-${SUBMIT_TIMESTAMP}"
fi

# Assign GPUs based on CUDA_VISIBLE_DEVICES and how many workers are sharing one node.
# For example, if CUDA_VISIBLE_DEVICES = 0,1,2,3,4,5,6,7 and NUM_WORKERS_PER_NODE = 2,
# then NUM_GPUS_PER_WORKER = 4. If the number of devices does not divide, throw an error.
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  export NUM_GPUS_PER_NODE="$(echo "$CUDA_VISIBLE_DEVICES" | sed 's/,/\n/g' | wc -l)"
else
  export NUM_GPUS_PER_NODE="0"
fi
# Horovod expects one GPU per worker
if [[ "$USE_HOROVOD" == "true" ]] && [[ "$NUM_GPUS_PER_NODE" != "0" ]]; then
  export NUM_WORKERS_PER_NODE="$NUM_GPUS_PER_NODE"
  export NUM_GPUS_PER_WORKER="1"
else
  export NUM_WORKERS_PER_NODE="${NUM_WORKERS_PER_NODE:=$DEFAULT_NUM_WORKERS_PER_NODE}"
  export NUM_GPUS_PER_WORKER="$((NUM_GPUS_PER_NODE / NUM_WORKERS_PER_NODE))"
fi
# Check if all GPUs were assigned
if [[ "$((NUM_GPUS_PER_WORKER * NUM_WORKERS_PER_NODE))" != "$NUM_GPUS_PER_NODE" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES ($CUDA_VISIBLE_DEVICES) did not divide cleanly"\
    "among $NUM_WORKERS_PER_NODE workers"
  exit 1
fi

# Set NUM_WORKERS, NUM_PARAMETER_SERVERS and NUM_NODES based on each other while
# preserving NUM_WORKERS + NUM_PARAMETER_SERVERS = NUM_NODES
# If NUM_NODES is missing, either fill it in with the other variables,
# or default if we're missing information
if [[ -z "$NUM_NODES" ]]; then
  if [[ -n "$NUM_WORKERS" ]] && [[ -n "$NUM_PARAMETER_SERVERS" ]]; then
    NUM_NODES="$((NUM_WORKERS+NUM_PARAMETER_SERVERS))"
  else
    NUM_NODES="$DEFAULT_NUM_NODES"
  fi
fi
# At this point, NUM_NODES is set, so we just fill in the rest
if [[ -z "$NUM_WORKERS" ]]; then
  NUM_PARAMETER_SERVERS="${NUM_PARAMETER_SERVERS:=$DEFAULT_NUM_PARAMETER_SERVERS}"
  NUM_WORKERS="$((NUM_NODES-NUM_PARAMETER_SERVERS))"
elif [[ -z "$NUM_PARAMETER_SERVERS" ]]; then
  NUM_PARAMETER_SERVERS="$((NUM_NODES-NUM_WORKERS))"
fi
# Check that things add up
if [[ "$((NUM_WORKERS+NUM_PARAMETER_SERVERS))" != "$NUM_NODES" ]]; then
  echo "ERROR: NUM_WORKERS ($NUM_WORKERS) + NUM_PARAMETER_SERVERS"\
         "($NUM_PARAMETER_SERVERS) != NUM_NODES ($NUM_NODES)"
  exit 1
fi

# Export for downstream scripts
export NUM_WORKERS
export NUM_PARAMETER_SERVERS

# In tigerpu cluster, make sure we're actually running through MPI
if [[ "$ENVIRONMENT" = "tigergpu" ]]; then
  module load openmpi/gcc/3.0.0/64
fi

./deploy_mpi.sh

