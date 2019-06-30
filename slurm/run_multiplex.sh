#!/bin/bash

# =============================================================
#  A script to run multiple processes on a single node
#
#  This expects the following environment variables to be set:
#  JOB_NAME, NUM_WORKERS, and NUM_PARAMETER_SERVERS
# =============================================================

# Set common configs
source common_configs.sh

if [[ -z "$JOB_NAME" ]]; then
  echo "ERROR: JOB_NAME must be set."
  exit 1
fi

if [[ -z "$NUM_WORKERS" ]]; then
  echo "ERROR: NUM_WORKERS must be set."
  exit 1
fi

if [[ -z "$NUM_PARAMETER_SERVERS" ]]; then
  echo "ERROR: NUM_PARAMETER_SERVERS must be set."
  exit 1
fi

NUM_GPUS_PER_WORKER="${NUM_GPUS_PER_WORKER:=$DEFAULT_NUM_GPUS_PER_WORKER}"
CUDA_VISIBLE_DEVICES_PER_WORKER=()
ORIGINAL_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  # Decide which GPUs each worker gets, e.g. if CUDA_VISIBLE_DEVICES is "0,1,2,3" and
  # NUM_GPUS_PER_WORKER is 2, then CUDA_VISIBLE_DEVICES_PER_WORKER will be ("0,1", "2,3").
  # In this case, NUM_WORKERS must be 2 or the program will fail.
  i=1
  current_device_string=""
  for device in $(echo "$CUDA_VISIBLE_DEVICES" | sed "s/,/ /g"); do
    # Add the current device to current_device_string
    if [[ -z "$current_device_string" ]]; then
      current_device_string="$device"
    else
      current_device_string="$current_device_string,$device"
    fi
    # Collect and reset current_device_string
    if [[ "$((i % $NUM_GPUS_PER_WORKER))" = "0" ]]; then
      CUDA_VISIBLE_DEVICES_PER_WORKER+=("$current_device_string")
      current_device_string=""
    fi
    i="$((i+1))"
  done
  # Make sure we have the right number of workers
  # It's OK if we don't end up using all the GPUs on the machine!
  # TODO: also make sure each worker has the same number of GPUs
  if [[ "${#CUDA_VISIBLE_DEVICES_PER_WORKER[*]}" < "$NUM_WORKERS" ]]; then
    echo "ERROR: GPUs do not split evenly among workers:"
    echo "  NUM_WORKERS: $NUM_WORKERS"
    echo "  NUM_GPUS_PER_WORKER: $NUM_GPUS_PER_WORKER"
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "  CUDA_VISIBLE_DEVICES_PER_WORKER: ${CUDA_VISIBLE_DEVICES_PER_WORKER[*]}"
    exit 1
  fi
  echo "CUDA_VISIBLE_DEVICES_PER_WORKER: ${CUDA_VISIBLE_DEVICES_PER_WORKER[*]}"
fi

# For running NOT through slurm
if [[ -z "$SLURMD_NODENAME" ]]; then
  echo "SLURM mode not detected: Running locally!"
  export SLURM_JOB_NODELIST="localhost"
  export SLURM_JOB_NODENAME="localhost"
  export SLURM_JOB_NUM_NODES=1
  export SLURMD_NODENAME="localhost"
  export SLURM_JOB_ID="local"
  export SLURM_JOB_NAME="$JOB_NAME"
fi

function start_it() {
  index="$1"
  # Don't give the parameter server GPUs
  if [[ "$index" < "$NUM_PARAMETER_SERVERS" ]] || [[ "$NUM_GPUS_PER_WORKER" == 0 ]]; then
    export CUDA_VISIBLE_DEVICES=""
  elif [[ -n "$CUDA_VISIBLE_DEVICES_PER_WORKER" ]]; then
    # Export the right set of devices for this worker
    worker_index="$((index-NUM_PARAMETER_SERVERS))"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PER_WORKER[$worker_index]}"
  else
    # In the non-GPU case, just pass on the original CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES="$ORIGINAL_CUDA_VISIBLE_DEVICES"
  fi
  LOG_FILE="$LOG_DIR/${SLURM_JOB_NAME}-${index}.out"
  # Start the process
  echo "Starting tensorflow process on $SLURMD_NODENAME ($index), writing to $LOG_FILE"
  export SLURMD_PROC_INDEX="$index"
  export SLURM_NODEID="$index"
  ./run_cifar10.sh > "$LOG_FILE" 2>&1 &
}

# Actually start everything
export SLURM_JOB_NUM_PROCS_PER_NODE="$((NUM_WORKERS+NUM_PARAMETER_SERVERS))"

if [[ "$FATE_SHARING" == "true" ]]; then
  # In parameter server mode, the parameter servers and some workers often do not exit.
  # As such, we need a way to terminate all children tensorflow processes after we think
  # the training/eval job is done. Here we fate share all such processes such that the
  # first one (worker) to visit signals the end of all other ones.
  wait -n
  # TODO: also make sure the processes we kill here are in fact our descendants
  ps aux | grep "$(whoami)" | grep run_cifar10 | awk '{print $2}' | xargs kill -9
else
  # Otherwise, wait for all children processes to finish
  wait
fi

