#!/bin/bash

# ============================================================
#  Entry point for submitting a job through slurm or MPI
#
#  The caller should set the following environment variables:
#  NUM_CPUS_PER_NODE, MEMORY_PER_NODE, TIME_LIMIT_HOURS,
#  LAUNCH_SCRIPT_NAME, NUM_NODES, NUM_WORKERS,
#  and NUM_PARAMETER_SERVERS
# ============================================================

# Set common configs
source common_configs.sh

# Run configs
RUN_PATH="$MODELS_DIR/deploy/run_with_env.sh"
export LAUNCH_SCRIPT_NAME="${LAUNCH_SCRIPT_NAME:=run_cifar10.sh}"
if [[ -z "$JOB_NAME" ]]; then
  SUBMIT_TIMESTAMP="$(get_submit_timestamp)"
  RUN_TAG="${RUN_TAG:=models}"
  export JOB_NAME="${RUN_TAG}-${SUBMIT_TIMESTAMP}"
fi

# Slurm specific configs
NUM_TASKS_PER_NODE="1"
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:=$DEFAULT_NUM_CPUS_PER_NODE}"
MEMORY_PER_NODE="${MEMORY_PER_NODE:=$DEFAULT_MEMORY_PER_NODE}"
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:=144}"

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

# For normal operations (not using MPI), just run standard `sbatch` with `srun`
if [[ "$USE_HOROVOD" != "true" ]]; then
  sbatch\
    --nodes="$NUM_NODES"\
    --ntasks="$NUM_NODES"\
    --ntasks-per-node="$NUM_TASKS_PER_NODE"\
    --cpus-per-task="$NUM_CPUS_PER_NODE"\
    --mem="$MEMORY_PER_NODE"\
    --gres="gpu:$NUM_GPUS_PER_NODE"\
    --time="$TIME_LIMIT_HOURS:00:00"\
    --job-name="$JOB_NAME"\
    --mail-type="begin"\
    --mail-type="end"\
    --mail-user="$EMAIL"\
    --wrap "srun --output=$LOG_DIR/$JOB_NAME-%n.out $RUN_PATH $LAUNCH_SCRIPT_NAME"
else
  # Otherwise, we're running horovod, so we use `mpirun`.
  # Note: slurm has an API for dynamically expanding a job's allocation, but it is often
  # not accessible due to permission issues. Therefore, we avoid using slurm here at all.

  # Tell MPI which hosts to use
  HOST_FILE="${HOST_FILE:=hosts.txt}"
  if [[ -f "$HOST_FILE" ]]; then
    if [[ -n "$(grep 'slots' $HOST_FILE)" ]]; then
      # User already specified slots in the host file, so we just use those slot assignments
      HOST_FLAG="--hostfile $HOST_FILE"
    else
      # Otherwise, assign the default number of slots per host to each host
      HOSTS="$(cat hosts.txt | sed s/$/:$NUM_WORKERS_PER_NODE/g | tr '\n' ',' | sed 's/,$/\n/')"
      HOST_FLAG="--host $HOSTS"
    fi
  elif [[ -n "$(command -v sinfo)" ]]; then
    # If there is no host file then try to get the hosts from slurm
    SLURM_HOSTS="$(sinfo -N --state=idle | tail -n +2 | awk '{print $1}')"
    HOSTS="$(echo "$SLURM_HOSTS" | sed s/$/:$NUM_WORKERS_PER_NODE/g | tr '\n' ',' | sed 's/,$/\n/')"
    HOST_FLAG="--host $HOSTS"
  else
    # Otherwise, assume we're running single node
    HOST_FLAG="--host localhost:$NUM_WORKERS_PER_NODE"
  fi

  # Pass all environment variables to mpirun, with some exceptions
  # The format expected by MPI is "-x ENV_VAR1 -x ENV_VAR2 ..."
  ALL_ENV_VARS="$(printenv | grep "=" | awk -F "=" '{print $1}')"
  ALL_ENV_VARS="$(echo "$ALL_ENV_VARS" | grep -v "BASH\|SSH\|HOSTNAME\|TERMCAP\|_$\|^\s")"
  ENV_FLAG="-x $(echo "$ALL_ENV_VARS" | tr '\n' ',' | sed 's/,$/\n/g' | sed 's/,/ \-x /g')"

  # Horovod flags: see https://github.com/horovod/horovod/blob/master/docs/mpirun.rst
  ENV_FLAG="$ENV_FLAG -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^lo,docker0"
  HOROVOD_FLAGS="-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 "
  HOROVOD_FLAGS="$HOROVOD_FLAGS --bind-to none --map-by slot "

  # TODO: silence this call; it's very noisy
  # Note: setting --bind-to to "none" (default was "core") significantly improves MPI performance
  # for multi-threaded applications. See https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php
  mpirun\
    $ENV_FLAG\
    $HOST_FLAG\
    $HOROVOD_FLAGS\
    --allow-run-as-root\
    --nooversubscribe\
    --np "$NUM_NODES"\
    --output-filename "$LOG_DIR/$JOB_NAME"\
    "$RUN_PATH" "$LAUNCH_SCRIPT_NAME"
fi

