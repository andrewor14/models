#!/bin/bash

# ============================================================
#  Entry point for submitting a job through slurm
#
#  The caller should set the following environment variables:
#  NUM_CPUS_PER_NODE, NUM_GPUS_PER_NODE, MEMORY_PER_NODE,
#  TIME_LIMIT_HOURS, SCRIPT_NAME, NUM_NODES, NUM_WORKERS,
#  and NUM_PARAMETER_SERVERS
# ============================================================

# Set common configs
source common_configs.sh

# Run configs
RUN_PATH="$MODELS_DIR/slurm/run_with_env.sh"
SCRIPT_NAME="${SCRIPT_NAME:=run_cifar10.sh}"
if [[ -z "$JOB_NAME" ]]; then
  SUBMIT_TIMESTAMP="$(get_submit_timestamp)"
  RUN_TAG="${RUN_TAG:=models}"
  export JOB_NAME="${RUN_TAG}-${SUBMIT_TIMESTAMP}"
fi

# Slurm specific configs
# Note: we always launch 1 task per node; in multiplex mode, this task
# launches multiple python processes, but there is still only one task.
NUM_TASKS_PER_NODE="1"
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:=$DEFAULT_NUM_CPUS_PER_NODE}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:=$DEFAULT_NUM_GPUS_PER_NODE}"
MEMORY_PER_NODE="${MEMORY_PER_NODE:=$DEFAULT_MEMORY_PER_NODE}"
TIME_LIMIT_HOURS="${TIME_LIMIT_HOURS:=144}"

# In non-multiplex mode, NUM_GPUS_PER_NODE and NUM_GPUS_PER_WORKER should be the same.
if [[ "$SCRIPT_NAME" == "run_cifar10.sh" ]] &&\
    [[ -n "$NUM_GPUS_PER_WORKER" ]] &&\
    [[ "$NUM_GPUS_PER_NODE" != "$NUM_GPUS_PER_WORKER" ]]; then
  echo "ERROR: In non-multiplex mode, NUM_GPUS_PER_WORKER ($NUM_GPUS_PER_WORKER)"\
    "must match NUM_GPUS_PER_NODE ($NUM_GPUS_PER_NODE)."
  exit 1
fi

# Set NUM_WORKERS, NUM_PARAMETER_SERVERS and NUM_NODES
# In non-multiplex mode, set these variables based on each other while
# preserving NUM_WORKERS + NUM_PARAMETER_SERVERS = NUM_NODES
if [[ "$SCRIPT_NAME" == "run_cifar10.sh" ]]; then
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
# In multiplex mode, NUM_NODES should always be 1, while NUM_WORKERS is
# expected to be provided by the caller
elif [[ "$SCRIPT_NAME" == "run_multiplex.sh" ]]; then
  if [[ -n "$NUM_NODES" ]] && [[ "$NUM_NODES" != "1" ]]; then
    echo "ERROR: NUM_NODES must be 1 in multiplex mode."
    exit 1
  fi
  NUM_NODES=1
  NUM_PARAMETER_SERVERS="${NUM_PARAMETER_SERVERS:=$DEFAULT_NUM_PARAMETER_SERVERS}"
else
  echo "ERROR: Unknown script $SCRIPT_NAME"
  exit 1
fi

# Export for downstream scripts
export NUM_WORKERS
export NUM_PARAMETER_SERVERS

# In tigerpu cluster, make sure we're actually running through MPI
if [[ "$ENVIRONMENT" = "tigergpu" ]]; then
  module load openmpi/gcc/3.0.0/64
fi

# Set run command, either 'srun' or 'mpirun'
if [[ "$USE_HOROVOD" == "true" ]]; then
  RUN_COMMAND="mpirun --output-filename $LOG_DIR/$JOB_NAME $RUN_PATH $SCRIPT_NAME"
else
  RUN_COMMAND="srun --output=$LOG_DIR/$JOB_NAME-%n.out $RUN_PATH $SCRIPT_NAME"
fi

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
  --wrap "$RUN_COMMAND"

