#!/bin/bash

source common.sh

MODEL="${MODEL:=resnet}"
export NUM_NODES="${NUM_NODES:=4}"

# Set run script and job name based on task
if [[ "$MODEL" == "resnet" ]]; then
  RUN_SCRIPT="run_resnet.sh"
  DATASET="${DATASET:=imagenet}"
  BASE_JOB_NAME="resnet-${DATASET}"
elif [[ "$MODEL" == "bert" ]]; then
  BERT_TASK="${BERT_TASK:=glue}"
  RUN_SCRIPT="run_bert_${BERT_TASK}.sh"
  BASE_JOB_NAME="bert-${BERT_TASK}"
else
  echo "Unknown model '$MODEL'"
  exit 1
fi
export RUN_SCRIPT
set_job_name "$BASE_JOB_NAME"

# Tell MPI which hosts to use
HOST_FLAG=""
if [[ -n "$MPI_HOSTS" ]]; then
  # If MPI_HOSTS is defined, just use it directly
  HOST_FLAG="--host $MPI_HOSTS"
elif [[ -n "$MPI_HOST_FILE" ]]; then
  # Else, use the hostfile flag
  HOST_FLAG="--hostfile $MPI_HOST_FILE"
elif [[ -n "$(command -v sinfo)" ]]; then
  # If slurm is installed, get the idle hosts from it and use the hosts in --host
  SLURM_HOSTS="$(sinfo -N --state=idle | tail -n +2 | awk '{print $1}')"
  SLURM_HOSTS="$(echo "$SLURM_HOSTS" | tr '\n' ',' | sed 's/,$/\n/')"
  HOST_FLAG="--host $SLURM_HOSTS"
else
  # Otherwise, assuming we're running in a single node
  HOST_FLAG="--host localhost"
fi
export HOST_FLAG

# If elasticity is enabled, we always start with 1 worker first and let that worker
# spawn the remaining workers. We need to do this because MPI fate shares all workers
# launched in the same mpirun command. Otherwise, removing a worker would fail the
# entire application.
ACTUAL_NUM_NODES="$NUM_NODES"
if [[ "$ENABLE_ELASTICITY" == "true" ]]; then
  ACTUAL_NUM_NODES="1"
fi

# Horovod flags: see https://github.com/horovod/horovod/blob/master/docs/mpirun.rst
INTERFACES_TO_EXCLUDE="lo,docker0"
if [[ "$IN_DOCKER_CONTAINER" == "true" ]]; then
  INTERFACES_TO_EXCLUDE="$INTERFACES_TO_EXCLUDE,eth1"
fi
if [[ "$ENABLE_ELASTICITY" == "true" ]]; then
  HOROVOD_FLAGS="-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude $INTERFACES_TO_EXCLUDE --bind-to none"
  HOROVOD_FLAGS="$HOROVOD_FLAGS -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^$INTERFACES_TO_EXCLUDE"
  MPI_MAP_BY="slot"
else
  MPI_MAP_BY="node"
fi

# Verbosity settings
MPI_VERBOSE="${MPI_VERBOSE:=false}"
if [[ "$MPI_VERBOSE" == "true" ]]; then
  STDOUT_DEVICE="/dev/stdout"
  STDERR_DEVICE="/dev/stderr"
else
  STDOUT_DEVICE="/dev/null"
  STDERR_DEVICE="/dev/null"
fi

# Redirect downstream scripts outputs to our log file
export LOG_FILE="/dev/stderr"

# If we're inside a container, store model checkpoints under log dir for easier access
if [[ "$IN_DOCKER_CONTAINER" == "true" ]]; then
  export TRAIN_DIR="$LOG_DIR/$JOB_NAME"
fi

# Pass all environment variables to mpirun, with some exceptions
# The format expected by MPI is "-x ENV_VAR1 -x ENV_VAR2 ..."
ALL_ENV_VARS="$(printenv | grep "=" | awk -F "=" '{print $1}')"
ALL_ENV_VARS="$(echo "$ALL_ENV_VARS" | grep -v "BASH\|SSH\|HOSTNAME\|TERMCAP\|_$\|^\s")"
ENV_FLAG="-x $(echo "$ALL_ENV_VARS" | tr '\n' ',' | sed 's/,$/\n/g' | sed 's/,/ \-x /g')"

# Note: setting --bind-to to "none" (default was "core") significantly improves MPI performance
# for multi-threaded applications. See https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php
mpirun\
  $ENV_FLAG\
  $HOST_FLAG\
  $HOROVOD_FLAGS\
  --allow-run-as-root\
  --map-by "$MPI_MAP_BY"\
  --np "$ACTUAL_NUM_NODES"\
  --output-filename "$LOG_DIR/$JOB_NAME"\
  --oversubscribe\
  "$RUN_SCRIPT"\
  1> "$STDOUT_DEVICE" 2> "$STDERR_DEVICE"

