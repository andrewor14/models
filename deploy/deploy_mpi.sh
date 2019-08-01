#!/bin/bash

# ==========================================================
#  Helper script for submitting a job through MPI, to be
#  called by deploy.sh only
#
#  The caller must set the following environment variables:
#  NUM_WORKERS, NUM_WORKERS_PER_NODE, JOB_NAME, RUN_PATH,
#  LAUNCH_SCRIPT_NAME
#
#  The caller may set the following environment variables:
#  MPI_HOST_FILE, MPI_HOSTS, MPI_SILENCE_OUTPUT
# ==========================================================

# Set common configs
source common_configs.sh

if [[ -z "$NUM_WORKERS" ]]; then
  echo "ERROR: NUM_WORKERS must be set."
  exit 1
fi

if [[ -z "$NUM_WORKERS_PER_NODE" ]]; then
  echo "ERROR: NUM_WORKERS_PER_NODE must be set."
  exit 1
fi

if [[ -z "$JOB_NAME" ]]; then
  echo "ERROR: JOB_NAME must be set."
  exit 1
fi

if [[ -z "$RUN_PATH" ]]; then
  echo "ERROR: RUN_PATH must be set."
  exit 1
fi

if [[ -z "$LAUNCH_SCRIPT_NAME" ]]; then
  echo "ERROR: LAUNCH_SCRIPT_NAME must be set."
  exit 1
fi

# Tell MPI which hosts to use, priority order is as follows:
HOST_FLAG=""
DEFAULT_HOST_FILE="hosts.txt"
if [[ -n "$MPI_HOSTS" ]]; then
  # (1) If MPI_HOSTS is set, use it in --host
  HOST_FLAG="--host $MPI_HOSTS"
elif [[ -n "$MPI_HOST_FILE" ]] || [[ -f "$DEFAULT_HOST_FILE" ]]; then
  # (2) If MPI_HOST_FILE is set, use it in --hostfile
  # (3) If the default host file (hosts.txt) exists, use it in --hostfile
  MPI_HOST_FILE="${MPI_HOST_FILE:=$DEFAULT_HOST_FILE}"
  if [[ ! -f "$MPI_HOST_FILE" ]]; then
    echo "ERROR: Host file $MPI_HOST_FILE does not exist"
    exit 1
  fi
  if [[ -n "$(grep 'slots' $MPI_HOST_FILE)" ]]; then
    # User already specified slots in the host file, so we just use those slot assignments
    HOST_FLAG="--hostfile $MPI_HOST_FILE"
  else
    # Otherwise, assign the default number of slots per host to each host
    HOSTS="$(cat hosts.txt | sed s/$/:$NUM_WORKERS_PER_NODE/g | tr '\n' ',' | sed 's/,$/\n/')"
    HOST_FLAG="--host $HOSTS"
  fi
elif [[ -n "$(command -v sinfo)" ]]; then
  # (4) If slurm is installed, get the idle hosts from it and use the hosts in --host
  SLURM_HOSTS="$(sinfo -N --state=idle | tail -n +2 | awk '{print $1}')"
  HOSTS="$(echo "$SLURM_HOSTS" | sed s/$/:$NUM_WORKERS_PER_NODE/g | tr '\n' ',' | sed 's/,$/\n/')"
  HOST_FLAG="--host $HOSTS"
else
  # (5) Otherwise, assuming we're running in a single node
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

# Verbosity settings
STDOUT_DEVICE="/dev/stdout"
STDERR_DEVICE="/dev/stderr"
if [[ "$MPI_SILENCE_OUTPUT" == "true" ]]; then
  STDOUT_DEVICE="/dev/null"
  STDERR_DEVICE="/dev/null"
fi

# TODO: silence this call; it's very noisy
# Note: setting --bind-to to "none" (default was "core") significantly improves MPI performance
# for multi-threaded applications. See https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php
mpirun\
  $ENV_FLAG\
  $HOST_FLAG\
  $HOROVOD_FLAGS\
  --allow-run-as-root\
  --nooversubscribe\
  --np "$NUM_WORKERS"\
  --output-filename "$LOG_DIR/$JOB_NAME"\
  "$RUN_PATH" "$LAUNCH_SCRIPT_NAME"\
  1> "$STDOUT_DEVICE" 2> "$STDERR_DEVICE"

