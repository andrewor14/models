#!/bin/bash

# ======================================================================
#  A wrapper around an MPI launch script that properly redirects output
# ======================================================================

if [[ "$#" -lt 1 ]]; then
  echo "Usage: redirect_mpi.sh <script_name> [<arg1>] [<arg2>] ..."
  exit 1
fi

if [[ -z "$MPI_SPAWN_LOG_DIR" ]]; then
  echo "ERROR: MPI_SPAWN_LOG_DIR must be set!"
  exit 1
fi

if [[ -z "$MPI_SPAWN_RANK" ]]; then
  echo "ERROR: MPI_SPAWN_RANK must be set!"
  exit 1
fi

# Make sure our log directory doesn't already exist, then create it
RANK_DIR="$MPI_SPAWN_LOG_DIR/1/rank.$MPI_SPAWN_RANK"
if [[ -d "$RANK_DIR" ]]; then
  echo "ERROR: $RANK_DIR already exists!"
  exit 1
fi
mkdir -p "$RANK_DIR"

# Paths to our log files
STDERR_PATH="$RANK_DIR/stderr"
STDOUT_PATH="$RANK_DIR/stdout"

bash "$@" 2> "$STDERR_PATH" 1> "$STDOUT_PATH"

