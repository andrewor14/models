#!/bin/bash

ROLE="$1"

if [[ "$#" != 1 ]]; then
  echo "Usage: run_mpi.sh [role = 'client' or 'server' or 'ompi-server']"
  exit 1
fi

export NUM_NODES=1
export RUN_TAG="mpi-$ROLE"
export USE_HOROVOD="true"

# Client requires MPI_OPEN_PORT_NAME and OMPI_SERVER
if [[ "$ROLE" == "client" ]]; then
  export SCRIPT_NAME="mpi_client.sh"
  if [[ -z "$MPI_OPEN_PORT_NAME" ]]; then
    echo "MPI_OPEN_PORT_NAME must be set"
    exit 1
  fi
  if [[ -z "$OMPI_SERVER" ]]; then
    echo "OMPI_SERVER must be set"
    exit 1
  fi
# Server requires OMPI_SERVER only
elif [[ "$ROLE" == "server" ]]; then
  export SCRIPT_NAME="mpi_server.sh"
  export NUM_NODES=4
  if [[ -z "$OMPI_SERVER" ]]; then
    echo "OMPI_SERVER must be set"
    exit 1
  fi
# OMPI server requires nothing
elif [[ "$ROLE" == "ompi-server" ]]; then
  export SCRIPT_NAME="ompi_server.sh"
else
  echo "Unrecognized role $ROLE"
  exit 1
fi

./run_slurm.sh

