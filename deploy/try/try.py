#!/usr/bin/env python3

from mpi4py import MPI
from virtual import mpi_helper

print("try.py")
if MPI.COMM_WORLD.rank == 0:
  #MPI.COMM_WORLD.Spawn("bash", args=["spawned.sh"], maxprocs=1)
  mpi_helper.spawn_process(8)
print("try.py done")

