#!/usr/bin/env python3

from mpi4py import MPI

print("spawned.py")
print("Rank = %s, size = %s" % (MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size))

