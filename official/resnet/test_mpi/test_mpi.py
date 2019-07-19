#!/usr/bin/env python3

import os
import re
import socket
import sys
import time
import traceback

from mpi4py import MPI

MASTER_HOST = "MASTER_HOST"
OMPI_MCA_orte_hnp_uri = "OMPI_MCA_orte_hnp_uri"
MPI_PORT = 10222
MAX_ITERS = 10
STARTING_TAG = 14444

def log(msg):
  print(msg)

def main():
  comm = MPI.COMM_WORLD.Dup()
  is_joining = MASTER_HOST in os.environ
  tag = STARTING_TAG if not is_joining else None
  # We start with MPI.COMM_WORLD and STARTING TAG.
  # In each future iteration, we use the resulting communicator from the previous
  # iteration and increment the tag. To ensure everyone uses the same tag, the master
  # sends its tag to the newly spawned worker, and thereafter everyone increments his
  # own tag in each future iteration.
  for i in range(MAX_ITERS):
    log("Running iteration %s" % (i+1))
    comm, tag = join_procedure(comm, is_joining, tag)
    is_joining = False
    tag += 1
  log("====================================================\n"
      "|                                                  |\n"
      "|                     ALL DONE                     |\n"
      "|                                                  |\n"
      "====================================================")

def join_procedure(comm, is_joining, tag=None):
  log("Calling join procedure with comm = %s, is_joining = %s, tag = %s" % (comm, is_joining, tag))
  # Decide who's who
  rank = comm.rank
  is_master = rank == 0 and not is_joining

  # Master will start a TCP server that the spawned worker will connect to
  # Once this TCP connection is set up, the file descriptors on both sides
  # will be passed in to MPI.Comm.Join to merge the two nodes
  if is_master or is_joining:
    # Parse master IP
    master_ip = "ns-l10c1n3" #HACK
    target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set up connection on both ends
    if is_master:
      target_socket.bind((master_ip, MPI_PORT))
      target_socket.listen()
      log("Master listening for MPI join requests on (%s, %s)" % (master_ip, MPI_PORT))
      (conn, addr) = target_socket.accept()
      target_fd = conn.fileno()
      log("Master accepted connection from %s, file descriptor = %s" % (addr, target_fd))
    else:
      target_socket.connect((master_ip, MPI_PORT))
      target_fd = target_socket.fileno()
      log("Spawned worker connected to master at (%s, %s), file descriptor = %s" %\
        (master_ip, MPI_PORT, target_fd))
    # Merge master and the spawned worker into an intracomm
    log("Joining remote intercomm through socket file descriptor %s" % target_fd)
    remote_intercomm = MPI.Comm.Join(target_fd)
    log("Joined remote intercomm through socket file descriptor %s" % target_fd)
    merged_intracomm = remote_intercomm.Merge(is_joining)
    target_socket.close()
    log("Our rank is %s in merged_intracomm (size %s)" % (merged_intracomm.rank, merged_intracomm.size))
    # Send tag to spawned worker through merged_intracomm
    if is_master:
      log("Sending tag %s to spawned worker" % tag)
      merged_intracomm.send(tag, dest=1)
    else:
      log("Waiting to receive tag from master")
      tag = merged_intracomm.recv(source=0)
      log("Received tag = %s from master!" % tag)
  else:
    merged_intracomm = MPI.Intracomm(MPI.COMM_NULL)

  # Wait for everyone
  log("Waiting for everyone to finish merging")
  comm.barrier()

  # Create an intercomm out of the two intracomms: merged_intracomm and comm
  if is_joining:
    log("Spawned worker trying to super merge with master using tag %s" % tag)
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(MPI.COMM_SELF, 0, merged_intracomm, 0, tag)
    log("Spawned worker just super merged with master")
  else:
    log("Trying to super merge with spawned worker using tag %s" % tag)
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(comm, 0, merged_intracomm, 1, tag)
    log("Just super merged with spawned worker")

  # Convert the this new intercomm to an intracomm
  super_merged_intracomm = super_merged_intercomm.Merge(is_joining)
  log("Our rank is %s in super_merged_intracomm (size %s)" %\
    (super_merged_intracomm.rank, super_merged_intracomm.size))

  return super_merged_intracomm, tag

  # Train on this new intracomm, which contains the spawned worker
  #log("Initializing Horovod.")
  #hvd.init(super_merged_intracomm)
  #log("Running dummy for 5 seconds")
  #time.sleep(5)
  #log("Shutting down")
  #hvd.shutdown()
    
if __name__ == "__main__":
  main()

