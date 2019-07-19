#!/usr/bin/env python3

import os
import re
import socket
import sys
import time
import traceback

from mpi4py import MPI
import horovod.tensorflow as hvd
import tensorflow as tf

from official.resnet.autoscaling_agent import AutoscalingAgent
from official.resnet.autoscaling_params import *
from slurm.tensorflow_on_slurm import running_through_slurm, set_tf_config


OMPI_MCA_orte_hnp_uri = "OMPI_MCA_orte_hnp_uri"
MPI_PORT = 10222
TAG = 14444

def log(msg):
  tf.compat.v1.logging.info(msg)

def main():
  # Decide who's who
  rank = MPI.COMM_WORLD.Get_rank()
  is_spawned = AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_master = rank == 0 and not is_spawned

  # Set TF_CONFIG for autoscaling agent
  if not running_through_slurm():
    raise ValueError("Only slurm mode is supported")
  set_tf_config(num_ps=0)
  autoscaling_agent = AutoscalingAgent()

  #if is_master:
  #  port_name = MPI.Open_port()
  #  os.environ["MPI_OPEN_PORT_NAME"] = port_name

  # Wait for new node to join
  if not is_spawned:
    while True:
      with autoscaling_agent.pending_cluster_spec_lock:
        if autoscaling_agent.pending_cluster_spec is not None:
          break
      log("Still waiting for new worker to join...")
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
  autoscaling_agent.initialize()
  log("Initialized")

  #if is_master:
  #  log("Master waiting for client to connect at %s" % port_name)
  #  intercomm = MPI.COMM_SELF.Accept(port_name)
  #  log("Connected!")
  #elif is_spawned:
  #  master_mpi_vars = autoscaling_agent.client.master_server.get_mpi_variables()
  #  port_name = master_mpi_vars["MPI_OPEN_PORT_NAME"]
  #  log("Spawned worker connecting to master at %s" % port_name)
  #  intercomm = MPI.COMM_SELF.Connect(port_name)
  #  log("Connected!")

  # Master will start a TCP server that the spawned worker will connect to
  # Once this TCP connection is set up, the file descriptors on both sides
  # will be passed in to MPI.Comm.Join to merge the two nodes
  if is_master or is_spawned:
    # Parse master IP
    master_mpi_vars = autoscaling_agent.client.master_server.get_mpi_variables()
    master_mpi_uri = master_mpi_vars[OMPI_MCA_orte_hnp_uri]
    master_ip = re.match(".*tcp://([.0-9]*),.*", master_mpi_uri).groups()[0]
    log("Master ip = %s" % master_ip)
    target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set up connection on both ends
    if is_master:
      target_socket.bind((master_ip, MPI_PORT))
      target_socket.listen()
      log("Master listening for MPI join requests on (%s, %s)" % (master_ip, MPI_PORT))
      (conn, addr) = target_socket.accept()
      target_fd = conn.fileno()
      target_socket.close()
      log("Master accepted connection from %s, file descriptor = %s" % (addr, target_fd))
    else:
      log("Spawned worker connecting to master on (%s, %s)" % (master_ip, MPI_PORT))
      while True:
        try:
          target_socket.connect((master_ip, MPI_PORT))
          break
        except ConnectionRefusedError as connection_refused:
          log("... connection refused, trying again")
          time.sleep(1)
      target_fd = target_socket.fileno()
      log("Spawned worker connected to master at (%s, %s), file descriptor = %s" %\
        (master_ip, MPI_PORT, target_fd))
    # Merge master and the spawned worker into an intracomm
    log("Joining remote intercomm through socket file descriptor %s" % target_fd)
    remote_intercomm = MPI.Comm.Join(target_fd)
    log("Joined remote intercomm through socket file descriptor %s" % target_fd)
    merged_intracomm = remote_intercomm.Merge(is_spawned)
    merged_intracomm.barrier()
    log("Merged with remote intercomm through socket")

  log("Let's do this much for now")
  time.sleep(600)
  sys.exit(1)







  # Create an intercomm out of the two intracomms: merged_intracomm and MPI.COMM_WORLD
  if is_spawned:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(MPI.COMM_SELF, 0, merged_intracomm, 0, TAG)
    log("Spawned worker just super merged with everyone else")
  else:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(MPI.COMM_WORLD, 0, merged_intracomm, 1, TAG)
    log("Just super merged with spawned worker")

  # Convert the this new intercomm to an intracomm
  super_merged_intracomm = super_merged_intercomm.Merge(is_spawned)
  log("Our rank is %s/%s in super_merged_intracomm" %\
    (super_merged_intracomm.rank, super_merged_intracomm.size))

  # Train on this new intracomm, which contains the spawned worker
  log("Initializing Horovod.")
  hvd.init(super_merged_intracomm)
  log("Running dummy for 5 seconds")
  time.sleep(5)
  log("Shutting down")
  hvd.shutdown()
  log("====================================================\n"
      "|                                                  |\n"
      "|                     ALL DONE                     |\n"
      "|                                                  |\n"
      "====================================================")
    
if __name__ == "__main__":
  main()

