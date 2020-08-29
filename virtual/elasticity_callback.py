#!/usr/bin/env python3

import copy
import json
import math
import os
import socket
import threading
import time
import xmlrpc.server

from absl import logging
from mpi4py import MPI
import numpy as np
import tensorflow as tf

from virtual import virtual_helper


# Constants and environment variables
RETRY_INTERVAL_SECONDS = 1
ELASTICITY_PORT = 17272
ELASTICITY_VERBOSE = os.getenv("ELASTICITY_VERBOSE", "").lower() == "true"
ENABLE_ELASTICITY = os.getenv("ENABLE_ELASTICITY", "").lower() == "true"
ELASTICITY_MASTER = "ELASTICITY_MASTER"
SPAWN_START_RANK = "SPAWN_START_RANK"

# Elasticity state passed to tensorflow
# Setting these will signal tensorflow to rebuild the graph
NUM_VIRTUAL_NODES = None
START_BATCH = None
START_EPOCH = None

# Singleton elasticity callback, defined only if ENABLE_ELASTICITY is true
ELASTICITY_CALLBACK = None

def initialize_singleton_callback():
  """
  Initialize a singleton elasticity callback for this program.

  This needs to be initialized in the very beginning of the program because the
  elasticity callback fetches the correct CUDA_VISIBLE_DEVICES from the master and
  sets it for spawned workers.
  """
  global ELASTICITY_CALLBACK
  ELASTICITY_CALLBACK = ElasticityCallback()

def get_elasticity_client(host):
  """
  Return a client that can communicate with the elasticity server on the master.
  """
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, ELASTICITY_PORT))

class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self):
    logging.info("Initializing elasticity callback")
    self.comm = MPI.COMM_WORLD.Dup()
    self.is_joining = ELASTICITY_MASTER in os.environ
    self.is_master = virtual_helper.is_master(self.comm) and not self.is_joining
    self.current_batch = 0
    self.current_epoch = 0
    num_nodes = int(os.environ[virtual_helper.NUM_NODES])

    # Number of total virtual nodes across all devices
    self.num_total_virtual_nodes =\
      int(os.getenv(virtual_helper.NUM_VIRTUAL_NODES_PER_DEVICE, 1)) * num_nodes

    # List of all hosts in our communicator
    self.members = self.comm.allgather(MPI.Get_processor_name())

    if self.is_master:
      # If set, the next batch will resize the cluster accordingly
      self.new_size = None

      # The communicator returned from the last call to MPI spawn, if any
      self.spawned_communicator = None

      # Listen for elasticity requests from the user
      server = xmlrpc.server.SimpleXMLRPCServer(
        (socket.gethostname(), ELASTICITY_PORT), logRequests=False, allow_none=True)
      server.register_function(self.set_num_workers)
      server.register_function(self.spawn)
      server.register_function(self.handle_join)
      t = threading.Thread(target=server.serve_forever)
      t.setDaemon(True)
      t.start()

      # In elasticity mode, we always start with a single worker and let it
      # spawn the remaining workers so the fates of the workers are not tied
      self.awaiting_initial_workers = True
      self.spawn(num_nodes - 1)

  def set_num_workers(self, num_workers):
    """
    Resize the cluster to the given value, which must be a multiple of 2.
    This should only called on the master.
    """
    self._check_master_request("resize")
    if not math.log2(num_workers).is_integer():
      raise ValueError("Num workers must be a power of 2 (was %s)" % num_workers)
    if num_workers == self.comm.size:
      return
    if num_workers > self.comm.size:
      self.spawn(num_workers - self.comm.size)
    else:
      self.new_size = num_workers

  def spawn(self, n):
    """
    Spawn processes through MPI.

    This should only be called on the master, and can only be called when there are
    no outstanding spawned processes.
    """
    self._check_master_request("spawn")
    all_possible_hosts = virtual_helper.get_all_mpi_hosts()
    target_hosts = [h for h in all_possible_hosts if h not in self.members]
    if len(target_hosts) >= n:
      env = {
        ELASTICITY_MASTER: MPI.Get_processor_name(),
        SPAWN_START_RANK: self.comm.size
      }
      self.spawned_communicator = virtual_helper.mpi_spawn(target_hosts[:n], env)
    else:
      logging.warn("Not enough hosts to spawn %s more processes, ignoring spawn request" % n)

  def handle_join(self, spawned_size):
    """
    Notify the master that the spawned workers are ready to join the cluster.

    This should only be called on the master client.
    Return the expected size of the new cluster.
    """
    self._check_master_request("join")
    self.new_size = self.comm.size + spawned_size
    return self.new_size

  def _check_master_request(self, action):
    """
    Helper method to check if requests received at the master client are valid.
    """
    if not self.is_master:
      raise ValueError("Only the master can handle %s requests" % action)
    if self.new_size is not None:
      raise ValueError("Existing resizing request in progress, not accepting further requests")
    if action == "join":
      if self.spawned_communicator is None:
        raise ValueError("Unexpected join request: there was no spawned communicator")
    else:
      if self.spawned_communicator is not None:
        raise ValueError("Existing spawn request in progress, not accepting further requests")

  def transition(self, new_size):
    """
    Transition to a new cluster.

    This method adjusts the MPI communicator and informs tensorflow to rebuild the graph
    with a new horovod allreduce function. Workers that are removed will exit.
    """
    logging.info("Transitioning to new cluster of size %s" % new_size)

    # If this worker is removed, wait to receive a rejoin request from the master
    if self.comm.rank >= new_size:
      logging.info("Leaving cluster because our rank is %s and the new size is %s" %\
        (self.comm.rank, new_size))
      # Wait until all existing workers have reinitialized horovod before exiting
      # Exiting early will cause MPI to kill the entire job
      self.comm.barrier()
      os._exit(0)

    # If we are expanding the cluster, merge the old and new communicators
    old_comm = self.comm
    old_size = new_size - self.comm.size if self.is_joining else self.comm.size
    if new_size > self.comm.size:
      if self.is_joining:
        intercomm = MPI.Comm.Get_parent()
      elif self.is_master:
        intercomm = self.spawned_communicator
        if intercomm is None:
          raise ValueError("Attempted to expand but there was no spawned communicator")
        self.spawned_communicator = None
      else:
        intercomm = None
      self.comm = virtual_helper.mpi_expand(self.comm, intercomm)
    else:
      # Otherwise, we are shrinking the cluster, so just use a subset of ranks
      new_group = self.comm.group.Incl(list(range(new_size)))
      self.comm = self.comm.Create_group(new_group)
    self.members = self.comm.allgather(MPI.Get_processor_name())

    # Reinitialize horovod to use the new communicator
    # We can release the removed workers from the old communicator after doing this
    virtual_helper.initialize_horovod(self.comm)
    old_comm.barrier()

    # Transfer parameters from existing workers to new workers if the cluster expanded
    self.transfer_parameters(old_size, new_size)

    # Trigger tensorflow to update the graph with the correct number of virtual nodes,
    # step number, and epoch number
    global START_BATCH
    global START_EPOCH
    global NUM_VIRTUAL_NODES
    self.current_batch, self.current_epoch, self.num_total_virtual_nodes =\
      self.comm.bcast((self.current_batch, self.current_epoch, self.num_total_virtual_nodes), root=0)
    START_BATCH = self.current_batch
    START_EPOCH = self.current_epoch
    NUM_VIRTUAL_NODES = self.num_total_virtual_nodes / self.comm.size
    if not NUM_VIRTUAL_NODES.is_integer():
      raise ValueError("Num virtual nodes must be an integer! (was %s)" % NUM_VIRTUAL_NODES)
    NUM_VIRTUAL_NODES = int(NUM_VIRTUAL_NODES)
    self.new_size = None

  def transfer_parameters(self, old_size, new_size):
    """
    During transition, transfer parameters from existing workers to new workers.
    """
    if old_size >= new_size:
      return

    if self.model is None:
      raise ValueError("self.model was not set!")

    self.log_parameters("[Before transfer] ", old_size)

    # Split parameters into N slices, where N is the number of existing workers,
    # and perform an allgather to send these slices to the new workers
    bcast_parameters = None
    sending_ranks = list(range(old_size))
    if self.comm.rank in sending_ranks:
      bcast_parameters = [v.value() for v in np.array_split(\
        self.model.trainable_variables, old_size)[self.comm.rank]]
    bcast_parameters = self.comm.allgather(bcast_parameters)

    # Update model on receiving ranks
    if self.comm.rank not in sending_ranks:
      received_parameters = []
      for p in bcast_parameters:
        if p is not None:
          received_parameters.extend(p)
      assert(len(received_parameters) == len(self.model.trainable_variables))
      tf.python.keras.backend.batch_set_value(\
        list(zip(self.model.trainable_variables, received_parameters)))

    self.log_parameters("[After transfer] ", old_size)

  def log_parameters(self, prefix, num_chunks, n=10):
    """
    Helper method to print the first values of the first N parameters in each group.
    """
    if ELASTICITY_VERBOSE:
      reshaped = [float("%.6g" % tf.reshape(v, [-1])[:1].numpy().tolist()[0])\
        for v in self.model.trainable_variables]
      reshaped = [p[:n].tolist() for p in np.array_split(np.array(reshaped), num_chunks)]
      reshaped = ",\n".join(["  " + str(p) for p in reshaped])
      logging.info("%sThe first values of the first %s parameters in each group are:\n%s" %\
        (prefix, n, reshaped))

  def on_train_begin(self, logs=None):
    global NUM_VIRTUAL_NODES
    NUM_VIRTUAL_NODES = int(self.num_total_virtual_nodes / self.comm.size)

  def on_batch_begin(self, batch, logs=None):
    """
    Check if we need to resize the cluster and, if so, transition to the new cluster.
    """
    self.current_batch = batch
    # New workers should run a batch before joining the cluster
    # This significantly shortens the GPU idle time during transitions because the existing
    # workers do not have to wait for the new workers to bootstrap, which can take minutes.
    if self.is_joining and batch == 1:
      self.comm.barrier()
      new_size = None
      # Notify the master the new workers are ready to join the cluster
      if self.comm.rank == 0:
        master_client = get_elasticity_client(os.environ[ELASTICITY_MASTER])
        new_size = master_client.handle_join(self.comm.size)
      new_size = self.comm.bcast(new_size, root=0)
      self.transition(new_size)
      self.is_joining = False
    else:
      # Wait for the initial set of workers to join before proceeding further
      if self.is_master and self.awaiting_initial_workers and batch == 1:
        while self.new_size is None:
          logging.info("Waiting for the initial set of workers to join...")
          time.sleep(RETRY_INTERVAL_SECONDS)
        self.awaiting_initial_workers = False
      # Maybe transition to a new cluster
      new_size = self.new_size if self.is_master else None
      new_size = self.comm.bcast(new_size, root=0)
      if new_size is not None:
        self.transition(new_size)

  def on_batch_end(self, batch, logs=None):
    """
    Reset all elasticity state passed to tensorflow.
    """
    global NUM_VIRTUAL_NODES
    global START_BATCH
    global START_EPOCH
    NUM_VIRTUAL_NODES = None
    START_BATCH = None
    START_EPOCH = None

  def on_epoch_begin(self, epoch, logs=None):
    self.current_epoch = epoch

