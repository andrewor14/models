#!/usr/bin/env python3

import copy
import json
import math
import os
import socket
import sys
import threading
import time
import xmlrpc.server

from absl import logging
from mpi4py import MPI
import tensorflow as tf

from virtual import virtual_helper


# Constants
ELASTICITY_PORT = 17272
ELASTICITY_VERBOSE = os.getenv("ELASTICITY_VERBOSE", "").lower() == "true"
RETRY_INTERVAL_SECONDS = 1
STARTING_GROUP_KEY = 1234
GROUP_KEY_INCREMENT = 1000
INSTANCE_KEY_MULTIPLE = 10
VARIABLE_INSTANCE_KEY_MULTIPLE = 1000

# Elasticity state passed to tensorflow
# Setting these will signal tensorflow to rebuild the graph
COLLECTIVE_ALLREDUCE_GROUP_KEY = None
COLLECTIVE_ALLREDUCE_GROUP_SIZE = None
START_BATCH = None
START_EPOCH = None

def get_elasticity_client(host):
  """
  Return a client that can communicate with the elasticity server on the master.
  """
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, ELASTICITY_PORT))

class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self, strategy):
    logging.info("Initializing elasticity callback, strategy = %s" % strategy)
    self.strategy = strategy
    self.world_comm = MPI.COMM_WORLD
    self.is_master = virtual_helper.is_master(self.world_comm)
    self.current_epoch = 0

    # The communicator the master uses to communicate with active workers
    self.comm = self.world_comm

    # List of all host ports
    self.members = virtual_helper.get_tf_config()["cluster"]["worker"]

    if self.is_master:
      # If set, the next batch will resize the cluster accordingly
      self.new_size = None
      self.group_key = None

      # Next group key to use, incremented by X after each resize
      self.next_group_key = STARTING_GROUP_KEY

      # Map from MPI rank to tag to use when requesting the worker to rejoin
      self.tags_for_removed_workers = {}

      # Listen for elasticity requests from the user
      server = xmlrpc.server.SimpleXMLRPCServer(
        (socket.gethostname(), ELASTICITY_PORT), logRequests=False, allow_none=True)
      server.register_function(self.set_num_workers)
      t = threading.Thread(target=server.serve_forever)
      t.setDaemon(True)
      t.start()

  def set_num_workers(self, num_workers):
    """
    Resize the cluster to the given value, which must be a multiple of 2.
    This should only called on the master.
    """
    if not self.is_master:
      raise ValueError("Only the master can resize the cluster")
    if not math.log2(num_workers).is_integer():
      raise ValueError("Num workers must be a power of 2 (was %s)" % num_workers)
    if num_workers > self.world_comm.size:
      raise ValueError("New number of workers cannot exceed world size %s (was %s)" %\
        (self.world_comm.size, num_workers))
    if num_workers == self.comm.size:
      return
    self.new_size = num_workers
    self.group_key = self.next_group_key
    self.next_group_key += GROUP_KEY_INCREMENT
    for removed_rank in range(self.world_comm.size)[num_workers:]:
      self.tags_for_removed_workers[removed_rank] = self.group_key + removed_rank

  def transition(self, new_size, group_key, batch):
    """
    Transition to a new cluster.

    This method adjusts the MPI communicator, replaces the extended distribution strategy,
    and informs tensorflow to rebuild the graph with the new group key. If a worker is not
    included in the new cluster, this will block until the master invites the worker to
    rejoin the cluster.
    """
    logging.info("Transitioning to new cluster: size = %s, group key = %s" % (new_size, group_key))

    # Tell removed workers to rejoin the cluster if necessary
    if self.is_master:
      self.new_size = None
      self.group_key = None
      for removed_rank in list(self.tags_for_removed_workers.keys()):
        if removed_rank >= new_size:
          continue
        logging.info("Asking rank %s to rejoin the cluster..." % removed_rank)
        self.world_comm.send((new_size, group_key, self.current_epoch, batch),\
          dest=removed_rank, tag=self.tags_for_removed_workers[removed_rank])
        del self.tags_for_removed_workers[removed_rank]

    # First, replace the MPI communicator
    # If this worker is removed, wait to receive a rejoin request from the master
    if self.world_comm.rank >= new_size:
      logging.info("Waiting to rejoin cluster...")
      join_tag = group_key + self.world_comm.rank
      (new_size, group_key, epoch, batch) = self.world_comm.recv(source=0, tag=join_tag)
      logging.info("Rejoining cluster: size = %s, group key = %s, epoch = %s, batch = %s" %\
        (new_size, group_key, epoch, batch))
      global START_EPOCH
      global START_BATCH
      START_EPOCH = epoch
      START_BATCH = batch
      self.current_epoch = epoch
    new_group = self.world_comm.group.Incl(list(range(new_size)))
    self.comm = self.world_comm.Create_group(new_group)

    # Reset TF_CONFIG
    tf_config = json.dumps({
      "cluster": {"worker": self.members[:new_size]},
      "task": {"type": "worker", "index": self.world_comm.rank}
    })
    logging.info("New TF_CONFIG = %s" % tf_config)
    os.environ[virtual_helper.TF_CONFIG] = tf_config

    # Replace the extended strategy
    # Reset some stats in the extended strategy constructor
    from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
    from tensorflow.python.distribute import distribute_lib
    extended = CollectiveAllReduceExtended(
      self.strategy, self.strategy._extended._communication, None)
    self.strategy._extended = extended
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
      "num_workers").set(extended._num_workers)
    extended._retrace_functions_for_each_device = True

    # Reset group and instance keys to avoid collisions with previous strategies
    collective_keys = extended._collective_keys
    collective_keys._group_key = group_key
    collective_keys._op_instance_key_start = group_key * INSTANCE_KEY_MULTIPLE
    collective_keys._get_thread_local_object().op_instance_key =\
      collective_keys._op_instance_key_start
    collective_keys._variable_instance_key = group_key * VARIABLE_INSTANCE_KEY_MULTIPLE
    global COLLECTIVE_ALLREDUCE_GROUP_KEY
    global COLLECTIVE_ALLREDUCE_GROUP_SIZE
    COLLECTIVE_ALLREDUCE_GROUP_KEY = group_key
    COLLECTIVE_ALLREDUCE_GROUP_SIZE = self.strategy.extended._num_workers

  def on_batch_begin(self, batch, logs=None):
    """
    Check if we need to resize the cluster.
    """
    new_size, group_key = (self.new_size, self.group_key) if self.is_master else (None, None)
    new_size, group_key = self.comm.bcast((new_size, group_key), root=0)
    if new_size is not None and group_key is not None:
      self.transition(new_size, group_key, batch)

  def on_batch_end(self, batch, logs=None):
    """
    Reset all elasticity state passed to tensorflow.
    """
    global COLLECTIVE_ALLREDUCE_GROUP_KEY
    global COLLECTIVE_ALLREDUCE_GROUP_SIZE
    global START_BATCH
    global START_EPOCH
    COLLECTIVE_ALLREDUCE_GROUP_KEY = None
    COLLECTIVE_ALLREDUCE_GROUP_SIZE = None
    START_BATCH = None
    START_EPOCH = None

  def on_epoch_begin(self, epoch, logs=None):
    self.current_epoch = epoch

