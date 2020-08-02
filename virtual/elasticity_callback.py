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


ELASTICITY_PORT = 16060
RETRY_INTERVAL_SECONDS = 1
COLLECTIVE_ALLREDUCE_GROUP_KEY = None
COLLECTIVE_ALLREDUCE_GROUP_SIZE = None
START_BATCH = None
START_EPOCH = None
ELASTICITY_VERBOSE = os.getenv("ELASTICITY_VERBOSE", "").lower() == "true"
GROUP_KEY_INCREMENT = 1000


def get_elasticity_client(host):
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, ELASTICITY_PORT))

class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self, strategy):
    logging.info("Initializing elasticity callback, strategy = %s" % strategy)
    self.strategy = strategy
    self.comm = MPI.COMM_WORLD
    self.is_master = virtual_helper.is_master(self.comm)
    self.current_epoch = 0

    # List of all host ports
    self.members = virtual_helper.get_tf_config()["cluster"]["worker"]

    if self.is_master:
      # If set, the next batch will resize the cluster accordingly
      self.new_size = None
      self.group_key = None

      # Next group key to use, incremented by X after each resize
      self.next_group_key = 1234

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
    if num_workers > MPI.COMM_WORLD.size:
      raise ValueError("New number of workers cannot exceed world size %s (was %s)" %\
        (MPI.COMM_WORLD.size, num_workers))
    self.new_size = num_workers
    self.group_key = self.next_group_key
    self.next_group_key += GROUP_KEY_INCREMENT

  def resize(self, new_size, group_key):
    """
    TODO: write this.
    """
    from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
    from tensorflow.python.distribute import distribute_lib

    logging.info("ANDREW: CALLING RESIZE new_size = %s, group_key = %s" % (new_size, group_key))

    # First, replace the MPI communicator
    # If this worker is removed, wait to receive a rejoin request from the master
    comm = MPI.COMM_WORLD
    if comm.rank >= new_size:
      self.comm = comm
      logging.info("Waiting to rejoin cluster...")
      join_tag = group_key + comm.rank
      (new_size, group_key, batch, epoch) = self.comm.recv(source=0, tag=join_tag)
      global START_BATCH
      global START_EPOCH
      START_BATCH = batch
      START_EPOCH = epoch
      self.current_epoch = epoch
    new_group = comm.group.Incl(list(range(new_size)))
    self.comm = comm.Create_group(new_group)

    # TODO: tell removed workers to rejoin
    if self.is_master:
      self.new_size = None
      self.group_key = None

    # Reset TF_CONFIG
    tf_config = json.dumps({
      "cluster": {"worker": self.members[:new_size]},
      "task": {"type": "worker", "index": comm.rank}
    })
    logging.info("New TF_CONFIG = %s" % tf_config)
    os.environ[virtual_helper.TF_CONFIG] = tf_config

    # Replace the extended strategy
    # Reset some stats in the extended strategy constructor
    extended = CollectiveAllReduceExtended(
      self.strategy, self.strategy._extended._communication, None)
    self.strategy._extended = extended
    distribute_lib.distribution_strategy_replica_gauge.get_cell(
      "num_workers").set(extended._num_workers)
    extended._retrace_functions_for_each_device = True

    # Reset group and instance keys to avoid collisions with previous strategies
    collective_keys = extended._collective_keys
    collective_keys._group_key = group_key
    collective_keys._op_instance_key_start = group_key * 10
    collective_keys._get_thread_local_object().op_instance_key =\
      collective_keys._op_instance_key_start
    collective_keys._variable_instance_key = group_key * 1000
    global COLLECTIVE_ALLREDUCE_GROUP_KEY
    global COLLECTIVE_ALLREDUCE_GROUP_SIZE
    COLLECTIVE_ALLREDUCE_GROUP_KEY = group_key
    COLLECTIVE_ALLREDUCE_GROUP_SIZE = self.strategy.extended._num_workers

  def on_batch_begin(self, batch, logs=None):
    logging.info("Beginning batch %s" % batch)
    new_size, group_key = (self.new_size, self.group_key) if self.is_master else (None, None)
    new_size, group_key = self.comm.bcast((new_size, group_key), root=0)
    if new_size is not None and group_key is not None:
      self.resize(new_size, group_key)

  def on_batch_end(self, batch, logs=None):
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

