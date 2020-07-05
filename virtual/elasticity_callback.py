#!/usr/bin/env python3

import copy
import json
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


PORT = 16060
RETRY_INTERVAL_SECONDS = 1
COLLECTIVE_ALLREDUCE_GROUP_KEY = None
COLLECTIVE_ALLREDUCE_GROUP_SIZE = None
START_BATCH = None


def get_client(host):
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, PORT))

class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self, strategy):
    logging.info("Initializing elasticity callback, strategy = %s" % strategy)
    self.strategy = strategy
    self.join_lock = threading.Lock()
    self.join_metadata = None
    self.popped_workers = []
    # Listen for requests
    server = xmlrpc.server.SimpleXMLRPCServer(
      (socket.gethostname(), PORT), logRequests=False, allow_none=True)
    server.register_function(self.set_join_metadata)
    t = threading.Thread(target=server.serve_forever)
    t.setDaemon(True)
    t.start()

  def set_join_metadata(self, join_metadata):
    """
    Set the metadata needed to join an existing cluster.
    """
    with self.join_lock:
      self.join_metadata = join_metadata

  def wait_to_join(self):
    """
    Block until `self.join_metadata` is set.
    """
    logging.info("Waiting to join cluster...")
    while True:
      with self.join_lock:
        if self.join_metadata is not None:
          break
      time.sleep(RETRY_INTERVAL_SECONDS)
    logging.info("Received join metadata from master: %s" % self.join_metadata)
    (tf_config, group_key, batch) = self.join_metadata
    self.replace_strategy_extended(tf_config, group_key)
    global START_BATCH
    START_BATCH = batch
    with self.join_lock:
      self.join_metadata = None

  def replace_strategy_extended(self, tf_config, group_key):
    """
    Replace the extended distribution strategy used on this process with one that reflects
    the cluster configuration specified in `tf_config`, which can be a dictionary or a string.
    """
    from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
    from tensorflow.python.distribute import distribute_lib
    if isinstance(tf_config, dict):
      tf_config = json.dumps(tf_config)
    logging.info("New TF_CONFIG = %s" % tf_config)
    os.environ[virtual_helper.TF_CONFIG] = tf_config
    logging.info("Old strategy extended = %s" % self.strategy._extended)
    extended = CollectiveAllReduceExtended(
      self.strategy, self.strategy._extended._communication, None)
    self.strategy._extended = extended
    logging.info("New strategy extended = %s" % self.strategy._extended)
    # Reset some stats in the extended strategy constructor
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

  def first_change(self):
    """
    Ranks 0 and 1 will synchronize with each other while all other workers will block.
    """
    if MPI.COMM_WORLD.rank <= 1:
      tf_config = virtual_helper.get_tf_config()
      logging.info("Old TF_CONFIG = %s" % json.dumps(tf_config))
      workers = tf_config["cluster"]["worker"]
      self.popped_workers = workers[2:]
      tf_config["cluster"]["worker"] = workers[:2]
      self.replace_strategy_extended(tf_config, 1234)
    elif MPI.COMM_WORLD.rank == 2:
      self.wait_to_join()
    else:
      logging.info("It seems I'm no longer needed. Exiting.")
      sys.exit()

  def second_change(self, batch):
    """
    Ranks 0 and 1 will invite rank 2 to rejoin them.
    """
    if MPI.COMM_WORLD.rank <= 1:
      group_key = 2345
      tf_config = virtual_helper.get_tf_config()
      logging.info("Old TF_CONFIG = %s" % json.dumps(tf_config))
      workers = tf_config["cluster"]["worker"]
      new_worker = self.popped_workers[0]
      workers.append(new_worker)
      self.popped_workers = self.popped_workers[1:]
      self.replace_strategy_extended(tf_config, group_key)
      # Wake rank 2
      if MPI.COMM_WORLD.rank == 0:
        tf_config = copy.deepcopy(tf_config)
        tf_config["task"]["index"] = 2
        tf_config = json.dumps(tf_config)
        get_client(new_worker.split(":")[0]).set_join_metadata(
          (tf_config, group_key, batch))

  def on_batch_begin(self, batch, logs=None):
    logging.info("Beginning batch %s" % batch)
    if batch == 10:
      self.first_change()
    elif batch == 20:
      self.second_change(batch)

  def on_batch_end(self, batch, logs=None):
    global COLLECTIVE_ALLREDUCE_GROUP_KEY
    global COLLECTIVE_ALLREDUCE_GROUP_SIZE
    global START_BATCH
    COLLECTIVE_ALLREDUCE_GROUP_KEY = None
    COLLECTIVE_ALLREDUCE_GROUP_SIZE = None
    START_BATCH = None

