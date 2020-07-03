#!/usr/bin/env python3

import json
import os

from absl import logging
from mpi4py import MPI
import tensorflow as tf

from virtual import virtual_helper


class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self, strategy):
    logging.info("Initializing elasticity callback, strategy = %s" % strategy)
    self.strategy = strategy

  def on_batch_begin(self, batch, logs=None):
    logging.info("Beginning batch %s" % batch)
    if batch == 1 and MPI.COMM_WORLD.rank <= 1:
      logging.info("Rank %s is replacing strategy extended" % MPI.COMM_WORLD.rank)
      # Get new TF_CONFIG, dump 2 workers
      tf_config = virtual_helper.get_tf_config()
      logging.info("Old TF_CONFIG = %s" % json.dumps(tf_config))
      workers = tf_config["cluster"]["worker"]
      workers.pop()
      workers.pop()
      tf_config = json.dumps(tf_config)
      logging.info("New TF_CONFIG = %s" % tf_config)
      os.environ[virtual_helper.TF_CONFIG] = tf_config
      # Replace extended strategy
      from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
      from tensorflow.python.distribute import distribute_lib
      logging.info("Old strategy extended = %s" % self.strategy._extended)
      self.strategy._extended = CollectiveAllReduceExtended(
        self.strategy, self.strategy._extended._communication, None)
      logging.info("New strategy extended = %s" % self.strategy._extended)
      # Reset some stats in the strategy constructor
      distribute_lib.distribution_strategy_replica_gauge.get_cell(
        "num_workers").set(self.strategy.extended._num_workers)
      self.strategy._extended._retrace_functions_for_each_device = True
      self.strategy._extended._collective_keys._group_key = 1234

