#!/usr/bin/env python3

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
    if batch == 1:
      pass

