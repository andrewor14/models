#!/usr/bin/env python3

from mpi4py import MPI
import tensorflow as tf

from virtual import virtual_helper


class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  pass

