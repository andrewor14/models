#!/usr/bin/env python3

import os

import tensorflow as tf
from tensorflow.python import keras

from autoscaling.params import *


def log_fn(msg):
  tf.logging.info("[Autoscaling schedule] %s" % msg)

class PeriodicSpawnScheduleCallback(keras.callbacks.Callback):
  """
  A `keras.callbacks.Callback` that spawns a worker every N steps.
  """

  def __init__(self, agent, every_n_steps, max_workers):
    self.agent = agent
    self.every_n_steps = every_n_steps
    self.max_workers = max_workers
    self.step_count = 0
    self.spawn_next_step = False
    log_fn("Starting %s(every_n_steps = %s, max_workers = %s)" %\
      (self.__class__.__name__, every_n_steps, max_workers))

  def on_batch_end(self, batch, logs):
    self.step_count += 1
    if self.spawn_next_step or\
        (self.step_count % self.every_n_steps == 0 and\
        self.agent.mpi_communicator.size < self.max_workers):
      spawned = self.agent.mpi_spawn_worker()
      if not spawned:
        log_fn("Warning: agent was not ready to spawn worker, trying again next step")
      self.spawn_next_step = not spawned

