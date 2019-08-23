#!/usr/bin/env python3

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
    self.spawn_next_step = False
    log_fn("Starting %s(every_n_steps = %s, max_workers = %s)" %\
      (self.__class__.__name__, every_n_steps, max_workers))

  def on_batch_end(self, batch, logs):
    if self.spawn_next_step or\
        ((self.agent.step_count + 1) % self.every_n_steps == 0 and\
        self.agent.mpi_communicator.size < self.max_workers):
      # In 'checkpoint-restart' mode, we simply tell the all the workers to terminate
      if is_checkpoint_restart_mode():
        self.agent.client.all_servers_rpc(lambda s: s.set_pending_cluster_spec({"worker":[]}))
        self.agent.checkpoint_restart_num_workers = len(self.agent.cluster_spec["worker"]) + 1
      # Otherwise, spawn a worker on the master
      else:
        spawned = self.agent.mpi_spawn_workers(1)
        if not spawned:
          log_fn("Warning: agent was not ready to spawn worker, trying again next step")
        self.spawn_next_step = not spawned

