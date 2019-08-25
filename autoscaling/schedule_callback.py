#!/usr/bin/env python3

import numpy as np
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
    with self.agent.spawn_lock:
      # Flatten
      num_pending_workers = len(np.hstack(self.agent.spawned_ranks_to_wait_for or [[]]))
      num_expected_workers = self.agent.mpi_communicator.size + num_pending_workers
    if self.spawn_next_step or\
        ((batch + 1) % self.every_n_steps == 0 and num_expected_workers < self.max_workers):
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

