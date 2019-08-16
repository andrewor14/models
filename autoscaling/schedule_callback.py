#!/usr/bin/env python3

import os
import time

import numpy as np
from scipy.optimize import curve_fit
import tensorflow as tf
from tensorflow.python import keras

from autoscaling.params import *


def log_fn(msg):
  tf.logging.info("[Autoscaling schedule] %s" % msg)

class AutoscalingScheduleCallback(keras.callbacks.Callback):
  """
  A `keras.callbacks.Callback` that specifies the schedule for adding and removing workers.
  """

  def __init__(self, agent, max_workers):
    self.agent = agent
    self.max_workers = max_workers
    self.start_time = None
    self.step_count = 0
    self.num_workers_to_spawn_next_step = 0
    # Number of workers => list of throughputs
    self.throughputs = {}
    # Number of throughputs before it counts as a data point in our curve fitting
    self.curve_fitting_min_throughputs = 5
    # Number of points before curve fitting is used
    self.curve_fitting_min_points = 5
    # No more workers are spawned if the throughput increase falls below this threshold
    self.throughput_increase_threshold =\
      float(os.getenv(AUTOSCALING_THROUGHPUT_INCREASE_THRESHOLD, 0.1))
    # No more workers are spawned if the throughput increase to cost increase ratio
    # falls below this threshold
    self.throughput_increase_over_cost_increase_threshold =\
      float(os.getenv(AUTOSCALING_THROUGHPUT_INCREASE_OVER_COST_INCREASE_THRESHOLD, 1))

  def on_batch_begin(self, batch, logs):
    # Skip measuring the first batch, which often takes longer
    if batch > 0:
      self.start_time = time.time()

  def on_batch_end(self, batch, logs):
    """
    Collect throughput and potentially spawn some workers.
    """
    self.step_count += 1
    if self.start_time is not None:
      step_time = time.time() - self.start_time
      batch_size = self.agent.global_batch_size
      num_workers = len(self.agent.cluster_spec["worker"])
      throughput = batch_size / step_time
      if num_workers not in self.throughputs:
        self.throughputs[num_workers] = []
      self.throughputs[num_workers].append(throughput)
      self.start_time = None
    self.maybe_spawn_workers()

  def maybe_spawn_workers(self):
    """
    Spawn schedule decision logic, to be overridden by subclasses.
    """
    raise NotImplementedError

  def get_num_workers_to_spawn(self, num_additional_workers):
    """
    Return a number of workers to add that satisfies the throughput and cost thresholds.

    The number of workers returned is <= `num_additional_workers`.
    A negative number means that many workers should be removed instead.
    """
    if num_additional_workers < 1:
      raise ValueError("Cannot spawn %s workers" % num_additional_workers)
    num_workers = []
    average_throughputs = []
    current_num_workers = len(self.agent.cluster_spec["worker"])
    for n, throughputs in self.throughputs.items():
      if len(throughputs) >= self.curve_fitting_min_throughputs:
        num_workers.append(n)
        average_throughputs.append(np.mean(throughputs))
    # If we only have 1 data point, always allow adding 1 worker
    if len(average_throughputs) == 1:
      return 1
    # If we don't have enough points to use curve fitting, add 1 worker if the previous
    # increase in throughput satisfies our thresholds
    if len(average_throughputs) < self.curve_fitting_min_points:
      current_throughput = average_throughputs[current_num_workers]
      previous_throughput = average_throughputs[current_num_workers - 1]
      return 1 if self.check_spawn_thresholds(previous_throughput, current_throughput) else 0
    # Otherwise, use curve fitting to estimate the largest additional number of workers
    # under `num_additional_workers` that still satisfies the thresholds
    def line(x, a, b):
      return a - b / x
    fit_vars, _ = curve_fit(line, num_workers, average_throughputs)
    def fitted_line(x):
      return line(x, fit_vars[0], fit_vars[1])
    for i in range(num_additional_workers): 
      current_throughput = fitted_line(current_num_workers + i)
      next_throughput = fitted_line(current_num_workers + i + 1)
      # Stop as soon as the thresholds are not satisfied
      if not self.check_spawn_thresholds(current_throughput, next_throughput):
        return i
    return num_additional_workers 

  def check_spawn_thresholds(self, old_throughput, new_throughput):
    """
    Return whether the increase in throughput exceeds the threshold.
    """
    return (new_throughput - old_throughput) / old_throughput > self.throughput_increase_threshold

  def spawn_workers(self, num_additional_workers):
    """
    Spawn N workers.
    If the cluster does not have enough spare capacity, then spawn as many as possible.
    If not all N were workers spawned this time, try the failed ones again next batch.
    """
    # Check cluster capacity first
    current_num_workers = len(self.agent.cluster_spec["worker"])
    spare_capacity = self.max_workers - current_num_workers
    if spare_capacity == 0:
      return
    if spare_capacity < num_additional_workers:
      log_fn("Warning: unable to launch %s workers; launching %s instead" %\
        (num_additional_workers, spare_capacity))
      num_additional_workers = spare_capacity
    # In 'checkpoint-restart' mode, we simply tell the all the workers to terminate
    if is_checkpoint_restart_mode():
      self.agent.checkpoint_restart_num_workers =\
        current_num_workers + num_additional_workers
      for server in self.agent.client.servers:
        server.set_pending_cluster_spec({"worker":[]})
    else:
      # Otherwise, spawn workers through MPI
      self.num_workers_to_spawn_next_step = 0
      for _ in range(num_additional_workers):
        if not self.agent.mpi_spawn_worker():
          log_fn("Warning: agent was not ready to spawn worker, trying again next step")
          self.num_workers_to_spawn_next_step += 1

class PeriodicSpawnScheduleCallback(AutoscalingScheduleCallback):
  """
  A `keras.callbacks.Callback` that spawns a worker every N steps.
  """

  def __init__(self, agent, every_n_steps, max_workers):
    super(PeriodicSpawnScheduleCallback, self).__init__(agent, max_workers)
    self.every_n_steps = every_n_steps
    log_fn("Starting %s(every_n_steps = %s, max_workers = %s)" %\
      (self.__class__.__name__, every_n_steps, max_workers))

  def maybe_spawn_workers(self):
    num_workers_to_spawn = self.num_workers_to_spawn_next_step
    if self.step_count % self.every_n_steps == 0:
      num_workers_to_spawn += self.get_num_workers_to_spawn(1)
    if num_workers_to_spawn > 0:
      self.spawn_workers(num_workers_to_spawn)

