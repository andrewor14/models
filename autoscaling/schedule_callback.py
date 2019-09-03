#!/usr/bin/env python3

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from autoscaling.params import *


def log_fn(msg):
  tf.logging.info("[Autoscaling schedule] %s" % msg)

class AutoscalingScheduleCallback(keras.callbacks.Callback):
  """
  A `keras.callbacks.Callback` that specifies the schedule for adding and removing workers.
  """
  def __init__(self, agent, every_n_steps, min_workers, max_workers, spawn_size):
    self.agent = agent
    self.every_n_steps = every_n_steps
    self.min_workers = min_workers
    self.max_workers = max_workers
    self.spawn_size = spawn_size
    self.start_time = None
    self.num_workers_to_spawn_next_step = 0
    # Number of workers => list of throughputs
    # Note: after compaction, the first item of each list will be (avg_throughput, count)
    self.throughputs = {}
    # Maximum length of each list in `self.throughputs`
    # If the number of values exceed this, then all existing values are averaged
    self.max_throughputs = 1000
    # If we are awaiting new workers already, don't spawn more
    self.awaiting_new_workers = False
    # Whether to replace stragglers with new workers or not
    self.replace_stragglers = os.getenv(AUTOSCALING_REPLACE_STRAGGLERS, "").lower() == "true"
    # If throughput as a fraction of the median throughput falls below this value,
    # then the worker is considered a straggler
    self.straggler_threshold = float(os.getenv(AUTOSCALING_STRAGGLER_THRESHOLD, 0.8))
    # Host port => exponential weighted moving average (EWMA) of throughput as a
    # fraction of median throughput. If the value falls below `self.straggler_threshold`,
    # then the worker is removed
    self.straggler_stats = {}
    # Alpha value for EWMA calculation in `self.straggler_stats`
    self.straggler_ewma_alpha = 0.1
    # Workers to remove the next time there is a pending cluster spec
    # This is for delaying removing stragglers until their replacements have joined
    self.remove_on_pending = []
    # No more workers are spawned if the throughput scaling efficiency falls below this
    self.throughput_scaling_threshold =\
      float(os.getenv(AUTOSCALING_THROUGHPUT_SCALING_THRESHOLD, 0.1))
    # Maybe restore values from checkpoint metadata file
    if is_checkpoint_restart_mode():
      for key, value in os.environ.items():
        if key.startswith(AUTOSCALING_SCHEDULE_THROUGHPUT_PREFIX):
          num_workers = int(key.lstrip(AUTOSCALING_SCHEDULE_THROUGHPUT_PREFIX))
          average, count = tuple(value.split(","))
          self.throughputs[num_workers] = [(float(average), int(count))]
      log_fn("Restored throughputs from checkpoint metadata file: %s" % self.throughputs)
    log_fn("Starting %s (every_n_steps = %s, min_workers = %s, max_workers = %s)" %\
      (self.__class__.__name__, every_n_steps, min_workers, max_workers))

  def on_batch_begin(self, batch, logs):
    self.start_time = time.time()

  def on_batch_end(self, batch, logs):
    """
    Collect throughput and potentially spawn some workers.
    """
    # Skip measuring the first batch after every restart, which often takes longer
    if self.agent.num_steps_since_last_restart < 1:
      self.awaiting_new_workers = False
      return

    # Record throughput
    if self.start_time is not None:
      step_time = time.time() - self.start_time
      num_workers = len(self.agent.cluster_spec["worker"])
      throughput = self.agent.global_batch_size / step_time
      if num_workers not in self.throughputs:
        self.throughputs[num_workers] = []
      self.throughputs[num_workers].append(throughput)
      # Potentially compact values, saving the count
      if len(self.throughputs[num_workers]) > self.max_throughputs:
        self.compact_list(self.throughputs[num_workers])

    # Optionally record straggler statistics
    if self.replace_stragglers:
      self.record_straggler_stats(logs[TENSORFLOW_COMPUTE_ELAPSED])

    # If we are not the master, or we just restarted, don't spawn or remove workers
    if self.agent.task_index > 0 or\
        AUTOSCALING_MASTER_HOST_PORT in os.environ or\
        self.agent.num_steps_since_last_restart < self.every_n_steps:
      return

    # Optionally replace stragglers
    stragglers = []
    if self.replace_stragglers:
      stragglers = self.get_stragglers()
      stragglers = [s for s in stragglers if s not in self.remove_on_pending]
      # Note: we delay to the actual removal of the stragglers to the next time
      # a pending cluster spec is available, which is a potential indication that
      # the replacements have joined
      if len(stragglers) > 0:
        log_fn("Replacing stragglers %s" % stragglers)
        self.remove_on_pending.extend(stragglers)
      # If there is a pending cluster spec, then new workers have joined, in which
      # case we can actually remove any workers that were pending to be removed
      with self.agent.pending_cluster_spec_lock:
        is_pending = self.agent.pending_cluster_spec is not None
      if is_pending:
        if len(self.remove_on_pending) > 0:
          self.agent.client.remove_workers(self.remove_on_pending)
          self.remove_on_pending.clear()

    # Potentially spawn workers
    num_workers_to_spawn = self.num_workers_to_spawn_next_step + len(stragglers)
    self.num_workers_to_spawn_next_step = 0
    # Only spawn if we are not waiting for new workers
    if self.agent.num_steps_since_last_restart % self.every_n_steps == 0 and\
        not self.awaiting_new_workers:
      num_workers_to_spawn += self.get_num_workers_to_spawn()
    if num_workers_to_spawn > 0:
      self.spawn_workers(num_workers_to_spawn)

    # Potentially remove workers
    if num_workers_to_spawn < 0:
      workers = self.agent.cluster_spec["worker"].copy()
      num_workers_to_remove = num_workers_to_spawn * -1
      if len(workers) <= num_workers_to_remove:
        raise ValueError("Cannot remove %s workers when we only have %s" %\
          (num_workers_to_remove, len(workers)))
      # Never remove ourselves
      workers.remove(self.agent.host_port)
      workers_to_remove = workers[num_workers_to_spawn:]
      if is_checkpoint_restart_mode():
        self.checkpoint_restart_adjust_size(num_workers_to_spawn)
      else:
        self.agent.client.remove_workers(workers_to_remove)

  def on_train_end(self, logs):
    """
    If we're running 'checkpoint-restart' mode, save throughput averages to metadata file.
    """
    if is_checkpoint_restart_mode():
      for num_workers, throughputs in self.throughputs.items():
        if len(throughputs) == 0:
          continue
        self.compact_list(throughputs)
        key = "%s%s" % (AUTOSCALING_SCHEDULE_THROUGHPUT_PREFIX, num_workers)
        value = "%s,%s" % throughputs[0]
        self.agent.checkpoint_restart_variables[key] = value

  def compact_list(self, _list):
    """
    Compact a list by replacing its values with a single element (average, count).
    Note that the list to be compacted can already contain this element.
    """
    if len(_list) == 0:
      return
    if isinstance(_list[0], tuple):
      average, count = _list[0]
      new_count = len(_list[1:]) + count
      new_average = (sum(_list[1:]) + (average * count)) / new_count
    else:
      new_count = len(_list)
      new_average = np.mean(_list)
    _list.clear()
    _list.append((new_average, new_count))

  def get_average_throughputs(self):
    """
    Return a map from num workers to average throughputs, collected from `self.throughputs`.
    Note: This method compacts the values of the map in place.
    """
    average_throughputs = {}
    for key, values in self.throughputs.items():
      if len(values) > 0:
        self.compact_list(values)
        average, _ = values[0]
        average_throughputs[key] = average
    return average_throughputs

  def record_straggler_stats(self, compute_time):
    """
    Record per-worker statistics for detecting stragglers.
    This is called at the end of each batch.
    """
    value_to_send = (self.agent.host_port, compute_time)
    gathered_values = self.agent.mpi_communicator.gather(value_to_send, root=0)
    if gathered_values is None:
      return
    # Unzip gathered values, convert compute time to compute throughput
    host_ports = []
    compute_throughputs = []
    for host_port, compute_time in gathered_values:
      host_ports.append(host_port)
      compute_throughputs.append(self.agent.global_batch_size / compute_time)
    # Record an EWMA of throughput as a fraction of the median throughput
    compute_throughputs = np.array(compute_throughputs)
    median_throughput = np.median(compute_throughputs)
    throughput_fractions = compute_throughputs / median_throughput
    for i, host_port in enumerate(host_ports):
      # Master is never a straggler
      if i == 0:
        continue
      if host_port not in self.straggler_stats:
        self.straggler_stats[host_port] = throughput_fractions[i]
      else:
        self.straggler_stats[host_port] =\
          self.straggler_ewma_alpha * throughput_fractions[i] +\
          (1 - self.straggler_ewma_alpha) * self.straggler_stats[host_port]
    # Clean up removed hosts from `self.straggler_stats`
    for host_port in list(self.straggler_stats.keys()):
      if host_port not in host_ports:
        del self.straggler_stats[host_port]

  def get_stragglers(self):
    """
    Return a list of hosts we consider to be stragglers.
    """
    return [hp for hp, throughput_fraction in self.straggler_stats.items()\
      if throughput_fraction < self.straggler_threshold]

  def get_num_workers_to_spawn(self):
    """
    Return a number of workers to add that satisfies the scaling conditions.
    A negative number means that many workers should be removed instead.
    """
    average_throughputs = self.get_average_throughputs()
    current_num_workers = len(self.agent.cluster_spec["worker"])
    # If we only have one data point, always add workers
    if len(average_throughputs) == 1:
      return self.spawn_size
    target = None
    num_workers = list(average_throughputs.keys())
    num_workers.sort()
    condition_results = []
    for i in range(len(num_workers) - 1):
      condition_results.append(self.met_scaling_conditions(
        average_throughputs[num_workers[i]],
        num_workers[i],
        average_throughputs[num_workers[i+1]],
        num_workers[i+1]))
    # If conditions have all failed, then keep removing workers from the
    # earliest point of failure
    if True not in condition_results:
      target = num_workers[0] - self.spawn_size
    elif False not in condition_results:
      # If conditions have all passed, then keep adding workers from the
      # latest point of success
      target = num_workers[-1] + self.spawn_size
    else:
      # Otherwise, some have passed and some have failed, in which case
      # we go to the earliest point of failure and stay threre
      target = num_workers[np.where(np.array(condition_results) == False)[0][0]]
    log_fn("Current num workers = %s" % current_num_workers)
    log_fn("Num workers = %s" % num_workers)
    log_fn("Scaling condition results = %s" % condition_results)
    log_fn("New target = %s" % target)
    return target - current_num_workers

  def met_scaling_conditions(
      self,
      old_throughput,
      old_num_workers,
      new_throughput,
      new_num_workers):
    """
    Return whether the scaling conditions are met, true means keep training.
    """
    # Check throughput scaling efficiency, given by
    #   s_k = (delta(R) / delta(k)) / (r_{k-1}), where
    #     k is the number of workers
    #     R_k is the aggregate throughput with k workers
    #     r_k is the per worker throughput with k workers
    #     s_k is the throughput scaling efficiency
    slope = (new_throughput - old_throughput) / (new_num_workers - old_num_workers)
    old_per_worker_throughput = old_throughput / old_num_workers
    throughput_scaling_efficiency = slope / old_per_worker_throughput
    passed = throughput_scaling_efficiency > self.throughput_scaling_threshold
    log_fn("========== Checking scaling conditions ==========")
    log_fn("Old throughput = %s (num workers = %s)" % (old_throughput, old_num_workers))
    log_fn("New throughput = %s (num workers = %s)" % (new_throughput, new_num_workers))
    log_fn("Slope = %s" % slope)
    log_fn("Throughput scaling efficiency = %s" % throughput_scaling_efficiency)
    log_fn("Throughput scaling threshold = %s" % self.throughput_scaling_threshold)
    log_fn("Passed = %s" % passed)
    log_fn("================================================")
    # TODO: check utility function too
    return passed

  def spawn_workers(self, num_additional_workers):
    """
    Spawn N workers.

    If the cluster does not have enough spare capacity, then spawn as many as possible.
    If not all N were workers spawned this time, try the failed ones again next batch.
    """
    # Check cluster capacity first
    current_num_workers = self.agent.num_expected_workers()
    spare_capacity = self.max_workers - current_num_workers
    if spare_capacity == 0:
      return
    if spare_capacity < num_additional_workers:
      log_fn("Warning: unable to launch %s workers; launching %s instead" %\
        (num_additional_workers, spare_capacity))
      num_additional_workers = spare_capacity
    if is_checkpoint_restart_mode():
      self.checkpoint_restart_adjust_size(num_additional_workers)
    else:
      # Otherwise, spawn workers through MPI
      self.awaiting_new_workers = self.agent.mpi_spawn_workers(num_additional_workers)
      if not self.awaiting_new_workers:
        log_fn("Warning: agent was not ready to spawn worker, trying again next step")
        self.num_workers_to_spawn_next_step = num_additional_workers

  def checkpoint_restart_adjust_size(self, num_additional_workers):
    """
    Adjust the size of the cluster in 'checkpoint-restart' mode.
    We simply tell all workers to terminate after setting the correct number of workers.
    """
    self.agent.client.all_servers_rpc(lambda s: s.set_pending_cluster_spec({"worker":[]}))
    self.agent.checkpoint_restart_variables["NUM_WORKERS"] =\
      len(self.agent.cluster_spec["worker"]) + num_additional_workers

