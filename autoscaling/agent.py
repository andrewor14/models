#!/usr/bin/env python3

import errno
import io
import json
import os
import socket
import threading
import time
import traceback
import xmlrpc.client

from mpi4py import MPI
import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import cross_device_utils, distribute_coordinator
from tensorflow.python.eager import context

from autoscaling import autoscaling_helper
from autoscaling.client import connect, convert_port, AutoscalingClient
from autoscaling.service import listen_for_requests
from autoscaling.params import *
from deploy import mpi_helper


class AutoscalingAgent:
  """
  An agent that keeps track of autoscaling state for this process.

  This agent communicates with other agents on remote processes to
  ensure that the autoscaling status on this process transitions in
  synchrony with everyone else.

  Upon detecting a change in cluster membership, the agents vote to
  decide when to restart. Restarting requires a unanimous consensus
  among all the agents.
  """

  def __init__(self, num_gpus_per_worker=0, global_batch_size=0, use_horovod=False):
    self.saved_variables = None
    self.checkpoint_restart_variables = {}
    self.num_gpus_per_worker = num_gpus_per_worker
    self.use_horovod = use_horovod
    self.num_steps_since_last_restart = 0
    self.min_steps_between_restart = int(os.getenv(AUTOSCALING_MIN_STEPS_BETWEEN_RESTART, 1))
    self.detach_when_removed = os.getenv(AUTOSCALING_DETACH_WHEN_REMOVED, "true").lower() == "true"
    self.detached_mode = False
    self.cluster_initialized = False

    # Batch sizes
    self.global_batch_size = global_batch_size
    self.max_local_batch_size = os.getenv(AUTOSCALING_MAX_LOCAL_BATCH_SIZE)
    self.max_global_batch_size = os.getenv(AUTOSCALING_MAX_GLOBAL_BATCH_SIZE)
    if is_autoscaling_mode():
      if self.max_local_batch_size is None or self.max_global_batch_size is None:
        raise ValueError("Max local and global batch sizes must both be set")
      self.max_local_batch_size = int(self.max_local_batch_size)
      self.max_global_batch_size = int(self.max_global_batch_size)
      autoscaling_helper.MAX_LOCAL_BATCH_SIZE = self.max_local_batch_size
      log_fn("Running with max local batch size = %s, max global batch size = %s" %\
        (self.max_local_batch_size, self.max_global_batch_size))
      log_fn("Warning: ignoring BATCH_SIZE (currently set to %s)" % global_batch_size)
      self.global_batch_size = self.max_local_batch_size
    else:
      autoscaling_helper.MAX_LOCAL_BATCH_SIZE = global_batch_size

    # Initial number of workers in "autoscaling" mode
    self.autoscaling_initial_workers = int(os.getenv(AUTOSCALING_INITIAL_WORKERS, 1))

    # A lambda that returns a 4-tuple:
    #  (1) Number of batches processed in this epoch so far,
    #  (2) Number of epochs processed so far, and
    #  (3) Number of batches per epoch
    #  (4) Number of total epochs
    # This is used for serving our progress through the autoscaling service
    self.get_progress_method = None

    # A map from host port to cuda visible devices (list of numbers) assigned
    self.cuda_visible_devices_map = {}

    # Parse this process' host port from TF_CONFIG
    tf_config = get_tf_config()
    self.task_type = tf_config["task"]["type"]
    self.task_index = tf_config["task"]["index"]
    self.cluster_spec = tf_config["cluster"]
    self.host_port = self.cluster_spec[self.task_type][self.task_index]

    # If we are using horovod, unset TF_CONFIG to avoid interference from tensorflow
    if self.use_horovod:
      del os.environ["TF_CONFIG"]

    # ========= MPI stuff ==========

    # The MPI communicator that Horovod will use
    self.mpi_communicator = MPI.COMM_WORLD.Dup()

    # A lock for controlling spawn variables
    self.spawn_lock = threading.Lock()

    # A map from rank to MPI intercommunicators created from spawning worker processes
    # These communicators will be merged into `self.mpi_communicator` on restart
    # Accesses must be guarded by `self.spawn_lock`
    self.mpi_spawned_communicators = {}

    # The unique rank assigned to the next spawned worker, incremented every time a
    # worker is spawned. Only used on the master server.
    # Accesses must be guarded by `self.spawn_lock`.
    self.mpi_next_rank = len(self.cluster_spec["worker"])

    # A queue of lists of spawned ranks to wait for on restart
    # The first item is a list of numbers that represent the ranks to wait for the
    # next time `self.initialize` is called. A new item is appended every time
    # `self.mpi_spawn_workers` succeeds.
    # Accesses must be guarded by `self.spawn_lock`
    self.spawned_ranks_to_wait_for = []

    # If non-empty, `self.mpi_spawn_workers` will simply tell the first N workers
    # in this list to reattach themselves to this cluster
    self.removed_host_ports = []

    # Keep track of all possible hosts so we can manually spawn using round robin
    # Guarded by `self.spawn_lock`
    self.all_possible_hosts = []
    # Keep track of how many workers live on which hosts (including expected ones)
    # Guarded by `self.spawn_lock`
    self.host_counts = {}
    # Maximum number of workers that can be launched on each host
    # This shares the same indices as `self.all_possible_hosts`
    # Guarded by `self.spawn_lock`
    self.max_slots = []

    # Optionally plant stragglers in the cluster
    # We simulate stragglers by inflating local batch size on the selected ranks
    self.local_batch_size_multiplier = 1
    straggler_ranks = os.getenv(AUTOSCALING_STRAGGLER_RANKS)
    if straggler_ranks is not None:
      straggler_ranks = [int(r) for r in straggler_ranks.split(",")]
      my_rank = int(os.getenv(mpi_helper.MPI_SPAWN_RANK, self.mpi_communicator.rank))
      if my_rank in straggler_ranks:
        if my_rank == 0:
          raise ValueError("Rank 0 cannot be a straggler")
        self.local_batch_size_multiplier = float(os.getenv(AUTOSCALING_STRAGGLER_MULTIPLIER, 1.5))
        log_fn("This rank is a straggler (%sx slower than normal)" % self.local_batch_size_multiplier)
    if self.local_batch_size_multiplier < 0:
      raise ValueError("Local batch size multiplier must be >= 1, was %s" %\
        self.local_batch_size_multiplier)

    # ========= Autoscaling stuff ==========

    # Status to synchronize cluster membership changes
    # Accesses must be guarded by `self._status_lock`
    self._status = AutoscalingStatus.READY_TO_SYNC
    self._status_lock = threading.Lock()

    # The cluster spec to apply next time we reinitialize
    # Note: We can't apply the new cluster spec right away because doing so will
    # signal to other workers that they can go ahead and build their graphs, which
    # might be inconsistent with ours if we haven't finished reinitializing yet.
    # Accesses must be guarded by `self.pending_cluster_spec_lock`.
    self.pending_cluster_spec = None
    self.pending_cluster_spec_lock = threading.Lock()    

    # Start autoscaling server
    listen_for_requests(self, convert_port(self.host_port))

    # Start autoscaling client, connected to the autoscaling server on the first worker
    first_worker = tf_config["cluster"]["worker"][0]
    first_worker = convert_port(first_worker)
    self.client = AutoscalingClient(first_worker)

    # Spawned workers join after running one batch by themselves
    # This avoids blocking existing workers while the spawned workers are starting up
    self.joined = AUTOSCALING_MASTER_HOST_PORT not in os.environ

    # Parse host flag to get a list of all possible hosts
    if self.joined and self.mpi_communicator.rank == 0:
      with self.spawn_lock:
        flag_name, hosts_with_slots = os.environ["HOST_FLAG"].split(" ")
        if "hostfile" in flag_name:
          raise ValueError("--hostfile is currently not supported")
        if ":" in hosts_with_slots:
          self.all_possible_hosts = [h.split(":")[0] for h in hosts_with_slots.split(",")]
          self.max_slots = [int(h.split(":")[1]) for h in hosts_with_slots.split(",")]
        else:
          # There are no slots provided
          self.all_possible_hosts = hosts_with_slots.split(",")
          self.max_slots = [1] * len(self.all_possible_hosts)
        for hp in self.cluster_spec["worker"]:
          host, _ = hp.split(":")
          if host not in self.host_counts:
            self.host_counts[host] = 0
          self.host_counts[host] += 1

  @property
  def status(self):
    """
    Atomic getter for `self._status`, guarded by `self._status_lock`.
    """
    with self._status_lock:
      return self._status

  @status.setter
  def status(self, s):
    """
    Atomic setter for `self._status`, guarded by `self._status_lock`.
    """
    with self._status_lock:
      if not isinstance(s, AutoscalingStatus):
        raise ValueError("'%s' is not an AutoscalingStatus" % s)
      self._status = s

  def initialize(self):
    """
    Ensure everyone sees the same cluster spec, then set TF_CONFIG accordingly.
    This should be called on start up and after every time we restart.
    """
    log_fn("Initializing")
    self.mpi_communicator.barrier()

    # Check if cluster membership changed. If so, update cluster spec accordingly.
    self.status = AutoscalingStatus.READY_TO_SYNC
    self.mpi_communicator.barrier()
    with self.pending_cluster_spec_lock:
      if self.pending_cluster_spec is not None:
        self.apply_cluster_spec(self.pending_cluster_spec)
        self.pending_cluster_spec = None

    # Expand our communicator to include the provided ranks, if any
    if self.joined:
      with self.spawn_lock:
        new_ranks = []
        if len(self.spawned_ranks_to_wait_for) > 0:
          new_ranks = self.spawned_ranks_to_wait_for.pop(0)
        self.maybe_expand_mpi_communicator(new_ranks)

    # Sync cluster spec and update relevant variables
    self.sync_cluster_spec()
    self.task_index = self.cluster_spec["worker"].index(self.host_port)

    # Transfer saved model parameters to new workers
    # Here we use allgather instead of gather in case there are multiple workers joining
    if self.joined:
      fetch_start = time.time()
      all_saved_variables = self.mpi_communicator.allgather(self.saved_variables)
      if self.saved_variables is None:
        self.saved_variables = {}
        for var_map in all_saved_variables:
          # Ignore the values from new workers, including ourselves
          if var_map is not None:
            self.saved_variables.update(var_map)
        log_fn("Fetched %s variables, took %s seconds" %\
          (len(self.saved_variables), time.time() - fetch_start))

    # Set CUDA_VISIBLE_DEVICES, assigned by master
    # This assumes CUDA_VISIBLE_DEVICES assignment will never change
    if self.num_gpus_per_worker > 0 and self.host_port not in self.cuda_visible_devices_map:
      if self.joined:
        cuda_visible_devices = self.client.master_server_rpc(
          lambda s: s.assign_cuda_visible_devices(self.host_port))
      else:
        # Note: If we haven't joined yet, our client is not aware of the real
        # master server yet, so we cannot use our client here
        cuda_visible_devices = connect(os.environ[AUTOSCALING_MASTER_HOST_PORT])\
          .assign_cuda_visible_devices(self.host_port)
      self.cuda_visible_devices_map[self.host_port] = cuda_visible_devices
      cuda_visible_devices = ",".join([str(d) for d in cuda_visible_devices])
      os.environ[CUDA_VISIBLE_DEVICES] = cuda_visible_devices
      log_fn("Set CUDA_VISIBLE_DEVICES = %s" % cuda_visible_devices)

    # Compute new batch sizes
    # In autoscaling mode, the global batch size can change
    if is_autoscaling_mode():
      local_batch_size = self.max_local_batch_size
      global_batch_size = local_batch_size * self.mpi_communicator.size
      global_batch_size = min(global_batch_size, self.max_global_batch_size)
      self.global_batch_size = global_batch_size
    # Make sure everyone has the same global batch size
    self.global_batch_size = self.mpi_communicator.bcast(self.global_batch_size, root=0)
    # Everyone will then independently tell tensorflow what their local batch sizes are
    local_batch_size = autoscaling_helper.local_batch_size(
      self.global_batch_size, self.mpi_communicator.size, self.mpi_communicator.rank)
    local_batch_size = int(local_batch_size * self.local_batch_size_multiplier)
    autoscaling_helper.LOCAL_BATCH_SIZE = local_batch_size
    log_fn("Set global batch size = %s, local batch size = %s" %\
      (self.global_batch_size, local_batch_size))

    self.status = AutoscalingStatus.RUNNING
    self.mpi_communicator.barrier()

    # If we are the master, spawn the remaining workers
    if not self.cluster_initialized and self.joined and self.mpi_communicator.rank == 0:
      num_remaining_workers = self.autoscaling_initial_workers - 1
      if num_remaining_workers > 0:
        self.mpi_spawn_workers(num_remaining_workers)

  def maybe_expand_mpi_communicator(self, ranks):
    """
    Merge newly spawned workers, if any, into our existing communicator.

    This method must be called at the same time on all the processes participating
    in the final communicator. The process is driven by the root, where `ranks`
    specifies the specific communicators to merge into the existing one. This value
    is not read on other processes.

    Must be called while holding `self.spawn_lock`.
    """
    # First, figure out our role
    comm = self.mpi_communicator
    is_joining = comm.rank == 0 and AUTOSCALING_MASTER_HOST_PORT in os.environ
    is_root = comm.rank == 0 and not is_joining
    # If we're joining, then expand once first
    if is_joining:
      comm = mpi_helper.expand(comm, MPI.Comm.Get_parent())
    # Note: MPI only allows us to expand once at a time, so we will do so in a loop.
    # In each iteration, the root will broadcast whether there are still spawned
    # communicators waiting to join. If so, all members of the existing communicator
    # will participate in a round of expansion.
    while True:
      should_expand = len(ranks) > 0 if is_root else False
      should_expand = comm.bcast(should_expand, root=0)
      if not should_expand:
        break
      if is_root:
        spawned_rank = ranks.pop(0)
        if spawned_rank not in self.mpi_spawned_communicators:
          raise ValueError("Attempted to expand with unknown rank %s" % spawned_rank)
        spawned_comm = self.mpi_spawned_communicators[spawned_rank]
        comm = mpi_helper.expand(comm, spawned_comm)
      else:
        comm = mpi_helper.expand(comm)
    self.mpi_communicator = comm

  def mpi_spawn_workers(self, num_workers):
    """
    Spawn one or more worker processes through MPI.

    The spawned workers, if any, are added to `self.mpi_spawned_communicators`.
    This succeeds only if called on the root while the root is running batches.
    Return whether spawn was attempted.
    """
    if self.mpi_communicator.rank > 0:
      raise ValueError("Only the root can spawn workers")
    if self.status != AutoscalingStatus.RUNNING:
      log_fn("Not spawning worker because we are initializing")
      return False
    # If there were previously removed workers, just ask them to rejoin instead
    # of spawning new ones
    spawn_ranks = []
    while len(self.removed_host_ports) > 0 and num_workers > 0:
      removed_host_port = self.removed_host_ports.pop(0)
      log_fn("Asking worker %s to rejoin cluster" % removed_host_port)
      spawn_ranks.append(connect(convert_port(removed_host_port)).request_attach())
      num_workers -= 1
    # Spawn the remaining workers through MPI as new processes
    if num_workers > 0:
      log_fn("Spawning %s worker(s) through MPI" % num_workers)
    with self.spawn_lock:
      starting_rank = self.mpi_next_rank
      spawn_ranks.extend(list(range(starting_rank, starting_rank + num_workers)))
      self.spawned_ranks_to_wait_for.append(spawn_ranks)
      self.mpi_next_rank += num_workers
    def do_spawn(new_worker_rank):
      # Figure out which host to launch this node on
      target_host = None
      if os.getenv("MPI_MAP_BY", "") == "node":
        with self.spawn_lock:
          possible_hosts_index = new_worker_rank % len(self.all_possible_hosts)
          target_host = self.all_possible_hosts[possible_hosts_index]
          if target_host not in self.host_counts:
            self.host_counts[target_host] = 0
          self.host_counts[target_host] += 1
          if self.host_counts[target_host] > self.max_slots[possible_hosts_index]:
            raise ValueError("Unable to spawn host on %s because we've reached the max slot: %s" %\
              (target_host, self.host_counts))
      # Set other environment variables
      env = {
        AUTOSCALING_MASTER_HOST_PORT: self.client.master_host_port,
      }
      spawned_communicator = mpi_helper.spawn(new_worker_rank, target_host=target_host, env=env)
      with self.spawn_lock:
        self.mpi_spawned_communicators[new_worker_rank] = spawned_communicator
    # Spawn asynchronously in parallel; each call to MPI_SPAWN takes about 3 seconds
    # However, if we spawn too many at the same time, we may run into seg faults
    # from MPI, so we need to spawn moderately in batches
    spawn_threads = []
    spawn_limit = 10
    for r in spawn_ranks:
      if r >= starting_rank:
        t = threading.Thread(target=do_spawn, args=[r])
        t.start()
        spawn_threads.append(t)
        if len(spawn_threads) >= spawn_limit:
          for tt in spawn_threads:
            tt.join()
          spawn_threads.clear()
    return True

  def num_expected_workers(self):
    """
    Return the number of expected workers after all spawned workers have joined.
    """
    with self.spawn_lock:
      # Flatten
      num_pending_workers = len(np.hstack(self.spawned_ranks_to_wait_for or [[]]))
      return self.mpi_communicator.size + num_pending_workers

  def join_cluster(self):
    """
    If we are not part of the original cluster, request to join it.
    This fails if the master server is initializing, in which case we keep retrying.
    """
    if self.joined:
      raise ValueError("Already joined cluster!")
    master_host_port = os.getenv(AUTOSCALING_MASTER_HOST_PORT)
    if master_host_port is None:
      raise ValueError("AUTOSCALING_MASTER_HOST_PORT not set on spawned worker")
    # Our client currently thinks we are the master, so we need to reset it
    self.client.reset(master_host_port)
    # Wait until master is ready to accept our join request
    my_rank = int(os.environ[mpi_helper.MPI_SPAWN_RANK])
    log_fn("Joining cluster as %s (rank %s)" % (self.host_port, my_rank))
    while not self.client.master_server_rpc(lambda s: s.join_cluster(self.host_port, my_rank)):
      log_fn("Master server is not ready to service our join request, trying again later.")
      time.sleep(AUTOSCALING_JOIN_RETRY_INTERVAL_SECONDS)
    log_fn("Master server accepted our join request")
    self.joined = True

  def detach_from_cluster(self):
    """
    Transition into detached mode where this process runs everything by itself.
    """
    self.detached_mode = True
    self.saved_variables = None
    self.task_index = 0
    self.cluster_spec = {"worker": [self.host_port]}
    with self.pending_cluster_spec_lock:
      self.pending_cluster_spec = None
    self.mpi_communicator = MPI.COMM_SELF
    self.local_batch_size_multiplier = 1
    self.client.reset(convert_port(self.host_port))
    self.joined = False

  def sync_cluster_spec(self):
    """
    Retry until all the following are true
      (1) Our cluster spec contains our host name
      (2) We have the same cluster spec as everyone else
      (3) Everyone is SYNCED

    Our status must be READY_TO_SYNC before we call this method, and SYNCED when we return.
    """
    log_fn("Attempting to sync cluster spec with everyone")
    self.status = AutoscalingStatus.SYNCING
    self.mpi_communicator.barrier()
    while True:
      failure_message = None
      my_cluster_spec = json.dumps(self.client.cluster_spec)
      # (1) Does our cluster spec contain our host name?
      if self.host_port not in self.client.hosts:
        failure_message = "cluster spec does not contain this host (%s)" % self.host_port
        _ = self.mpi_communicator.allgather(None)
      else:
        # (2) Do we have the same cluster spec as everyone else?
        for cluster_spec in self.mpi_communicator.allgather(json.dumps(self.cluster_spec)):
          if cluster_spec != my_cluster_spec:
            failure_message = "cluster spec sync failed"
      synced = all(self.mpi_communicator.allgather(failure_message is None))
      if synced:
        log_fn("... cluster spec synced: %s" % my_cluster_spec)
        self.status = AutoscalingStatus.SYNCED
        self.mpi_communicator.barrier()
        return
      # On failure, reset client with cluster spec from the master autoscaling server
      failure_message = failure_message or "not everyone is synced"
      log_fn("... %s, trying again in %s second(s)" %\
        (failure_message, AUTOSCALING_RETRY_INTERVAL_SECONDS))
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
      master_cluster_spec = self.mpi_communicator.bcast(self.cluster_spec, root=0)
      self.apply_cluster_spec(master_cluster_spec)
      self.client.reset(new_cluster_spec=master_cluster_spec)

  def apply_cluster_spec(self, cluster_spec):
    """
    Update our fields to match the provided cluster spec (python dict).
    """
    self.cluster_spec = cluster_spec
    tasks = cluster_spec[self.task_type]
    if self.host_port in tasks:
      self.task_index = tasks.index(self.host_port)

  def save_variables(self, variables, session=None, for_new_worker=False):
    """
    Save the values of the given variables to memory.

    If `session` is provided, use it to compute the value of the variables.
    Otherwise, we assume this is running in eager mode, in which case we can
    just directly access the values of the variables.

    If `for_new_worker` is true, then we will only save the slice of
    variables this rank is assigned, and the values will be compressed.
    """
    save_start = time.time()
    # Find the variables in the slice assigned to this process
    if for_new_worker:
      num_chunks = self.mpi_communicator.size
      chunk_index = self.mpi_communicator.rank
      all_indexes = np.array_split(np.array(range(len(variables))), num_chunks)
      my_indexes = all_indexes[chunk_index].tolist()
      new_variables = []
      for i in my_indexes:
        new_variables.append(variables[i])
      variables = new_variables
    # Compute the values for these variables
    if session is not None:
      session.graph._finalized = False
      values = session.run(variables)
    else:
      # We're running in eager mode
      values = [v.value() for v in variables]
    save_end = time.time()
    self.saved_variables = {}
    for i, v in enumerate(variables):
      value = values[i]
      if for_new_worker:
        # If this is for new workers, compress it so we can send less data
        buf = io.BytesIO()
        np.savez_compressed(buf, value=value)
        value = buf.getvalue()
      self.saved_variables[v.name] = value
    log_fn("Saved %s variables in memory, took %s seconds" %\
      (len(self.saved_variables), save_end - save_start))

  def restore_variables(self, variables, session=None):
    """
    Restore the values of saved variables from memory, if any.
    This assumes `saved_variables` is not None.

    If `session` is provided, use it to restore the values of the variables.
    Otherwise, we assume this is running in eager mode, in which case we can
    just directly update the variables.
    """
    try:
      if len(self.saved_variables) != len(variables):
        raise ValueError("Number of saved variables (%s) != number of variables provided (%s)" %\
          (len(self.saved_variables), len(variables)))
      log_fn("Restoring %s variables from memory" % len(variables))
      restore_ops = []
      restore_start = time.time()
      for var in variables:
        val = self.saved_variables[var.name]
        # Value may be compressed, in which case it will be in bytes
        if isinstance(val, xmlrpc.client.Binary):
          val = val.data
        if isinstance(val, bytes):
          data = np.load(io.BytesIO(val))
          if len(data.files) != 1:
            raise ValueError("Unknown compression format")
          val = data[data.files[0]]
        update_fn = lambda var, val: var.assign(val)
        restore_ops.append(
          tf.distribute.get_strategy().extended.update(var, update_fn, args=(val,)))
      if session is not None:
        session.graph._finalized = False
        session.run(restore_ops)
      restore_end = time.time()
      log_fn("Restored %s variables from memory, took %s seconds" %\
        (len(variables), restore_end - restore_start))
    finally:
      self.saved_variables = None

  def step_end(self):
    """
    Listen for changes in cluster membership and react by reinitializing the server.

    This method is called at the end of each step and involves two synchronization
    barriers: one to see whether everyone has seen the new pending cluster spec,
    and another to see whether everyone has observed that everyone has seen it.
    This guarantees that all workers make the same decision on whether to restart.

    Return whether this process should reinitialize. New workers should
    reinitialize after the first step and existing workers should reinitialize
    if there are new workers and everyone is aware of their existence.
    """
    # Wait for the initial set of workers to join
    if not self.cluster_initialized and self.joined and self.mpi_communicator.rank == 0:
      self.cluster_initialized = True
      num_remaining_workers = self.autoscaling_initial_workers - 1
      if num_remaining_workers > 0:
        log_fn("Waiting for %s workers to join" % num_remaining_workers)
        num_workers_joined = 0
        while num_remaining_workers > num_workers_joined:
          with self.pending_cluster_spec_lock:
            if self.pending_cluster_spec is not None:
              num_workers_joined =\
                len(set(self.pending_cluster_spec["worker"]) - set(self.cluster_spec["worker"]))
          time.sleep(AUTOSCALING_JOIN_RETRY_INTERVAL_SECONDS)
        return True
    # If we're in detached mode, keep looping until master notifies us to join
    if self.detached_mode:
      log_fn("Waiting for signal from master in detached mode")
      while self.detached_mode:
        time.sleep(AUTOSCALING_JOIN_RETRY_INTERVAL_SECONDS)
    # If this is a spawned worker, join the cluster after the first step
    if not self.joined:
      self.join_cluster()
      self.num_steps_since_last_restart = 0
      self.status = AutoscalingStatus.RESTARTING
      return True
    # We need to restart if
    # (1) there is a pending cluster spec,
    # (2) all new workers previously spawned, if any, have joined, and
    # (3) we last restarted more than N steps ago
    should_restart = False
    has_pending = False
    num_workers_joined = 0
    num_workers_expected = 0
    with self.pending_cluster_spec_lock:
      if self.pending_cluster_spec is not None:
        has_pending = True
        num_workers_joined =\
          len(set(self.pending_cluster_spec["worker"]) - set(self.cluster_spec["worker"]))
    if has_pending:
      with self.spawn_lock:
        if len(self.spawned_ranks_to_wait_for) > 0:
          num_workers_expected = len(self.spawned_ranks_to_wait_for[0])
      if num_workers_expected > 0:
        log_fn("%s/%s spawned worker(s) have joined" % (num_workers_joined, num_workers_expected))
    self.num_steps_since_last_restart += 1
    should_restart = has_pending and\
      num_workers_joined >= num_workers_expected and\
      self.num_steps_since_last_restart >= self.min_steps_between_restart
    # Do a two-phase synchronization to make sure that everyone agrees on whether
    # or not to restart.
    ready_to_restart = all(self.mpi_communicator.allgather(should_restart))
    restarting = all(self.mpi_communicator.allgather(ready_to_restart))
    if restarting:
      # If the new cluster spec does not contain us, then that means we are removed
      should_terminate = False
      with self.pending_cluster_spec_lock:
        should_terminate = self.host_port not in self.pending_cluster_spec["worker"]
      if should_terminate:
        log_fn("Received signal to terminate")
        self.status = AutoscalingStatus.TERMINATED
      else:
        log_fn("Received signal to restart server")
        self.status = AutoscalingStatus.RESTARTING
      self.num_steps_since_last_restart = 0
      self.remove_terminated_workers()
    return restarting

  def remove_terminated_workers(self):
    """
    Remove workers that are TERMINATED, if any, from our communicator.

    Note: This must be called *before* the terminated workers exit.
    """
    terminated_ranks = []
    gathered = self.mpi_communicator.allgather((self.status, self.host_port))
    for r, (status, host_port) in enumerate(gathered):
      if status == AutoscalingStatus.TERMINATED:
        terminated_ranks.append(r)
        if self.mpi_communicator.rank == 0:
          # Optionally remember the removed server in case we need it again
          if self.detach_when_removed:
            self.removed_host_ports.append(host_port)
          else:
            host, _ = host_port.split(":")
            if host in self.host_counts:
              del self.host_counts[host]
            else:
              log_fn("Warning: expected %s to be in host counts: %s" % (host, self.host_counts))
            if host_port in self.cuda_visible_devices_map:
              del self.cuda_visible_devices_map[host_port]
    if len(terminated_ranks) > 0 and self.status != AutoscalingStatus.TERMINATED:
      new_group = self.mpi_communicator.group.Excl(terminated_ranks)
      self.mpi_communicator = self.mpi_communicator.Create_group(new_group)
      log_fn("Removed ranks %s from communicator, new size = %s" %\
        (terminated_ranks, self.mpi_communicator.size))

# ================== HELPER METHODS ==================

def log_fn(msg):
  tf.logging.info("[Autoscaling agent] %s" % msg)

def get_tf_config():
  """
  Return TF_CONFIG as a dictionary, assuming it was set.
  """
  tf_config = os.getenv("TF_CONFIG")
  if tf_config is None:
    raise ValueError("'TF_CONFIG' must be set.")
  return json.loads(tf_config.replace("'", "\""))

def log_exceptions(fn):
  """
  Do not swallow exceptions.
  """
  try:
    fn()
  except Exception as e:
    log_fn("%s (%s)" % (e, e.__class__.__name__))
    traceback.print_exc()
    raise e

