#!/usr/bin/env python3

import json
import math
import os
import socket
import threading
import time
import xmlrpc.server

from absl import logging
from mpi4py import MPI
import numpy as np
import tensorflow as tf

from virtual import virtual_helper


# Constants and environment variables
RETRY_INTERVAL_SECONDS = 1
BASE_ELASTICITY_PORT = 17272
ELASTICITY_VERBOSE = os.getenv("ELASTICITY_VERBOSE", "").lower() == "true"
ENABLE_ELASTICITY = os.getenv("ENABLE_ELASTICITY", "").lower() == "true"
ELASTICITY_MASTER = "ELASTICITY_MASTER"
SCHEDULER_ADDR = "SCHEDULER_ADDR"
GPU_BLACKLIST_VALUE = -1
NUM_GPUS_PER_NODE = "NUM_GPUS_PER_NODE"
SPAWN_RANK = "SPAWN_RANK"
JOB_ID = "JOB_ID"

# Elasticity state passed to tensorflow
# Setting these will signal tensorflow to rebuild the graph
NUM_VIRTUAL_NODES = None
START_BATCH = None
START_EPOCH = None

# Singleton elasticity callback, defined only if ENABLE_ELASTICITY is true
ELASTICITY_CALLBACK = None

def initialize_singleton_callback():
  """
  Initialize a singleton elasticity callback for this program.

  Initializing the elasticity callback at the beginning of the program allows the
  master to spawn the rest of the initial set of workers early on, which reduces
  the start up delay.
  """
  global ELASTICITY_CALLBACK
  ELASTICITY_CALLBACK = ElasticityCallback()

def get_elasticity_client(host, job_id=0):
  """
  Return a client that can communicate with the elasticity server on the master.
  """
  return xmlrpc.client.ServerProxy("http://%s:%s" % (host, BASE_ELASTICITY_PORT + job_id))

def assign_gpus(n, gpu_availability, all_possible_hosts=None):
  """
  Assign the specified number of GPUs based on the current availability.

  `gpu_availability` is a map from host to a list, where each entry of the list
  is None if the GPU is available to be assigned. We assume `gpu_availability`
  contains all the hosts in `all_possible_hosts`, if it is defined. This function
  does not mutate any state.

  Return a list of 2-tuples (host, gpu_index), one for each GPU assigned.
  """
  assigned_gpus = []
  host_index = 0
  gpu_index = 0
  if all_possible_hosts is None:
    all_possible_hosts = list(gpu_availability.keys())
  # Fill up GPUs within a host first
  while len(assigned_gpus) < n and host_index < len(all_possible_hosts):
    host = all_possible_hosts[host_index]
    gpus = gpu_availability[host]
    if gpu_index < len(gpus):
      if gpus[gpu_index] is None:
        # Found a free GPU, assign it
        assigned_gpus.append((host, gpu_index))
      gpu_index += 1
    else:
      # If a host does not have any more free GPUs, move onto the next
      gpu_index = 0
      host_index += 1
  assert(len(assigned_gpus) <= n)
  return assigned_gpus

class ElasticityCallback(tf.keras.callbacks.Callback):
  """
  A callback that maintains elasticity state for this process.
  """
  def __init__(self):
    logging.info("Initializing elasticity callback")
    self.comm = MPI.COMM_WORLD.Dup()
    self.is_joining = ELASTICITY_MASTER in os.environ
    self.is_master = virtual_helper.is_master(self.comm) and not self.is_joining
    self.current_batch = 0
    self.current_epoch = 0
    self.job_id = int(os.getenv(JOB_ID, 0))
    num_nodes = int(os.environ[virtual_helper.NUM_NODES])

    # Number of total virtual nodes across all devices
    self.num_total_virtual_nodes =\
      int(os.getenv(virtual_helper.NUM_VIRTUAL_NODES_PER_DEVICE, 1)) * num_nodes

    # List of all hosts in our communicator
    self.members = self.comm.allgather(MPI.Get_processor_name())

    if self.is_master:
      # If set, the next batch will resize the cluster accordingly
      self.new_size = None

      # Number of GPUs per node, assumed to be the same across all nodes
      self.num_gpus_per_node = int(os.getenv(NUM_GPUS_PER_NODE, 1))

      # Map from hostname to a list of size `self.num_gpus_per_node`, where each item in the
      # list the rank of the process the GPU is assigned to, or None if the GPU is not assigned
      self.cuda_visible_devices_map = {}
      for host in virtual_helper.get_all_mpi_hosts():
        self.cuda_visible_devices_map[host] = [None] * self.num_gpus_per_node

      # If running in a shared cluster, communicate with the scheduler for GPU assignment
      from deploy.scheduler import get_scheduler_client
      scheduler_addr = os.getenv(SCHEDULER_ADDR)
      if scheduler_addr is not None:
        self.scheduler_client = get_scheduler_client(scheduler_addr)
      else:
        self.scheduler_client = None

      # Assign a GPU to this process
      # All other GPUs are assigned at spawn time
      master_host = MPI.Get_processor_name()
      assigned_gpus = self.assign_gpus(1, all_possible_hosts=[master_host])
      if len(assigned_gpus) != 1:
        raise ValueError("Unable to assign the first GPU for the master")
      _, master_gpu_index = assigned_gpus[0]
      os.environ[virtual_helper.CUDA_VISIBLE_DEVICES] = str(master_gpu_index)
      self.cuda_visible_devices_map[master_host][master_gpu_index] = 0
      logging.info("Setting CUDA_VISIBLE_DEVICES to %s" % master_gpu_index)

      # A list of communicators returned from the last call to MPI spawn, if any
      self.spawned_communicators = []

      # A list of ranks that were spawned and are now ready to join the cluster
      self.ready_to_join_ranks = []

      # Listen for elasticity requests from the user
      server = xmlrpc.server.SimpleXMLRPCServer(
        (socket.gethostname(), BASE_ELASTICITY_PORT + self.job_id),
        logRequests=False, allow_none=True)
      server.register_function(self.get_num_workers)
      server.register_function(self.set_num_workers)
      server.register_function(self.spawn)
      server.register_function(self.handle_join)
      t = threading.Thread(target=server.serve_forever)
      t.setDaemon(True)
      t.start()

      # In elasticity mode, we always start with a single worker and let it
      # spawn the remaining workers so the fates of the workers are not tied
      self.awaiting_initial_workers = num_nodes > 1
      self.spawn(num_nodes - 1)

  def assign_gpus(self, n, all_possible_hosts=None):
    """
    Assign the specified number of GPUs based on the current availability.

    If running in a shared cluster, we first request the latest GPU availability
    from the scheduler and update our own mapping accordingly.

    Return a list of 2-tuples (host, gpu_index), one for each GPU assigned.
    """
    if not self.is_master:
      raise ValueError("Only the master can assign GPUs")
    # Request the latest GPU availability from the scheduler, if applicable
    if self.scheduler_client is not None:
      assigned_gpus = self.scheduler_client.get_assigned_gpus(self.job_id)
      # Confirm the clusters are the same
      assert(set(assigned_gpus.keys()) == set(self.cuda_visible_devices_map.keys()))
      for host in assigned_gpus.keys():
        assert(len(assigned_gpus[host]) == len(self.cuda_visible_devices_map[host]))
        for i, j in enumerate(assigned_gpus[host]):
          # Blacklist entries in our own mapping
          if j == GPU_BLACKLIST_VALUE:
            self.cuda_visible_devices_map[host][i] = GPU_BLACKLIST_VALUE
          # If a GPU was previously blacklisted but now assigned to this job,
          # mark the GPU as available
          if j == self.job_id and self.cuda_visible_devices_map[host][i] == GPU_BLACKLIST_VALUE:
            self.cuda_visible_devices_map[host][i] = None
    return assign_gpus(n, self.cuda_visible_devices_map, all_possible_hosts)

  def get_num_workers(self):
    """
    Return the current number of workers in this job.
    This should only called on the master.
    """
    if not self.is_master:
      raise ValueError("Only the master accepts cluster size queries")
    return self.comm.size

  def set_num_workers(self, num_workers):
    """
    Resize the cluster to the given value, which must be a multiple of 2.
    This should only called on the master.
    """
    self._check_master_request("resize")
    if not math.log2(num_workers).is_integer():
      raise ValueError("Num workers must be a power of 2 (was %s)" % num_workers)
    if num_workers == self.comm.size:
      return
    if num_workers > self.comm.size:
      self.spawn(num_workers - self.comm.size)
    else:
      self.new_size = num_workers

  def spawn(self, n):
    """
    Spawn processes through MPI.

    This should only be called on the master, and can only be called when there are
    no outstanding spawned processes.
    """
    self._check_master_request("spawn")
    assigned_gpus = self.assign_gpus(n)
    if len(assigned_gpus) < n:
      logging.warn("Not enough hosts to spawn %s more processes, ignoring spawn request" % n)
      return
    # Spawn processes in parallel because MPI spawn takes a few seconds each
    spawn_threads = []
    max_spawn_parallel = 16
    self.spawned_communicators = [None] * n
    for i, (host, gpu_index) in enumerate(assigned_gpus):
      rank = self.comm.size + i
      # Inform the spawned worker of a way to contact us, its rank, and its assigned GPUs
      env = {
        ELASTICITY_MASTER: MPI.Get_processor_name(),
        SPAWN_RANK: rank,
        virtual_helper.CUDA_VISIBLE_DEVICES: gpu_index
      }
      self.cuda_visible_devices_map[host][gpu_index] = rank
      def do_spawn(host, env, index):
        self.spawned_communicators[index] = virtual_helper.mpi_spawn(host, env)
      t = threading.Thread(target=do_spawn, args=(host, env, i))
      t.setDaemon(True)
      t.start()
      spawn_threads.append(t)
      # Limit the number of parallel spawns to avoid MPI errors
      if (i+1) % max_spawn_parallel == 0:
        for t in spawn_threads:
          t.join()
        spawn_threads = []
    for t in spawn_threads:
      t.join()


  def handle_join(self, rank):
    """
    Notify the master that a spawned rank is ready to join the cluster.

    This should only be called on the master client.
    Return the expected size of the new cluster.
    """
    self._check_master_request("join")
    new_size = self.comm.size + len(self.spawned_communicators)
    if rank in self.ready_to_join_ranks:
      logging.warn("Already received join request from rank %s, ignoring" % rank)
      return new_size
    self.ready_to_join_ranks.append(rank)
    # If all spawned workers are ready, trigger resize
    if len(self.ready_to_join_ranks) == len(self.spawned_communicators):
      self.new_size = new_size
      self.ready_to_join_ranks = []
    return new_size

  def _check_master_request(self, action):
    """
    Helper method to check if requests received at the master client are valid.
    """
    if not self.is_master:
      raise ValueError("Only the master can handle %s requests" % action)
    if self.new_size is not None:
      raise ValueError("Existing resizing request in progress, not accepting further requests")
    if action == "join":
      if len(self.spawned_communicators) == 0:
        raise ValueError("Unexpected join request: there was no spawned communicator")
    else:
      if len(self.spawned_communicators) > 0:
        raise ValueError("Existing spawn request in progress, not accepting further requests")

  def transition(self, new_size):
    """
    Transition to a new cluster.

    This method adjusts the MPI communicator and informs tensorflow to rebuild the graph
    with a new horovod allreduce function. Workers that are removed will exit.
    """
    logging.info("Transitioning to new cluster of size %s" % new_size)

    # If this worker is removed, wait to receive a rejoin request from the master
    if self.comm.rank >= new_size:
      logging.info("Leaving cluster because our rank is %s and the new size is %s" %\
        (self.comm.rank, new_size))
      # Wait until all existing workers have reinitialized horovod before exiting
      # Exiting early will cause MPI to kill the entire job
      self.comm.barrier()
      os._exit(0)

    old_comm = self.comm
    old_size = self.comm.size
    released_gpus = []
    if new_size > old_size:
      # If we are expanding the cluster, merge the old and new communicators
      logging.info("Expanding communicator from size %s to %s" % (self.comm.size, new_size))
      if self.is_joining:
        self.comm = virtual_helper.mpi_expand(self.comm, MPI.Comm.Get_parent())
      # Note: MPI only allows us to expand once at a time, so we will do so in a loop.
      # In each iteration, the master will broadcast whether there are still spawned
      # communicators waiting to join. If so, all members of the existing communicator
      # will participate in a round of expansion.
      while True:
        should_expand = len(self.spawned_communicators) > 0 if self.is_master else False
        should_expand = self.comm.bcast(should_expand, root=0)
        if not should_expand:
          break
        if self.is_master:
          intercomm = self.spawned_communicators.pop(0)
        else:
          intercomm = None
        self.comm = virtual_helper.mpi_expand(self.comm, intercomm)
      if self.comm.size != new_size:
        raise ValueError("New communicator size after expanding was %s != expected %s" %\
          (self.comm.size, new_size))
    else:
      # Otherwise, we are shrinking the cluster, so just use a subset of ranks
      new_group = self.comm.group.Incl(list(range(new_size)))
      self.comm = self.comm.Create_group(new_group)
      # On the master, remove corresponding entries in our CUDA_VISIBLE_DEVICES mapping
      if self.is_master:
        for removed_rank in list(range(new_size, old_size)):
          host = self.members[removed_rank]
          for i, r in enumerate(self.cuda_visible_devices_map[host]):
            if r == removed_rank:
              self.cuda_visible_devices_map[host][i] = None
              released_gpus.append((host, i))

    # Update some state using the new communicator
    self.members = self.comm.allgather(MPI.Get_processor_name())
    old_size = self.comm.bcast(old_size, root=0)

    # Reinitialize horovod to use the new communicator
    # We can release the removed workers from the old communicator after doing this
    virtual_helper.initialize_horovod(self.comm)
    old_comm.barrier()

    # Transfer parameters from existing workers to new workers if the cluster expanded
    self.transfer_parameters(old_size, new_size)

    # Trigger tensorflow to update the graph with the correct number of virtual nodes,
    # step number, and epoch number
    global START_BATCH
    global START_EPOCH
    global NUM_VIRTUAL_NODES
    self.current_batch, self.current_epoch, self.num_total_virtual_nodes =\
      self.comm.bcast((self.current_batch, self.current_epoch, self.num_total_virtual_nodes), root=0)
    START_BATCH = self.current_batch
    START_EPOCH = self.current_epoch
    NUM_VIRTUAL_NODES = self.num_total_virtual_nodes / self.comm.size
    if not NUM_VIRTUAL_NODES.is_integer():
      raise ValueError("Num virtual nodes must be an integer! (was %s)" % NUM_VIRTUAL_NODES)
    NUM_VIRTUAL_NODES = int(NUM_VIRTUAL_NODES)
    self.new_size = None

    # Notify the scheduler that our transition is complete, if applicable
    if self.is_master and self.scheduler_client is not None:
      self.scheduler_client.notify_transition(self.job_id, self.comm.size, released_gpus)

  def transfer_parameters(self, old_size, new_size):
    """
    During transition, transfer parameters from existing workers to new workers.
    """
    if old_size >= new_size:
      return

    if self.model is None:
      raise ValueError("self.model was not set!")

    self.log_parameters("[Before transfer] ", old_size)

    # Split parameters into N slices, where N is the number of existing workers,
    # and perform an allgather to send these slices to the new workers
    bcast_parameters = None
    sending_ranks = list(range(old_size))
    if self.comm.rank in sending_ranks:
      bcast_parameters = [v.value() for v in np.array_split(\
        self.model.trainable_variables, old_size)[self.comm.rank]]
    bcast_parameters = self.comm.allgather(bcast_parameters)

    # Update model on receiving ranks
    if self.comm.rank not in sending_ranks:
      received_parameters = []
      for p in bcast_parameters:
        if p is not None:
          received_parameters.extend(p)
      assert(len(received_parameters) == len(self.model.trainable_variables))
      tf.python.keras.backend.batch_set_value(\
        list(zip(self.model.trainable_variables, received_parameters)))

    self.log_parameters("[After transfer] ", old_size)

  def log_parameters(self, prefix, num_chunks, n=10):
    """
    Helper method to print the first values of the first N parameters in each group.
    """
    if ELASTICITY_VERBOSE:
      reshaped = [float("%.6g" % tf.reshape(v, [-1])[:1].numpy().tolist()[0])\
        for v in self.model.trainable_variables]
      reshaped = [p[:n].tolist() for p in np.array_split(np.array(reshaped), num_chunks)]
      reshaped = ",\n".join(["  " + str(p) for p in reshaped])
      logging.info("%sThe first values of the first %s parameters in each group are:\n%s" %\
        (prefix, n, reshaped))

  def on_train_begin(self, logs=None):
    global NUM_VIRTUAL_NODES
    NUM_VIRTUAL_NODES = int(self.num_total_virtual_nodes / self.comm.size)

  def on_batch_begin(self, batch, logs=None):
    """
    Check if we need to resize the cluster and, if so, transition to the new cluster.
    """
    self.current_batch = batch
    # New workers should run a batch before joining the cluster
    # This significantly shortens the GPU idle time during transitions because the existing
    # workers do not have to wait for the new workers to bootstrap, which can take minutes.
    if self.is_joining and batch == 1:
      self.comm.barrier()
      new_size = None
      # Notify the master the new workers are ready to join the cluster
      if self.comm.rank == 0:
        master_client = get_elasticity_client(os.environ[ELASTICITY_MASTER], self.job_id)
        new_size = master_client.handle_join(int(os.environ[SPAWN_RANK]))
      new_size = self.comm.bcast(new_size, root=0)
      self.transition(new_size)
      self.is_joining = False
    else:
      # Wait for the initial set of workers to join before proceeding further
      if self.is_master and self.awaiting_initial_workers and batch == 1:
        while self.new_size is None:
          logging.info("Waiting for the initial set of workers to join...")
          time.sleep(RETRY_INTERVAL_SECONDS)
        self.awaiting_initial_workers = False
      # Maybe transition to a new cluster
      new_size = self.new_size if self.is_master else None
      new_size = self.comm.bcast(new_size, root=0)
      if new_size is not None:
        self.transition(new_size)

  def on_batch_end(self, batch, logs=None):
    """
    Reset all elasticity state passed to tensorflow.
    """
    global NUM_VIRTUAL_NODES
    global START_BATCH
    global START_EPOCH
    NUM_VIRTUAL_NODES = None
    START_BATCH = None
    START_EPOCH = None

  def on_epoch_begin(self, epoch, logs=None):
    self.current_epoch = epoch

