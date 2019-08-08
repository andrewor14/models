#!/usr/bin/env python3

import copy
import json
import time
import threading
import xmlrpc.server

import tensorflow as tf

from autoscaling.client import convert_port, connect
from autoscaling.params import *


def log_fn(msg):
  tf.logging.info("[Autoscaling service]: %s" % msg)

def listen_for_requests(agent, host_port):
  '''
  Start a server listening for autoscaling requests

  This is a simple RPC server that exposes an interface for adjusting
  the number of workers in a running job.
  '''
  log_fn("Listening for autoscaling requests on host port %s" % host_port)
  split = host_port.split(":")
  server = xmlrpc.server.SimpleXMLRPCServer(
    (split[0], int(split[1])), logRequests=False, allow_none=True)
  server.register_introspection_functions()
  server.register_multicall_functions()
  server.register_instance(AutoscalingService(agent))
  threading.Thread(target=server.serve_forever).start()


class AutoscalingService:
  '''
  A service for handling autoscaling requests.
  '''
  def __init__(self, agent):
    self.agent = agent

  def get_status(self):
    return self.agent.status.value

  def get_cluster_spec(self):
    return copy.deepcopy(self.agent.cluster_spec)

  def get_progress(self):
    '''
    Return a 2-tuple of
      (1) Number of batches processed in this epoch so far, and
      (2) Number of epochs processed so far.
    '''
    if self.agent.get_progress_method is not None:
      return self.agent.get_progress_method()
    return (None, None)

  ### FOR TESTING ONLY ###
  def set_mpi_communicator_size(self, target_size):
    current_size = self.agent.mpi_communicator.size
    def run():
      if target_size > current_size:
        self.agent.add_workers_to_mpi_communicator(target_size - current_size)
      elif target_size < current_size:
        workers_to_remove = self.agent.cluster_spec["worker"][-1 * (current_size - target_size):]
        self.agent.remove_workers_from_mpi_communicator(workers_to_remove)
    import threading
    threading.Thread(target=run).start()

  def join_cluster(self, host_port):
    '''
    Handle a join request, only called on the master server.

    The join request is rejected if it is received during initialization.
    Return whether the join request has been accepted.
    '''
    log_fn("Received join cluster request from %s" % host_port)
    if not is_running(self.agent.status):
      log_fn("Rejecting join cluster request from %s because we are initializing" % host_port)
      return False
    cluster_spec = self.get_cluster_spec()
    ps_hosts = cluster_spec["ps"] if "ps" in cluster_spec else []
    worker_hosts = cluster_spec["worker"]
    hosts = ps_hosts + worker_hosts
    if host_port not in hosts:
      # Wait until client is ready
      while self.agent.client is None:
        log_fn("... autoscaling client is not ready yet, waiting %s second(s)" %\
          AUTOSCALING_RETRY_INTERVAL_SECONDS)
        time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
      # Tell everyone to add this worker.
      # Note: Calling client.add_worker directly will hang because this server
      # is single threaded and so cannot process the add_workers request asynchronously.
      # Thus, we need to treat the master server (ourselves) separately.
      client = self.agent.client
      for server in client.servers:
        if server != client.master_server:
          server.add_workers([host_port])
      self.add_workers([host_port])
      # Note: There may be pending workers that are not part of the autoscaling client yet.
      # Here we manually tell them to add this new worker. In the future there may be a
      # cleaner way to do this.
      with self.agent.pending_cluster_spec_lock:
        if self.agent.pending_cluster_spec is not None:
          pending_workers = self.agent.pending_cluster_spec["worker"]
          pending_workers = list(set(pending_workers) - set(self.get_cluster_spec()["worker"]))
          if host_port in pending_workers:
            pending_workers.remove(host_port)
          for pending_worker in pending_workers:
            log_fn("Telling pending worker %s to add worker %s" % (pending_worker, host_port))
            server = connect(convert_port(pending_worker))
            server.add_workers([host_port])
    else:
      log_fn("Warning: received join request from a worker who had already joined: %s" % host_port)
    return True

  def _get_or_create_pending_cluster_spec(self):
    '''
    Return the existing pending cluster spec or create a new one.
    The caller must hold `self.agent.pending_cluster_spec_lock`.
    '''
    if self.agent.pending_cluster_spec is None:
      self.agent.pending_cluster_spec = self.get_cluster_spec()
    return self.agent.pending_cluster_spec

  def add_workers(self, host_ports):
    log_fn("Handling add_workers request: %s" % host_ports)
    with self.agent.pending_cluster_spec_lock:
      cluster_spec = self._get_or_create_pending_cluster_spec()
      for host_port in host_ports:
        if host_port not in cluster_spec["worker"]:
          cluster_spec["worker"].append(host_port)
      # If we're using horovod, then sort the workers by rank to ensure everyone has the same order
      if self.agent.using_horovod():
        cluster_spec["worker"].sort(key=lambda hp: self.agent.all_host_ports.index(hp))

  def remove_workers(self, host_ports):
    log_fn("Handling remove_workers request: %s" % host_ports)
    with self.agent.pending_cluster_spec_lock:
      cluster_spec = self._get_or_create_pending_cluster_spec()
      for hp in host_ports:
        cluster_spec["worker"].remove(hp)

  def set_pending_cluster_spec(self, cluster_spec):
    '''
    Override the existing pending cluster spec to the one specified.
    '''
    log_fn("Handling request to set pending cluster spec to %s" % cluster_spec)
    with self.agent.pending_cluster_spec_lock:
      self.agent.pending_cluster_spec = cluster_spec

  def spawn_worker(self):
    log_fn("Handling spawn_worker request")
    return self.agent.mpi_spawn_worker()

