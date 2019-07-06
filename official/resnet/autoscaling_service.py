#!/usr/bin/env python3

import json
import time
import threading
import xmlrpc.server

import tensorflow as tf

from autoscaling_params import *


def log_fn(msg):
  msg = "[Autoscaling service]: %s" % msg
  tf.compat.v1.logging.info(msg)

def listen_for_autoscaling_requests(hook, host_port):
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
  server.register_instance(AutoscalingService(hook))
  threading.Thread(target=server.serve_forever).start()


class AutoscalingService:
  '''
  A service for handling autoscaling requests.
  '''
  def __init__(self, hook):
    self.hook = hook

  def get_epoch(self):
    return self.hook.epoch

  def get_cluster_spec(self):
    return self.hook.cluster_spec

  def get_pending_cluster_spec(self):
    with self.hook.pending_cluster_spec_lock:
      return self.hook.pending_cluster_spec

  def get_global_batch_size(self):
    return self.hook.global_batch_size

  def join_cluster(self, host_port):
    '''
    Handle a join request, only called on the master server.
    '''
    log_fn("Received join cluster request from %s" % host_port)
    cluster_spec = self.get_cluster_spec()
    ps_hosts = cluster_spec["ps"] if "ps" in cluster_spec else []
    worker_hosts = cluster_spec["worker"]
    hosts = ps_hosts + worker_hosts
    is_new_worker = host_port not in hosts
    if is_new_worker:
      # Wait until client is ready
      while self.hook.client is None:
        log_fn("... autoscaling client is not ready yet, waiting %s second(s)" %\
          AUTOSCALING_RETRY_INTERVAL_SECONDS)
        time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
      # Tell everyone to add this worker.
      # Note: Calling client.add_worker directly will hang because this server
      # is single threaded and so cannot process the add_workers request asynchronously.
      # Thus, we need to treat the master server (ourselves) separately.
      client = self.hook.client
      for server in client.servers:
        if server != client.master_server:
          server.add_workers([host_port])
      self.add_workers([host_port])
      # Note: There may be pending workers that are not part of the autoscaling client yet.
      # Here we manually tell them to add this new worker. In the future there may be a
      # cleaner way to do this.
      with self.hook.pending_cluster_spec_lock:
        if self.hook.pending_cluster_spec is not None:
          from autoscaling_client import convert_port, connect
          pending_workers = self.hook.pending_cluster_spec["worker"]
          pending_workers = list(set(pending_workers) - set(self.get_cluster_spec()["worker"]))
          if host_port in pending_workers:
            pending_workers.remove(host_port)
          for pending_worker in pending_workers:
            log_fn("Telling pending worker %s to add worker %s" % (pending_worker, host_port))
            server = connect(convert_port(pending_worker))
            server.add_workers([host_port])
    return is_new_worker

  def _get_or_create_pending_cluster_spec(self):
    '''
    Return the existing pending cluster spec or create a new one.
    The caller must hold `self.hook.pending_cluster_spec_lock`.
    '''
    if self.hook.pending_cluster_spec is None:
      self.hook.pending_cluster_spec = self.get_cluster_spec()
    return self.hook.pending_cluster_spec

  def add_workers(self, host_ports):
    log_fn("Handling add_workers request: %s" % host_ports)
    with self.hook.pending_cluster_spec_lock:
      cluster_spec = self._get_or_create_pending_cluster_spec()
      cluster_spec["worker"].extend(host_ports)

  def remove_workers(self, host_ports):
    log_fn("Handling remove_workers request: %s" % host_ports)
    with self.hook.pending_cluster_spec_lock:
      cluster_spec = self._get_or_create_pending_cluster_spec()
      for hp in host_ports:
        cluster_spec["worker"].remove(hp)

