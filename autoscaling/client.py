#!/usr/bin/env python3

import copy
import http.client
import json
import os
import xmlrpc.client
import re
import subprocess
import sys
import time

import tensorflow as tf

from autoscaling.params import *


VERBOSE = False
RUNNING_IN_SHELL = sys.__stdin__.isatty()

def log_fn(msg):
  tf.logging.info("[Autoscaling client]: %s" % msg)

def convert_port(host_port):
  '''
  Helper method to convert a gRPC port to an autoscaling service port.
  '''
  split = host_port.split(":")
  new_port = int(split[1]) + AUTOSCALING_RPC_PORT_OFFSET
  return "%s:%s" % (split[0], new_port)

def connect(host_port):
  '''
  Connect to the given host port and return the corresponding ServerProxy object.

  If `convert_port` is true, convert the gRPC port to an autoscaling RPC port.
  This method retries indefinitely until success.
  '''
  if not host_port.startswith("http://"):
    host_port = "http://%s" % host_port
  if VERBOSE:
    log_fn("Connecting to autoscaling server at %s" % host_port)
  server = xmlrpc.client.ServerProxy(host_port)
  while True:
    try:
      # The connection is not complete until we can access the server's methods
      server.system.listMethods()
      if VERBOSE:
        log_fn("Connected to autoscaling server at %s!" % host_port)
      return server
    except (ConnectionRefusedError, OSError) as e:
      if VERBOSE:
        log_fn("... connection to %s failed, trying again in %s second(s)"\
          % (host_port, AUTOSCALING_RETRY_INTERVAL_SECONDS))
      time.sleep(AUTOSCALING_RETRY_INTERVAL_SECONDS)
    except Exception as e:
      log_fn("Unexpected error %s (%s)" % (e, type(e)))
      raise e


class AutoscalingClient:

  def __init__(self, master_host_port):
    self.master_host_port = None
    self.master_server = None
    self._cluster_spec = None
    self._servers = None
    self.reset(master_host_port)

  def reset(self, new_master_host_port=None, new_cluster_spec=None):
    '''
    Open a connection to each server in the system, found by fetching the cluster spec
    from the given master host port.
    '''
    if new_master_host_port is not None:
      self.master_host_port = new_master_host_port
      self.master_server = connect(self.master_host_port)
    self._cluster_spec = new_cluster_spec or self.master_server.get_cluster_spec()
    self._servers = {}

  @property
  def ps_hosts(self):
    return self._cluster_spec["ps"].copy() if "ps" in self._cluster_spec else []

  @property
  def worker_hosts(self):
    # Assume there will always be at least one worker
    return self._cluster_spec["worker"].copy()

  @property
  def hosts(self):
    return self.ps_hosts + self.worker_hosts

  @property
  def cluster_spec(self):
    return copy.deepcopy(self._cluster_spec)

  @property
  def servers(self):
    '''
    Return the ServerProxy objects associated with the hosts in the system.
    All RPC calls should go through this accessor.
    '''
    # If there are workers we know about but haven't connected to yet, connect to them
    if len(self.hosts) > len(self._servers):
      pending_hosts = list(set(self.hosts) - set(self._servers.keys()))
      for hp in pending_hosts:
        # Reuse the connection to the master server
        converted_hp = convert_port(hp)
        if converted_hp == self.master_host_port:
          self._servers[hp] = self.master_server
        else:
          self._servers[hp] = connect(converted_hp)
    # Otherwise, if there are expired workers, remove them
    elif len(self.hosts) < len(self._servers):
      expired_hosts = list(set(self._servers.keys()) - set(self.hosts))
      for hp in expired_hosts:
        del self._servers[hp]
    # Make sure we are connected to all hosts we know about
    if len(self.hosts) != len(self._servers):
      raise ValueError("Number of hosts is different from number of server proxies!\n" +
        "Hosts: %s\nServer proxies: %s" % (self.hosts, self._servers.keys()))
    return [self._servers[k] for k in sorted(self._servers.keys())]

  def rpc_without_connection_problems(self, rpc_closure, max_attempts=10):
    """
    Run a closure with guards against common transient connection problems.
    """
    num_attempts = 0
    while num_attempts < max_attempts:
      error = None
      try:
        return rpc_closure()
      except (http.client.ResponseNotReady, http.client.CannotSendRequest) as e:
        error = e
      except xmlrpc.client.Fault as e:
        if "CannotSendRequest" in e.faultString:
          error = e
        else:
          raise e
      log_fn("Warning: RPC failed with the following exception, trying again")
      log_fn("%s: %s" % (error.__class__.__name__, error))
      num_attempts += 1

  def all_servers_rpc(self, rpc_closure, except_master=False):
    """
    Run the specified RPC on all servers.
    """
    wrapped_closure = lambda: [rpc_closure(s) for s in self.servers\
      if not except_master or s != self.master_server]
    return self.rpc_without_connection_problems(wrapped_closure)

  def master_server_rpc(self, rpc_closure):
    """
    Run the specified RPC on the master server.
    """
    wrapped_closure = lambda: rpc_closure(self.master_server)
    return self.rpc_without_connection_problems(wrapped_closure)

  def add_worker(self, host_port):
    '''
    Add a worker identified by the given host_port to the system.
    '''
    self.add_workers([host_port])

  def remove_worker(self, host_port):
    '''
    Remove a worker identified by the given host_port from the system.
    '''
    self.remove_workers([host_port])

  def add_workers(self, host_ports):
    '''
    Add workers identified by the given host_ports to the system.
    '''
    known_host_ports = [hp for hp in host_ports if hp in self.worker_hosts]
    new_host_ports = [hp for hp in host_ports if hp not in self.worker_hosts]
    if len(known_host_ports) > 0:
      log_fn("Warning: not adding the following workers because they already exist: %s" % known_host_ports)
    for server in self.servers:
      server.add_workers(new_host_ports)
    self._cluster_spec["worker"].extend(new_host_ports)

  def remove_workers(self, host_ports):
    '''
    Remove workers identified by the given host_ports from the system.
    Note: the first worker may not be removed because that's the one we use for syncing cluster membership.
    '''
    known_host_ports = [hp for hp in host_ports if hp in self.worker_hosts]
    new_host_ports = [hp for hp in host_ports if hp not in self.worker_hosts]
    if len(new_host_ports) > 0:
      log_fn("Warning: not removing the following workers because they are not known to us: %s" % new_host_ports)
    if len(known_host_ports) == 0:
      log_fn("Warning: not removing any workers")
      return
    # Check if there are workers to remove in the first place
    workers = self._cluster_spec["worker"]
    if len(workers) == 0:
      raise ValueError("No workers to remove")
    # Do not allow removing the first worker
    first_worker = workers[0]
    if first_worker in known_host_ports:
      raise ValueError("Not allowed to remove the first worker %s" % first_worker)
    # Actually remove
    for server in self.servers:
      server.remove_workers(known_host_ports)
    for hp in known_host_ports:
      self._cluster_spec["worker"].remove(hp)

