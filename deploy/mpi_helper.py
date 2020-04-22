#!/usr/bin/env python3

import json
import os

from absl import logging
from mpi4py import MPI
import tensorflow as tf


# Environment variables
TF_CONFIG = "TF_CONFIG"
MPI_SPAWN_RANK = "MPI_SPAWN_RANK"
NUM_VIRTUAL_NODES_PER_DEVICE = "NUM_VIRTUAL_NODES_PER_DEVICE"

def set_tf_config(base_port=2222):
  """
  Set TF_CONFIG based on hostnames of all processes in MPI.COMM_WORLD.
  To avoid port collisions, we add a process' rank to its port.
  """
  my_host = MPI.Get_processor_name()
  my_index = MPI.COMM_WORLD.rank
  if MPI.COMM_WORLD.size == 1:
    host_ports = ["%s:%s" % (my_host, base_port + int(os.getenv(MPI_SPAWN_RANK, 0)))]
  else:
    all_hosts = MPI.COMM_WORLD.allgather(my_host)
    host_ports = ["%s:%s" % (host, base_port + i) for i, host in enumerate(all_hosts)]
  tf_config = {"cluster": {"worker": host_ports}, "task": {"type": "worker", "index": my_index}}
  tf_config = json.dumps(tf_config)
  logging.info("Setting %s to %s" % (TF_CONFIG, tf_config))
  os.environ[TF_CONFIG] = tf_config

def get_input_contexts():
  """
  Return a list of `tf.distribute.InputContext`s that matches this process' rank.
  """
  n = int(os.getenv(NUM_VIRTUAL_NODES_PER_DEVICE) or 1)
  num_shards = MPI.COMM_WORLD.size * n
  shard_indices = [MPI.COMM_WORLD.rank * n + i for i in range(n)]
  return [tf.distribute.InputContext(num_shards, i) for i in shard_indices]

