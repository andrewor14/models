#!/usr/bin/env python3

import json
import os

import tensorflow as tf


# Environment variables
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
ORIGINAL_CUDA_VISIBLE_DEVICES = "ORIGINAL_CUDA_VISIBLE_DEVICES"
TF_CONFIG = "TF_CONFIG"

def log_fn(msg):
  tf.logging.info("[CUDA helper]: %s" % msg)

def set_cuda_visible_devices(num_gpus_per_worker, tf_config=None):
  """
  Set the correct CUDA_VISIBLE_DEVICES based on how many other workers share this machine.

  This assumes CUDA_VISIBLE_DEVICES is already set and `num_gpus_per_worker` divides the
  number of devices in CUDA_VISIBLE_DEVICES. This should be called every time cluster
  membership changes.

  If `tf_config` is not provided, the environment variable TF_CONFIG is read instead.
  """
  # Keep around the original in case we need to rebalance GPUs
  if ORIGINAL_CUDA_VISIBLE_DEVICES not in os.environ:
    os.environ[ORIGINAL_CUDA_VISIBLE_DEVICES] = os.environ[CUDA_VISIBLE_DEVICES]    
  cuda_visible_devices = os.environ[ORIGINAL_CUDA_VISIBLE_DEVICES].split(",")
  tf_config = tf_config or json.loads(os.environ[TF_CONFIG])
  # Find out our local index on this machine
  # Note: we cannot use OMPI_COMM_WORLD_LOCAL_RANK because spawned workers belong to different worlds
  workers = tf_config["cluster"]["worker"]
  my_index = tf_config["task"]["index"]
  my_host_port = workers[my_index]
  workers_on_my_host = [w for w in workers if w.split(":")[0] == my_host_port.split(":")[0]]
  my_local_index = workers_on_my_host.index(my_host_port)
  # Parse our devices from CUDA_VISIBLE_DEVICES
  cuda_start_index = my_local_index * num_gpus_per_worker
  cuda_end_index = cuda_start_index + num_gpus_per_worker
  cuda_visible_devices = cuda_visible_devices[cuda_start_index:cuda_end_index]
  if len(cuda_visible_devices) != num_gpus_per_worker:
    raise ValueError("%s (%s) did not split evenly among workers (expecting %s GPUs each)" %\
      (CUDA_VISIBLE_DEVICES, os.environ[CUDA_VISIBLE_DEVICES], num_gpus_per_worker))
  cuda_visible_devices = ",".join(cuda_visible_devices)
  if os.environ[CUDA_VISIBLE_DEVICES] == cuda_visible_devices:
    log_fn("%s is currently set to %s" % (CUDA_VISIBLE_DEVICES, cuda_visible_devices))
  else:
    log_fn("Setting %s to %s" % (CUDA_VISIBLE_DEVICES, cuda_visible_devices))
    os.environ[CUDA_VISIBLE_DEVICES] = cuda_visible_devices


