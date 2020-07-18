#!/usr/bin/env python3

import gc
import glob
import json
import os

from absl import logging
from mpi4py import MPI
import tensorflow as tf


# Environment variables
TF_CONFIG = "TF_CONFIG"
MPI_SPAWN_RANK = "MPI_SPAWN_RANK"
NUM_VIRTUAL_NODES_PER_DEVICE = "NUM_VIRTUAL_NODES_PER_DEVICE"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

def initialize():
  """
  Initialize the program for virtual node processing.
  """
  set_tf_config()

def is_master(comm=MPI.COMM_WORLD):
  """
  Return whether this process is the master in the given communicator.
  """
  return comm.rank == 0

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
  ## TODO: delete
  #if "SPAWNED" in os.environ:
  #  old_workers = ["ns-l10c1n1:2222", "ns-l10c1n3:2223", "ns-l10c1n4:2224", "ns-l10c1n5:2225"]
  #  tf_config = json.loads(tf_config)
  #  tf_config["cluster"]["worker"] = old_workers + tf_config["cluster"]["worker"]
  #  tf_config["task"]["index"] += len(old_workers)
  #  tf_config = json.dumps(tf_config)
  #  os.environ[TF_CONFIG] = tf_config

def get_tf_config():
  """
  Get the value of TF_CONFIG as a python dictionary.
  """
  tf_config = os.getenv(TF_CONFIG)
  if tf_config is None:
    return None
  else:
    return json.loads(tf_config)

def get_input_context(comm=MPI.COMM_WORLD):
  """
  Return a `tf.distribute.InputContext`s that matches this process' rank.
  """
  return tf.distribute.InputContext(comm.size, comm.rank)

def get_checkpoint_path(checkpoint_dir):
  """
  Given a checkpoint directory, return the path to the final checkpoint in the directory.
  """
  metadata_file = "%s/checkpoint" % checkpoint_dir
  if not os.path.isfile(metadata_file):
    raise ValueError("Did not find metadata file 'checkpoint' in directory %s" % checkpoint_dir)
  checkpoint_name = None
  with open(metadata_file) as f:
    for line in f.readlines():
      if "model_checkpoint_path" in line:
        checkpoint_name = line.split(":")[1].strip().strip("\"")
        break
  if checkpoint_name is None:
    raise ValueError("Could not parse checkpoint name from metadata file %s" % metadata_file)
  return os.path.join(checkpoint_dir, checkpoint_name)

class DeleteOldCheckpointsCallback(tf.keras.callbacks.Callback):
  """
  Helper callback to delete old checkpoints.
  """
  def __init__(self, model_dir, num_to_keep=None):
    self.model_dir = model_dir
    self.num_to_keep = num_to_keep or 5
    if self.num_to_keep < 1:
      raise ValueError("Must keep at least 1 checkpoint")

  def on_epoch_end(self, epoch, logs=None):
    index_files = glob.glob("%s/*ckpt*.index" % self.model_dir)
    index_files.sort(key=lambda x: os.path.getmtime(x))
    for index_file in index_files[:-1 * self.num_to_keep]:
      for f in glob.glob("%s*" % index_file.strip(".index")):
        logging.info("Deleting old checkpoint %s" % f)
        os.remove(f)

class MonitorMemoryCallback(tf.keras.callbacks.Callback):
  """
  Helper callback to monitor memory usage periodically.
  """
  def __init__(self, should_trigger_gc=None):
    self.should_trigger_gc = should_trigger_gc or\
      os.getenv("DISABLE_GC_COLLECT", "").lower() != "true"
    self.devices = os.getenv(CUDA_VISIBLE_DEVICES)
    if self.devices is not None:
      self.devices = [int(d) for d in self.devices.split(",")]
      import nvidia_smi
      nvidia_smi.nvmlInit()

  def on_batch_end(self, batch, logs=None):
    import psutil
    main_memory = psutil.virtual_memory()
    logging.info("Main memory at the end of batch %s: used = %s bytes (out of %s bytes)" %\
      (batch, main_memory.used, main_memory.total))
    if self.devices is not None:
      import nvidia_smi
      gpu_memory_used = {}
      total_gpu_memory = None
      for d in self.devices:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(d)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used[d] = info.used
        total_gpu_memory = total_gpu_memory or info.total
      average_gpu_memory_used = round(sum(gpu_memory_used.values()) / len(gpu_memory_used))
      logging.info("GPU memory at the end of batch %s: avg = %s bytes (out of %s bytes), all = %s" %\
        (batch, average_gpu_memory_used, total_gpu_memory, gpu_memory_used))

  def on_epoch_end(self, epoch, logs=None):
    if self.should_trigger_gc:
      logging.info("Triggering gc.collect()")
      gc.collect()

  def on_train_end(self, logs=None):
    if self.devices is not None:
      import nvidia_smi
      nvidia_smi.nvmlShutdown()

