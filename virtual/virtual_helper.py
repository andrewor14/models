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
PYTHONPATH = "PYTHONPATH"
MODELS_DIR = "MODELS_DIR"
NUM_VIRTUAL_NODES_PER_DEVICE = "NUM_VIRTUAL_NODES_PER_DEVICE"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
OMPI_MCA_initial_wdir = "OMPI_MCA_initial_wdir"
RUN_SCRIPT = "RUN_SCRIPT"
SPAWN_GROUP = "SPAWN_GROUP"
SPAWN_START_RANK = "SPAWN_START_RANK"

# Constants
LAUNCH_DIRECTORY = os.getenv(OMPI_MCA_initial_wdir, "")
EXECUTABLE = "bash"

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

def set_tf_config(base_port=2222, comm=MPI.COMM_WORLD):
  """
  Set TF_CONFIG based on hostnames of all processes in the given MPI communicator.
  To avoid port collisions, we add a process' rank to its port.
  """
  all_hosts = comm.allgather(MPI.Get_processor_name())
  host_ports = ["%s:%s" % (host, base_port + i) for i, host in enumerate(all_hosts)]
  tf_config = {"cluster": {"worker": host_ports}, "task": {"type": "worker", "index": comm.rank}}
  tf_config = json.dumps(tf_config)
  logging.info("Setting %s to %s" % (TF_CONFIG, tf_config))
  os.environ[TF_CONFIG] = tf_config

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

def get_all_mpi_hosts():
  """
  Return a list of all possible hosts that MPI can use to spawn processes.
  """
  flag_name, value = os.environ["HOST_FLAG"].split(" ")
  if flag_name == "--host":
    return [h.split(":")[0] for h in value.split(",")]
  if flag_name == "--hostfile":
    with open(value) as f:
      return [l.strip() for l in f.readlines()]
  return []

def mpi_spawn(target_hosts, start_rank, env={}):
  """
  Spawn a process on each of the given target hosts through MPI.

  This is a helper for spawning a process that will be merged into an existing communicator.
  This method assumes the caller was launched using `mpirun` with either the `--host` or the
  `--hostfile` option, which controls the machines on which the new processes will be launched.
  The spawned processes will be grouped in the same MPI world.
  """
  logging.info("MPI spawn on target hosts %s (start rank = %s)" % (target_hosts, start_rank))
  # Set environment variables
  env = env.copy()
  env[PYTHONPATH] = os.getenv(MODELS_DIR)
  env[SPAWN_GROUP] = ",".join(target_hosts)
  env[SPAWN_START_RANK] = start_rank
  # Note: there is a max character limit for the value of MPI.Info!
  # Here we take care not to exceed it, otherwise we will see MPI_ERR_INFO_VALUE...
  env = "\n".join(["%s=%s" % (k, v) for k, v in env.items() if v is not None])
  if len(env) > MPI.MAX_INFO_VAL:
    raise ValueError("MPI environment string is longer than MPI_MAX_INFO_VAL(%s):\n%s" %\
      (MPI.MAX_INFO_VAL, env))
  info = MPI.Info.Create()
  info.Set("env", env)
  info.Set("host", ",".join(target_hosts))
  info.Set("map_by", "node")
  # Setting "bind_to" to "none" (default was "core") significantly improves MPI performance
  # for multi-threaded applications. See https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php
  info.Set("bind_to", "none")
  # Set arguments, assuming the scripts are in the same directory as this file
  run_script = os.path.join(LAUNCH_DIRECTORY, os.environ[RUN_SCRIPT])
  return MPI.COMM_SELF.Spawn(EXECUTABLE, args=[run_script], info=info, maxprocs=len(target_hosts))

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

