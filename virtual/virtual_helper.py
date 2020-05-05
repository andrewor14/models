#!/usr/bin/env python3

import glob
import json
import os

from absl import logging
from mpi4py import MPI
import tensorflow as tf


# A tensorflow function that averages a list of gradients with horovod
# We refresh this function every time the cluster membership changes
# instead of rebuilding the entire graph to speed up the restart process
HOROVOD_ALLREDUCE_FUNCTION = None

# Environment variables
TF_CONFIG = "TF_CONFIG"
MPI_SPAWN_RANK = "MPI_SPAWN_RANK"
NUM_VIRTUAL_NODES_PER_DEVICE = "NUM_VIRTUAL_NODES_PER_DEVICE"
HOROVOD_ENABLED = "HOROVOD_ENABLED"
HOROVOD_VERBOSE = "HOROVOD_VERBOSE"
HOROVOD_COMPRESS = "HOROVOD_COMPRESS"
HOROVOD_USE_CPU = "HOROVOD_USE_CPU"

def initialize():
  """
  Initialize the program for virtual node processing.
  """
  if horovod_enabled():
    initialize_horovod(MPI.COMM_WORLD)
  else:
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

def get_input_contexts():
  """
  Return a list of `tf.distribute.InputContext`s that matches this process' rank.
  """
  n = int(os.getenv(NUM_VIRTUAL_NODES_PER_DEVICE) or 1)
  num_shards = MPI.COMM_WORLD.size * n
  shard_indices = [MPI.COMM_WORLD.rank * n + i for i in range(n)]
  return [tf.distribute.InputContext(num_shards, i) for i in shard_indices]

def horovod_enabled():
  """
  Return whether to use Horovod for gradient synchronization.
  """
  return os.getenv(HOROVOD_ENABLED, "").lower() == "true"

def initialize_horovod(comm, restarting=False):
  """
  Initialize Horovod with the given communicator and set the allreduce function for
  tensorflow to call during training.
  """
  if not horovod_enabled():
    raise ValueError("Attempted to initialize horovod but %s is not set" % HOROVOD_ENABLED)
  logging.info("Initializing horovod with communicator (size = %s)" % comm.size)
  import horovod.tensorflow as hvd
  if restarting:
    hvd.shutdown()
  hvd.init(comm)
  # Truncate tensor for printing
  @tf.function
  def truncate_tensor(t):
    return tf.reshape(t, [-1])[:5]
  # Allreduce function
  @tf.function
  def allreduce(grads):
    import horovod.tensorflow as hvd
    logging.info("Averaging gradients with horovod (size %s)" % hvd.size())
    verbose = os.getenv(HOROVOD_VERBOSE, "").lower() == "true"
    compress = os.getenv(HOROVOD_COMPRESS, "").lower() == "true"
    use_cpu = os.getenv(HOROVOD_USE_CPU, "").lower() == "true"
    if verbose:
      tf.print("First gradient before horovod allreduce: ", truncate_tensor(grads[0]))
    compression = hvd.Compression.fp16 if compress else hvd.Compression.none
    device_dense = "/cpu:0" if use_cpu else ""
    grads = [hvd.allreduce(grad, device_dense=device_dense, compression=compression) for grad in grads]
    if verbose:
      tf.print("First gradient after horovod allreduce: ", truncate_tensor(grads[0]))
    return grads
  global HOROVOD_ALLREDUCE_FUNCTION
  HOROVOD_ALLREDUCE_FUNCTION = allreduce

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
  def __init__(self, model_dir, num_to_keep=5):
    if num_to_keep < 1:
      raise ValueError("Must keep at least 1 checkpoint")
    self.model_dir = model_dir
    self.num_to_keep = num_to_keep

  def on_epoch_end(self, epoch, logs=None):
    index_files = glob.glob("%s/*ckpt*.index" % self.model_dir)
    index_files.sort(key=lambda x: os.path.getmtime(x))
    for index_file in index_files[:-1 * self.num_to_keep]:
      for f in glob.glob("%s*" % index_file.strip(".index")):
        logging.info("Deleting old checkpoint %s" % f)
        os.remove(f)

