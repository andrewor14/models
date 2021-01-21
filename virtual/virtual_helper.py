#!/usr/bin/env python3

import gc
import glob
import json
import math
import os

from absl import logging
from mpi4py import MPI
import tensorflow as tf


# Environment variables
TF_CONFIG = "TF_CONFIG"
PYTHONPATH = "PYTHONPATH"
MODELS_DIR = "MODELS_DIR"
NUM_NODES = "NUM_NODES"
NUM_GPUS = "NUM_GPUS"
NUM_VIRTUAL_NODES_PER_DEVICE = "NUM_VIRTUAL_NODES_PER_DEVICE"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
OMPI_MCA_initial_wdir = "OMPI_MCA_initial_wdir"
RUN_SCRIPT = "RUN_SCRIPT"
HOROVOD_COMPRESS = "HOROVOD_COMPRESS"
HOROVOD_USE_CPU = "HOROVOD_USE_CPU"
FORCE_EXIT = "FORCE_EXIT"
GLOBAL_BATCH_SIZE = "GLOBAL_BATCH_SIZE"

# Heterogeneous training environment variables
HETEROGENEOUS_SPLIT = "HETEROGENEOUS_SPLIT"
HETEROGENEOUS_RANGE_START = "HETEROGENEOUS_RANGE_START"
HETEROGENEOUS_RANGE_END = "HETEROGENEOUS_RANGE_END"
HETEROGENEOUS_PROFILE_CURRENT_BATCH_SIZE = "HETEROGENEOUS_PROFILE_CURRENT_BATCH_SIZE"
HETEROGENEOUS_PROFILE_MIN_BATCH_SIZE = "HETEROGENEOUS_PROFILE_MIN_BATCH_SIZE"
HETEROGENEOUS_PROFILE_MAX_BATCH_SIZE = "HETEROGENEOUS_PROFILE_MAX_BATCH_SIZE"
HETEROGENEOUS_PROFILE_STEPS = "HETEROGENEOUS_PROFILE_STEPS"

# Constants
LAUNCH_DIRECTORY = os.getenv(OMPI_MCA_initial_wdir, "")
EXECUTABLE = "bash"
ENABLE_HETEROGENEOUS = os.getenv("ENABLE_HETEROGENEOUS", "").lower() == "true"
HETEROGENEOUS_VERBOSE = os.getenv("HETEROGENEOUS_VERBOSE", "").lower() == "true"

# Tag used for expanding the MPI communicator, incremented once per expand
MPI_CURRENT_TAG = 14444

# A tensorflow function that averages a list of gradients with horovod
# We refresh this function every time the cluster membership changes
# instead of rebuilding the entire graph to speed up the restart process
HOROVOD_ALLREDUCE_FUNCTION = None

def initialize(comm=MPI.COMM_WORLD):
  """
  Initialize the program for virtual node processing.
  """
  set_tf_config(comm=comm)
  from virtual.elasticity_callback import ENABLE_ELASTICITY, initialize_singleton_callback
  if ENABLE_ELASTICITY:
    initialize_horovod()
    initialize_singleton_callback()
  if ENABLE_HETEROGENEOUS:
    global_batch_size = get_global_batch_size()
    if global_batch_size < 0:
      raise ValueError("'%s' must be set" % global_batch_size)
    start, end = get_heterogeneous_range(comm)
    os.environ[HETEROGENEOUS_RANGE_START] = str(start)
    os.environ[HETEROGENEOUS_RANGE_END] = str(end)
    os.environ[GLOBAL_BATCH_SIZE] = str(global_batch_size)

def get_global_batch_size():
  """
  Return the global batch size used in this job.
  If the batch size was not set, return -1.
  """
  return int(os.getenv(GLOBAL_BATCH_SIZE, os.getenv("BATCH_SIZE", -1)))

def get_heterogeneous_batch_size(comm=MPI.COMM_WORLD):
  """
  Return the batch size assigned to this rank.
  """
  start, end = get_heterogeneous_range(comm)
  range_size = end - start
  logging.info("Heterogeneous mode is enabled: batching dataset by" +\
    "%s instead of the global batch size %s" % (range_size, get_global_batch_size()))
  return range_size

def get_heterogeneous_range(comm=MPI.COMM_WORLD):
  """
  Return a 2-tuple (start, end) that represents the indices of examples in
  each batch assigned to this rank. The start is inclusive while the end is
  non-inclusive. This is used for sharding the dataset unevenly.
  """
  global_batch_size = get_global_batch_size()
  # TODO: for now, explicitly require the user to set the heterogeneous split
  # Format: comma-delimited per-worker batch sizes, one value per rank, e.g. 32,16,16
  # The sum of these splits must add up to the global batch size
  if HETEROGENEOUS_SPLIT not in os.environ:
    raise ValueError("'%s' must be set" % HETEROGENEOUS_SPLIT)
  split = [int(bs) for bs in os.environ[HETEROGENEOUS_SPLIT].split(",")]
  if sum(split) != global_batch_size:
    raise ValueError("Heterogeneous split '%s' doesn't sum up to the " % split +\
      "global batch size %s" % global_batch_size)
  if len(split) != comm.size:
    raise ValueError("Heterogeneous split '%s' doesn't match the world size %s" %\
      (split, comm.size))
  start = sum(split[:comm.rank])
  end = start + split[comm.rank]
  return (start, end)

def get_heterogeneous_profile_info():
  """
  Return a 3-tuple (min batch size, max batch size, steps) that represents a range
  of batch sizes to run during the offline profile phase of heterogeneous training,
  and the number of steps to profile for each batch size. All batch sizes must be
  multiples of 2.

  Return None if heterogeneous training is not enabled, or we are not in the offline
  profiling phase of heterogeneous training.
  """
  num_gpus = int(os.environ[NUM_GPUS])
  min_batch_size = int(os.getenv(HETEROGENEOUS_PROFILE_MIN_BATCH_SIZE, num_gpus))
  max_batch_size = int(os.getenv(HETEROGENEOUS_PROFILE_MAX_BATCH_SIZE, -1))
  steps = int(os.getenv(HETEROGENEOUS_PROFILE_STEPS, 10))
  if max_batch_size < 0:
    return None
  if not math.log2(min_batch_size / num_gpus).is_integer() or\
      not math.log2(max_batch_size / num_gpus).is_integer():
    raise ValueError("Heterogeneous profiling batch size range must be powers of 2" +\
      "*for each GPU (%s)* (was [%s, %s))" % (num_gpus, min_batch_size, max_batch_size))
  return (min_batch_size, max_batch_size, steps)

def set_heterogeneous_profile_batch_size(batch_size):
  """
  Set the current batch size used for profiling for heterogeneous training.
  """
  os.environ[HETEROGENEOUS_PROFILE_CURRENT_BATCH_SIZE] = str(batch_size)

def get_heterogeneous_profile_batch_size():
  """
  Return the current batch size used for profiling for heterogeneous training.
  """
  batch_size = os.getenv(HETEROGENEOUS_PROFILE_CURRENT_BATCH_SIZE)
  return int(batch_size) if batch_size is not None else None

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
  from virtual.elasticity_callback import ENABLE_ELASTICITY, JOB_ID, SPAWN_RANK
  my_host = MPI.Get_processor_name()
  if ENABLE_ELASTICITY:
    my_index = 0
    rank = int(os.getenv(SPAWN_RANK, 0))
    job_id = int(os.getenv(JOB_ID, 0))
    job_id_multiplier = 16 # TODO: this is arbitrary
    host_ports = ["%s:%s" % (my_host, base_port + rank + job_id * job_id_multiplier)]
  else:
    my_index = comm.rank
    all_hosts = comm.allgather(my_host)
    host_ports = ["%s:%s" % (host, base_port + i) for i, host in enumerate(all_hosts)]
  tf_config = {"cluster": {"worker": host_ports}, "task": {"type": "worker", "index": my_index}}
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

def maybe_force_exit():
  """
  If the env var FORCE_EXIT is set to true, force exit this program without cleaning up.
  """
  if os.getenv(FORCE_EXIT, "").lower() == "true":
    os._exit(0)

def get_input_context():
  """
  Return a `tf.distribute.InputContext`s that matches this process' rank.
  """
  from virtual.elasticity_callback import ENABLE_ELASTICITY, ELASTICITY_CALLBACK, SPAWN_RANK
  if ENABLE_ELASTICITY:
    size = ELASTICITY_CALLBACK.get_expected_size()
    rank = int(os.getenv(SPAWN_RANK, 0))
    return tf.distribute.InputContext(size, rank)
  else:
    return tf.distribute.InputContext(MPI.COMM_WORLD.size, MPI.COMM_WORLD.rank)

def get_virtual_batch_size(batch_size, num_virtual_nodes_per_device):
  """
  Return the number of examples to be processed per virtual node.
  """
  # In elasticity mode, we don't rely on tensorflow's experimental_distribute_dataset
  # to further split our dataset across the devices, so we need to further divide our
  # batch size here to ensure everyone gets the right batch size
  from virtual.elasticity_callback import ENABLE_ELASTICITY
  if ENABLE_ELASTICITY:
    return batch_size // (num_virtual_nodes_per_device * int(os.environ[NUM_NODES]))
  else:
    # TODO: better handling for the case when the batch size doesn't divide
    return batch_size // num_virtual_nodes_per_device

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
  return [h.split(":")[0] for h in os.environ["MPI_HOSTS"].split(",")]

def mpi_spawn(target_host, env={}):
  """
  Spawn a process on the given target host through MPI

  This is a helper for spawning a process that will be merged into an existing communicator.
  This method assumes the caller was launched using `mpirun` with either the `--host` or the
  `--hostfile` option, which controls the machines on which the new processes will be launched.
  """
  logging.info("MPI spawn on target host %s" % target_host)
  # Set environment variables
  env = env.copy()
  env[PYTHONPATH] = os.getenv(MODELS_DIR)
  # Note: there is a max character limit for the value of MPI.Info!
  # Here we take care not to exceed it, otherwise we will see MPI_ERR_INFO_VALUE...
  env = "\n".join(["%s=%s" % (k, v) for k, v in env.items() if v is not None])
  if len(env) > MPI.MAX_INFO_VAL:
    raise ValueError("MPI environment string is longer than MPI_MAX_INFO_VAL(%s):\n%s" %\
      (MPI.MAX_INFO_VAL, env))
  info = MPI.Info.Create()
  info.Set("env", env)
  info.Set("host", target_host)
  # Setting "bind_to" to "none" (default was "core") significantly improves MPI performance
  # for multi-threaded applications. See https://www.open-mpi.org/doc/v1.8/man1/mpirun.1.php
  info.Set("bind_to", "none")
  # Set arguments, assuming the scripts are in the same directory as this file
  run_script = os.path.join(LAUNCH_DIRECTORY, os.environ[RUN_SCRIPT])
  return MPI.COMM_SELF.Spawn(EXECUTABLE, args=[run_script], info=info, maxprocs=1)

def mpi_expand(intracomm, intercomm):
  """
  Expand an existing intracommunicator by merging an intercommunicator into it.

  All members of both communicators must participate in this process.
  For example, the intercommunicator may be set to the following:
    - At the root: the communicator returned by MPI.Comm.Spawn
    - At the spawned worker: the communicator returned by MPI.Comm.Get_parent
    - At all other nodes: None

  Return a merged intracommunicator ready to be passed into `hvd.init`.
  """
  from virtual.elasticity_callback import ELASTICITY_MASTER
  global MPI_CURRENT_TAG
  is_joining = intracomm.rank == 0 and ELASTICITY_MASTER in os.environ
  is_root = intracomm.rank == 0 and not is_joining
  tag = MPI_CURRENT_TAG if is_root else None

  if is_joining:
    logging.info("Joining an existing communicator")
  else:
    logging.info("Expanding communicator %s (current size %s)" % (intracomm, intracomm.size))

  # Merge the two intercommunicators into an intracommunicator
  if intercomm is not None:
    merged_intracomm = intercomm.Merge(is_joining)
  else:
    merged_intracomm = MPI.Intracomm(MPI.COMM_NULL)

  # The root broadcasts its tag in both communicators to make sure everyone has the same tag
  if intercomm is not None:
    tag = merged_intracomm.bcast(tag, root=0)
  tag = intracomm.bcast(tag, root=0)

  # Merge the two intracommunicators into an intercommunicator
  logging.info("Merging communicators using tag %s" % tag)
  if is_joining:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, merged_intracomm, 0, tag)
  else:
    super_merged_intercomm = MPI.Intracomm.Create_intercomm(intracomm, 0, merged_intracomm, 1, tag)

  # Finally, convert this intercommunicator into an intracommunicator
  comm = super_merged_intercomm.Merge(is_joining)
  logging.info("Our rank in new communicator = %s (size %s)" % (comm.rank, comm.size))

  # Run some collective operations on this communicator
  mpi_test_communication(comm)

  if is_root:
    MPI_CURRENT_TAG += 1
  return comm

def mpi_test_communication(comm):
  """
  Helper method to make sure basic communication works in the given communicator.
  """
  logging.info("Testing communication in communicator %s (size %s)" % (comm, comm.size))
  # Try broadcasting a value
  value = "[root value]" if comm.rank == 0 else None
  value = comm.bcast(value, root=0)
  if comm.rank > 0:
    logging.info("  Received broadcast from root: %s" % value)
  # Try doing an allreduce
  value = comm.allreduce(comm.rank, op=MPI.SUM)
  logging.info("  Allreduce result: %s" % value)

def initialize_horovod(comm=None):
  """
  Initialize horovod with the given communicator and set the allreduce
  function for tensorflow to call during training.

  If `comm` is None, default to the MPI world communicator. Otherwise,
  we assume horovod has been initialized before, so we shut down the
  old context and initialize a new one.
  """
  import horovod.tensorflow as hvd
  if comm is None:
    comm = MPI.COMM_WORLD.Dup()
  else:
    hvd.shutdown()
  logging.info("Initializing horovod with communicator (size = %s)" % comm.size)
  hvd.init(comm)
  # Truncate tensor for printing
  @tf.function
  def truncate_tensor(t):
    return tf.reshape(t, [-1])[:5]
  # Allreduce function
  @tf.function
  def allreduce(grads):
    import horovod.tensorflow as hvd
    from virtual.elasticity_callback import ELASTICITY_VERBOSE
    logging.info("Averaging gradients with horovod (size %s)" % hvd.size())
    compress = os.getenv(HOROVOD_COMPRESS, "").lower() == "true"
    use_cpu = os.getenv(HOROVOD_USE_CPU, "").lower() == "true"
    if ELASTICITY_VERBOSE:
      tf.print("First gradient before horovod allreduce: ", truncate_tensor(grads[0]))
    compression = hvd.Compression.fp16 if compress else hvd.Compression.none
    device_dense = "/cpu:0" if use_cpu else ""
    grads = [hvd.allreduce(grad, device_dense=device_dense, compression=compression)\
      for grad in grads]
    if ELASTICITY_VERBOSE:
      tf.print("First gradient after horovod allreduce: ", truncate_tensor(grads[0]))
    return grads
  global HOROVOD_ALLREDUCE_FUNCTION
  HOROVOD_ALLREDUCE_FUNCTION = allreduce

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

