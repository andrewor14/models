import os
import sys
import textwrap

from mpi4py import MPI
import horovod.tensorflow as hvd
import tensorflow as tf
from tensorflow.python import keras
from autoscaling.agent import AutoscalingAgent

from deploy import mpi_helper


AUTOSCALING_MASTER_HOST_PORT = "AUTOSCALING_MASTER_HOST_PORT"
STARTING_TAG = 100
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))

def log(msg):
  identifier = os.getenv(AUTOSCALING_MASTER_HOST_PORT,\
    "Master" if MPI.COMM_WORLD.rank == 0 else "Worker")
  identifier = "%s on %s" % (identifier, MPI.Get_processor_name())
  print("%s: %s" % (identifier, msg))

def main():
  try:
    #tf.enable_eager_execution()
    algorithm()
  except Exception as e:
    log("##### ERROR #####")
    raise e

def experiment():
  comm = MPI.COMM_WORLD.Dup()
  is_joining = AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_root = comm.rank == 0 and not is_joining
  comm_world = comm
  #log("***** Testing original world communication *****")
  #mpi_helper.test_communication(comm_world)
 
  # Join once
  spawn_intercomm = None
  if is_root:
    log("Master is spawning worker %s" % comm.size)
    env = { AUTOSCALING_MASTER_HOST_PORT: "Joined worker(%s)" % comm.size }
    spawn_intercomm = mpi_helper.spawn(comm.size, env=env)
  elif is_joining:
    spawn_intercomm = MPI.Comm.Get_parent()
  comm = mpi_helper.expand(comm, spawn_intercomm)
  comm_joined_once = comm

  # Test again with different communicators
  #log("***** Testing expanded(1) communicator *****")
  #mpi_helper.test_communication(comm_joined_once)

  # Join twice
  spawn_intercomm = None
  if is_root:
    log("Master is spawning worker %s" % comm.size)
    env = { AUTOSCALING_MASTER_HOST_PORT: "Joined worker(%s)" % comm.size }
    spawn_intercomm = mpi_helper.spawn(comm.size, env=env)
  elif is_joining:
    spawn_intercomm = MPI.Comm.Get_parent()
  comm_joined_twice = mpi_helper.expand(comm, spawn_intercomm)

  # Test again with different communicators
  log("***** Testing expanded(2) communicator *****")
  mpi_helper.test_communication(comm_joined_twice)

  log("All done")

def algorithm():
  """
  Start the algorithm with a communicator that only includes the root
  then slowly spawn workers one by one and let them join our communicator.
  """
  #comm = agent.mpi_communicator
  comm = MPI.COMM_WORLD.Dup()
  is_joining = AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_root = comm.rank == 0 and not is_joining
  #if is_root:
  #  from autoscaling.params import AutoscalingStatus
  #  agent.status = AutoscalingStatus.RUNNING
  while comm.size < MAX_WORKERS:
    log("========== Join remote, current size = %s ==========" % comm.size)
    spawn_intercomm = None
    if is_root:
      log("Master is spawning worker %s" % comm.size)
      #agent.mpi_spawn_worker()
      #spawn_intercomm = agent.mpi_spawned_communicators.pop(0)
      #agent.mpi_spawned_communicators.append(spawn_intercomm)
      #env = { AUTOSCALING_MASTER_HOST_PORT: agent.client.master_host_port }
      env = { AUTOSCALING_MASTER_HOST_PORT: "Joined worker(%s)" % comm.size }
      spawn_intercomm = mpi_helper.spawn(comm.size, env=env)
    elif is_joining:
      spawn_intercomm = MPI.Comm.Get_parent()
    comm = mpi_helper.expand(comm, spawn_intercomm)
    #if is_root: log("BEFORE expand root's spawned communicators: %s" % agent.mpi_spawned_communicators)
    #agent.maybe_expand_mpi_communicator()
    #if is_root: log("AFTER expand root's spawned communicators: %s" % agent.mpi_spawned_communicators)
    #agent.mpi_communicator = comm
    is_joining = False
  log(textwrap.dedent("""
    ***********************************************************
      All done, our rank in final communicator = %s (size %s)
    ***********************************************************""" % (comm.rank, comm.size)))

if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  main()

