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
    mpi_helper.set_tf_config()
    algorithm2(AutoscalingAgent())
  except Exception as e:
    log("##### ERROR #####")
    log(e)
    import traceback
    traceback.print_stack()
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

def algorithm(agent):
  """
  Start the algorithm with a communicator that only includes the root
  then slowly spawn workers one by one and let them join our communicator.
  """
  comm = agent.mpi_communicator
  is_joining = AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_root = comm.rank == 0 and not is_joining
  if is_root:
    from autoscaling.params import AutoscalingStatus
    agent.status = AutoscalingStatus.RUNNING
  while comm.size < MAX_WORKERS:
    log("========== Join remote, current size = %s ==========" % comm.size)
    #spawn_intercomm = None
    if is_root:
      log("Master is spawning worker %s" % comm.size)
      agent.mpi_spawn_worker()
      #spawn_intercomm = agent.mpi_spawned_communicators.pop(0)
      #agent.mpi_spawned_communicators.append(spawn_intercomm)
      #env = { AUTOSCALING_MASTER_HOST_PORT: agent.client.master_host_port }
      #env = { AUTOSCALING_MASTER_HOST_PORT: "Joined worker(%s)" % comm.size }
      #spawn_intercomm = mpi_helper.spawn(comm.size, env=env)
    #elif is_joining:
    #  spawn_intercomm = MPI.Comm.Get_parent()
    #comm = mpi_helper.expand(comm, spawn_intercomm)
    agent.maybe_expand_mpi_communicator()
    comm = agent.mpi_communicator
    is_joining = False
  log(textwrap.dedent("""
    ***********************************************************
      All done, our rank in final communicator = %s (size %s)
    ***********************************************************""" % (comm.rank, comm.size)))

def algorithm2(agent):
  while agent.mpi_communicator.size < 10:
    agent.initialize()
    #import horovod.tensorflow as hvd
    #tf.logging.info("hvd.init")
    #hvd.init(autoscaling_callback.agent.mpi_communicator)
    #tf.logging.info("done hvd.init")
    #tf.logging.info("hvd size = %s" % hvd.size())
    ##if "AUTOSCALING_MASTER_HOST_PORT" in os.environ or not first_time:
    #tf.logging.info("Doing a round of allreduce before training")
    #avg_rank = hvd.allreduce(tf.constant(hvd.rank()))
    #tf.logging.info("Result was = %s" % avg_rank)
    #tf.logging.info("hvd.shutdown")
    #hvd.shutdown()

    if agent.mpi_communicator.rank == 0:
      agent.mpi_spawn_worker()
    # Wait until we have a pending cluster spec
    import time
    while True:
      with agent.pending_cluster_spec_lock:
        if agent.pending_cluster_spec is not None:
          break
      time.sleep(1)

if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  main()

