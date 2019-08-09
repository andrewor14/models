import os
import sys
import textwrap

from mpi4py import MPI
import tensorflow as tf

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
    if os.getenv("ENABLE_EAGER", "") == "true":
      algorithm2_eager(MPI.COMM_WORLD.Dup())
    else:
      algorithm2(MPI.COMM_WORLD.Dup())
  except Exception as e:
    log("##### ERROR #####")
    raise e

def algorithm2(comm):
  import horovod.tensorflow as hvd
  from tensorflow.python.keras import backend as K
  sub_comm = None
  my_rank = comm.rank
  for group_size in range(1, comm.size + 1):
    tf.keras.backend.clear_session()
    hvd.init(comm)
    with K.get_graph().as_default():
      avg_rank = hvd.allreduce(tf.constant(my_rank))
    log("Average rank in dummy allreduce was %s" % K.get_session().run(avg_rank))
    hvd.shutdown()
    tf.keras.backend.clear_session()
    if my_rank < group_size:
      new_group = MPI.COMM_WORLD.group.Incl(list(range(group_size)))
      sub_comm = MPI.COMM_WORLD.Create_group(new_group)
      log("Rank %s: created group of size %s" % (my_rank, sub_comm.size))
      log("Rank %s: before hvd.init" % my_rank)
      hvd.init(sub_comm)
      log("Rank %s: creating hvd.allreduce op" % my_rank)
      with K.get_graph().as_default():
        avg_rank = hvd.allreduce(tf.constant(my_rank))
      log("Rank %s: running hvd.allreduce op" % my_rank)
      log("Rank %s: average rank was %s" % (my_rank, K.get_session().run(avg_rank)))
      log("Rank %s: shutting down" % my_rank)
      hvd.shutdown()
    else:
      log("Rank %s not participating in allreduce yet" % my_rank)
    comm.barrier()

def algorithm2_eager(comm):
  tf.enable_eager_execution()
  import horovod.tensorflow as hvd
  sub_comm = None
  my_rank = comm.rank
  for group_size in range(1, comm.size + 1):
    tf.keras.backend.clear_session()
    hvd.init(comm)
    hvd.allreduce(tf.constant(my_rank))
    hvd.shutdown()
    tf.keras.backend.clear_session()
    if my_rank < group_size:
      new_group = MPI.COMM_WORLD.group.Incl(list(range(group_size)))
      sub_comm = MPI.COMM_WORLD.Create_group(new_group)
      log("Rank %s: created group of size %s" % (my_rank, sub_comm.size))
      log("Rank %s: before hvd.init" % my_rank)
      hvd.init(sub_comm)
      log("Rank %s: running hvd.allreduce" % my_rank)
      avg_rank = hvd.allreduce(tf.constant(my_rank))
      log("Rank %s: shutting down" % my_rank)
      hvd.shutdown()
      log("Rank %s: average rank was %s" % (my_rank, avg_rank))
    else:
      log("Rank %s not participating in allreduce yet" % my_rank)
    comm.barrier()
       
def algorithm(comm):
  """
  Start the algorithm with a communicator that only includes the root
  then slowly spawn workers one by one and let them join our communicator.
  """
  is_joining = AUTOSCALING_MASTER_HOST_PORT in os.environ
  is_root = comm.rank == 0 and not is_joining
  while comm.size < MAX_WORKERS:
    log("========== Join remote, current size = %s ==========" % comm.size)
    spawn_intercomm = None
    if is_root:
      log("Master is spawning worker %s" % comm.size)
      env = { AUTOSCALING_MASTER_HOST_PORT: "Joined worker(%s)" % comm.size }
      spawn_intercomm = mpi_helper.spawn(comm.size, env=env)
    elif is_joining:
      spawn_intercomm = MPI.Comm.Get_parent()
    comm = mpi_helper.expand(comm, spawn_intercomm)
    is_joining = False
  log(textwrap.dedent("""
    ***********************************************************
      All done, our rank in final communicator = %s (size %s)
    ***********************************************************""" % (comm.rank, comm.size)))

if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  main()

