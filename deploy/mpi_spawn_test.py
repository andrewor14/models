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
  tf.logging.info("%s: %s" % (identifier, msg))

def main():
  try:
    if os.getenv("ENABLE_EAGER", "") == "true":
      algorithm2_eager(MPI.COMM_WORLD.Dup())
    else:
      algorithm(MPI.COMM_WORLD.Dup())
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
    x = tf.constant(1)
    log("Tensor name = %s" % x.name)
    if my_rank < group_size:
      new_group = MPI.COMM_WORLD.group.Incl(list(range(group_size)))
      sub_comm = MPI.COMM_WORLD.Create_group(new_group)
      log("Rank %s: created group of size %s" % (my_rank, sub_comm.size))
      log("Rank %s: before hvd.init" % my_rank)
      hvd.init(sub_comm)
      log("Rank %s: creating hvd.allreduce op" % my_rank)
      with K.get_graph().as_default():
        rank_tensor = tf.constant(my_rank)
        log("Rank %s: rank tensor name = %s" % (my_rank, rank_tensor.name))
        avg_rank = hvd.allreduce(rank_tensor)
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

    # Try it out
    import horovod.tensorflow as hvd
    from tensorflow.python.keras import backend as K
    mpi_helper.test_communication(comm)
    my_rank = comm.rank
    # Dummy allreduce
    tf.keras.backend.clear_session()
    hvd.init(MPI.COMM_WORLD.Dup())
    with K.get_graph().as_default():
      avg_rank = hvd.allreduce(tf.constant(my_rank))
    log("Average rank in dummy allreduce was %s" % K.get_session().run(avg_rank))
    hvd.shutdown()
    tf.keras.backend.clear_session()

    # Try some stupid things
    strategy = tf.distribute.OneDeviceStrategy("device:CPU:0")
    from official.resnet.keras import keras_common
    from official.resnet.keras import resnet_cifar_model
    from official.resnet.keras import cifar_preprocessing
    tf.keras.backend.clear_session()
    with strategy.scope():
      log("Building some stupid optimizer")
      optimizer = keras_common.get_optimizer()
      optimizer = hvd.DistributedOptimizer(optimizer)
      #model_input = tf.constant([[1.0, 2.0, 3.0, 4.0]])
      #model_output = tf.constant([float(i) for i in range(10)])
      #model = tf.keras.Sequential()
      #model.add(tf.keras.layers.Dense(4, input_dim=4, activation='relu'))
      #model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
      #loss = tf.keras.losses.mse(model(model_input), model_output)
      log("Building some stupid model")
      model = resnet_cifar_model.resnet56(classes=cifar_preprocessing.NUM_CLASSES)
      log("Compiling the stupid model")
      model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=(['categorical_accuracy']),
        run_eagerly=False)
    input_dataset = cifar_preprocessing.input_fn(
        is_training=True,
        data_dir=os.getenv("CIFAR10_DATA_DIR"),
        batch_size=32,
        num_epochs=1,
        parse_record_fn=cifar_preprocessing.parse_record)

    # Do the real allreduce
    log("Rank %s: before hvd.init" % my_rank)
    hvd.init(comm)
    log("Rank %s: creating hvd.allreduce op" % my_rank)
    with K.get_graph().as_default():
      avg_rank = hvd.allreduce(tf.constant(my_rank))
      log("Rank %s: running hvd.allreduce op" % my_rank)
      log("Rank %s: average rank was %s" % (my_rank, K.get_session().run(avg_rank)))
      log("Rank %s: calling model.fit" % my_rank)
      #optimizer.get_gradients(loss, model.trainable_weights)
      model.fit(input_dataset, epochs=1, steps_per_epoch=1, verbose=2)
      log("Rank %s: model.fit done" % my_rank)
    log("Rank %s: shutting down" % my_rank)
    hvd.shutdown()

    is_joining = False
  log(textwrap.dedent("""
    ***********************************************************
      All done, our rank in final communicator = %s (size %s)
    ***********************************************************""" % (comm.rank, comm.size)))

if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  main()

