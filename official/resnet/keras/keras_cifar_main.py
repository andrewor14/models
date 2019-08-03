# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from autoscaling import autoscaling_helper
from autoscaling.params import AutoscalingStatus
from official.resnet import cifar10_main as cifar_main
from official.resnet.keras import keras_common
from official.resnet.keras import resnet_cifar_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils


LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]


def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = keras_common.BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


def parse_record_keras(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  This method converts the label to one hot to fit the loss function.

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image, label = cifar_main.parse_record(raw_record, is_training, dtype)
  label = tf.compat.v1.sparse_to_dense(label, (cifar_main.NUM_CLASSES,), 1)
  return image, label


def do_run(flags_obj, autoscaling_callback):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  tf.logging.info("Starting do_run")

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  tf.logging.info("Getting dist strat")

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus)
  strategy = None

  tf.logging.info("Dist strat was %s" % strategy)

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  if flags_obj.use_synthetic_data:
    distribution_utils.set_up_synthetic_data()
    input_fn = keras_common.get_synth_input_fn(
        height=cifar_main.HEIGHT,
        width=cifar_main.WIDTH,
        num_channels=cifar_main.NUM_CHANNELS,
        num_classes=cifar_main.NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj))
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = cifar_main.input_fn

  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=parse_record_keras)

  eval_input_dataset = input_fn(
      is_training=False,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      num_epochs=flags_obj.train_epochs,
      parse_record_fn=parse_record_keras)

  callbacks = keras_common.get_callbacks(
      learning_rate_schedule,
      cifar_main.NUM_IMAGES['train'],
      autoscaling_callback.num_batches_processed_this_epoch)

  tf.logging.info("Making optimizer")
  
  with strategy_scope:
    optimizer = keras_common.get_optimizer()
    #if flags_obj.use_horovod:
    #  import horovod.tensorflow as hvd
    #  optimizer = hvd.DistributedOptimizer(optimizer)
    tf.logging.info("Making model")
    model = resnet_cifar_model.resnet56(classes=cifar_main.NUM_CLASSES)
    tf.logging.info("Compiling model")
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  run_eagerly=flags_obj.run_eagerly,
                  metrics=['categorical_accuracy'])
    autoscaling_callback.set_model(model)

  # Add autoscaling callbacks
  callbacks.append(autoscaling_callback)
  autoscaling_schedule_callback = autoscaling_helper.get_schedule_callback(autoscaling_callback)
  if autoscaling_schedule_callback is not None:
    callbacks.append(autoscaling_schedule_callback)

  num_eval_steps = (cifar_main.NUM_IMAGES['validation'] //
                    flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  if not strategy and flags_obj.explicit_gpu_placement:
    # TODO(b/135607227): Add device scope automatically in Keras training loop
    # when not using distribition strategy.
    no_dist_strat_device = tf.device('/device:GPU:0')
    no_dist_strat_device.__enter__()

  #from deploy import mpi_spawn_test
  #mpi_spawn_test.algorithm2(autoscaling_callback.agent)

  while autoscaling_callback.agent.mpi_communicator.size < 10:
    if flags_obj.use_horovod:
      # Note: we force the user to enable eager mode when using horovod to simplify things.
      # For example, in eager mode, there are no global variables so we don't need to broadcast
      # them through horovod before training.
      if not flags_obj.enable_eager:
        raise ValueError("Eager mode must be enabled when using horovod")
      import horovod.tensorflow as hvd
      from mpi4py import MPI
      # Hack:
      tf.logging.info("hack begin")
      tf.keras.backend.clear_session()
      hvd.init(MPI.COMM_WORLD.Dup())
      hvd.allreduce(tf.constant(hvd.rank()))
      hvd.shutdown()
      tf.keras.backend.clear_session()
      tf.logging.info("hack end")

      tf.logging.info("hvd.init")
      hvd.init(autoscaling_callback.agent.mpi_communicator)
      tf.logging.info("done hvd.init")
      tf.logging.info("hvd size = %s" % hvd.size())
      #if "AUTOSCALING_MASTER_HOST_PORT" in os.environ or not first_time:
      tf.logging.info("Doing a round of allreduce before training")
      avg_rank = hvd.allreduce(tf.constant(hvd.rank()))
      tf.logging.info("Result was = %s" % avg_rank)
    if flags_obj.use_horovod:
      import horovod.tensorflow as hvd
      tf.logging.info("hvd.shutdown")
      hvd.shutdown()

    #(train_steps, train_epochs) = autoscaling_helper.get_train_steps_and_epochs(\
    #  cifar_main.NUM_IMAGES["train"], flags_obj, autoscaling_callback)

    tf.logging.info("model.fit")
    #history = model.fit(train_input_dataset,
    #                    epochs=train_epochs,
    #                    steps_per_epoch=train_steps,
    #                    callbacks=callbacks,
    #                    validation_steps=num_eval_steps,
    #                    validation_data=validation_data,
    #                    validation_freq=flags_obj.epochs_between_evals,
    #                    verbose=2)
    tf.logging.info("model.fit done")

    if autoscaling_callback.agent.mpi_communicator.rank == 0:
      autoscaling_callback.agent.mpi_spawn_worker()
    # Wait until we have a pending cluster spec
    import time
    while True:
      with autoscaling_callback.agent.pending_cluster_spec_lock:
        if autoscaling_callback.agent.pending_cluster_spec is not None:
          break
      time.sleep(1)
    autoscaling_callback.agent.initialize()

  # If we finished all the epochs already, then signal to above that we're terminating
  eval_output = None
  if autoscaling_callback.num_epochs_processed == flags_obj.train_epochs:
    autoscaling_callback.agent.status = AutoscalingStatus.TERMINATED
    if not flags_obj.skip_eval:
      eval_output = model.evaluate(eval_input_dataset,
                                   steps=num_eval_steps,
                                   verbose=2)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()

  stats = keras_common.build_stats(history, eval_output, callbacks)

  return stats


def define_cifar_flags():
  keras_common.define_keras_flags(dynamic_loss_scale=False)

  flags_core.set_defaults(data_dir='/tmp/cifar10_data/cifar-10-batches-bin',
                          model_dir='/tmp/cifar10_model',
                          train_epochs=182,
                          epochs_between_evals=10,
                          batch_size=128)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    autoscaling_helper.run_keras(flags.FLAGS, do_run)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_cifar_flags()
  absl_app.run(main)
