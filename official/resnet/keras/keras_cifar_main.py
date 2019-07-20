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

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import mpi_helper, cifar10_main as cifar_main
from official.resnet.autoscaling_agent import AutoscalingAgent
from official.resnet.autoscaling_params import AutoscalingStatus
from official.resnet.keras import keras_common
from official.resnet.keras import resnet_cifar_model
from official.resnet.keras.autoscaling_callback import AutoscalingCallback
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from slurm import slurm_helper


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


def run(flags_obj):
  """
  Wrapper around main loop for ResNet models that handles changes in cluster membership.
  """
  # If TF_CONFIG is not provided, set it based on environment variables from slurm or MPI
  if "TF_CONFIG" not in os.environ:
    if slurm_helper.running_through_slurm():
      num_ps = int(os.getenv("NUM_PARAMETER_SERVERS", "1"))
      slurm_helper.set_tf_config(num_ps)
    elif flags_obj.use_horovod:
      mpi_helper.set_tf_config()

  # Keep track of cluster membership changes through an autoscaling hook
  autoscaling_agent = AutoscalingAgent()
  autoscaling_callback = AutoscalingCallback(autoscaling_agent)

  while autoscaling_agent.status != AutoscalingStatus.TERMINATED:
    try:
      autoscaling_agent.initialize()
      result = do_run(flags_obj, autoscaling_callback)
      autoscaling_agent.on_restart()
      autoscaling_callback.reset()
    except Exception as e:
      tf.compat.v1.logging.error("Exception in resnet_main: %s (%s)" %\
        (e, e.__class__.__name__))
      traceback.print_exc()
      raise e
  return result


def do_run(flags_obj, autoscaling_callback):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  if flags_obj.use_horovod:
    # Note: we force the user to enable eager mode when using horovod to simplify things.
    # For example, in eager mode, there are no global variables so we don't need to broadcast
    # them through horovod before training.
    if not flags_obj.enable_eager:
      raise ValueError("Eager mode must be enabled when using horovod")
    import horovod.tensorflow.keras as hvd
    hvd.init(autoscaling_callback.agent.mpi_communicator)
    horovod_rank = hvd.local_rank()
  else:
    horovod_rank = None

  keras_utils.set_session_config(enable_eager=flags_obj.enable_eager,
                                 enable_xla=flags_obj.enable_xla,
                                 horovod_rank=horovod_rank)

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus)

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

  with strategy_scope:
    optimizer = keras_common.get_optimizer()
    if flags_obj.use_horovod:
      import horovod.tensorflow.keras as hvd
      optimizer = hvd.DistributedOptimizer(optimizer)
    model = resnet_cifar_model.resnet56(classes=cifar_main.NUM_CLASSES)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  run_eagerly=flags_obj.run_eagerly,
                  metrics=['categorical_accuracy'])
    autoscaling_callback.set_model(model)

  callbacks = keras_common.get_callbacks(
      learning_rate_schedule,
      cifar_main.NUM_IMAGES['train'],
      autoscaling_callback.num_batches_processed_this_epoch)
  callbacks.append(autoscaling_callback)

  train_steps = cifar_main.NUM_IMAGES['train'] // flags_obj.batch_size
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1

  original_train_steps = train_steps
  original_train_epochs = train_epochs

  # If we restarted in the middle of an epoch, finish the rest of the batches in the
  # epoch first, then restart again with the original number of batches in an epoch
  if autoscaling_callback.num_batches_processed_this_epoch > 0:
    train_steps -= autoscaling_callback.num_batches_processed_this_epoch
    tf.compat.v1.logging.info("There are %s/%s batches left in this epoch" %\
      (train_steps, original_train_steps))
    train_epochs = 1
  else:
    # Otherwise, just finish the remaining epochs
    train_epochs -= autoscaling_callback.num_epochs_processed
    tf.compat.v1.logging.info("There are %s/%s epochs left" %\
      (train_epochs, original_train_epochs))

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

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=train_steps,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)

  # If we finished all the epochs already, then signal to above that we're terminating
  if autoscaling_callback.num_epochs_processed == original_train_epochs:
    autoscaling_callback.agent.status = AutoscalingStatus.TERMINATED

  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()

  stats = keras_common.build_stats(history, eval_output, callbacks)

  if flags_obj.use_horovod:
    import horovod.tensorflow.keras as hvd
    hvd.shutdown()

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
    return run(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_cifar_flags()
  absl_app.run(main)
