# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Defining common flags used across all BERT models/applications."""

import os

from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from virtual import virtual_helper


def define_gin_flags():
  """Define common gin configurable flags."""
  flags.DEFINE_multi_string('gin_file', None,
                            'List of paths to the config files.')
  flags.DEFINE_multi_string(
      'gin_param', None, 'Newline separated list of Gin parameter bindings.')


def define_common_bert_flags():
  """Define common flags for BERT tasks."""
  flags_core.define_base(
      data_dir=False,
      model_dir=True,
      clean=False,
      train_epochs=False,
      epochs_between_evals=False,
      stop_threshold=False,
      batch_size=False,
      num_gpu=True,
      hooks=False,
      export_dir=False,
      distribution_strategy=True,
      run_eagerly=True)
  flags_core.define_distribution()
  flags.DEFINE_string('bert_config_file', None,
                      'Bert configuration file to define core bert layers.')
  flags.DEFINE_string(
      'model_export_path', None,
      'Path to the directory, where trainined model will be '
      'exported.')
  flags.DEFINE_string('tpu', '', 'TPU address to connect to.')
  flags.DEFINE_string(
      'init_checkpoint', None,
      'Initial checkpoint (usually from a pre-trained BERT model).')
  flags.DEFINE_integer('num_train_epochs', 3,
                       'Total number of training epochs to perform.')
  flags.DEFINE_integer(
      name='num_train_steps', default=None,
      help='The number of steps to run for training in each epoch. If it is '
      'larger than # batches per epoch, then use # batches per epoch.')
  flags.DEFINE_integer(
      'steps_per_loop', 200,
      'Number of steps per graph-mode loop. Only training step '
      'happens inside the loop. Callbacks will not be called '
      'inside.')
  flags.DEFINE_float('learning_rate', 5e-5,
                     'The initial learning rate for Adam.')
  flags.DEFINE_boolean(
      'scale_loss', False,
      'Whether to divide the loss by number of replica inside the per-replica '
      'loss function.')
  flags.DEFINE_boolean(
      'use_keras_compile_fit', False,
      'If True, uses Keras compile/fit() API for training logic. Otherwise '
      'use custom training loop.')
  flags.DEFINE_string(
      'hub_module_url', None, 'TF-Hub path/url to Bert module. '
      'If specified, init_checkpoint flag should not be used.')
  flags.DEFINE_bool('hub_module_trainable', True,
                    'True to make keras layers in the hub module trainable.')
  flags.DEFINE_integer(
      name='num_virtual_nodes_per_device', default=1,
      help='Number of virtual nodes mapped to each device in each batch. '
      'Virtual nodes are processed one after another, with the number of examples '
      'processed per virtual node equal to the per device batch size divided by '
      'this value.')
  flags.DEFINE_boolean(
      name='enable_checkpoints', default=False,
      help='Whether to enable a checkpoint callback and export the saved model.')
  flags.DEFINE_integer(
      name='num_checkpoints_to_keep', default=5,
      help='Number of most recent checkpoints to keep, only read if '
      '`enable_checkpoints` is set.')
  flags.DEFINE_boolean(
      name='enable_monitor_memory', default=False,
      help='Whether to enable a callback that periodically monitors GPU memory usage.')
  flags.DEFINE_boolean(
      name='enable_elasticity', default=False,
      help='Whether to enable a callback that provides resource elasticity.')
  flags.DEFINE_boolean(name='skip_eval', default=False, help='Skip evaluation')

  flags_core.define_log_steps()

  # Adds flags for mixed precision and multi-worker training.
  flags_core.define_performance(
      num_parallel_calls=False,
      inter_op=False,
      intra_op=False,
      synthetic_data=False,
      max_train_steps=False,
      dtype=True,
      dynamic_loss_scale=True,
      loss_scale=True,
      all_reduce_alg=True,
      num_packs=False,
      tf_gpu_thread_mode=True,
      datasets_num_private_threads=True,
      enable_xla=True,
      fp16_implementation=True,
  )


def get_callbacks(
    batch_size=None,
    model_dir=None,
    log_steps=None,
    enable_summaries=False,
    enable_checkpoints=False,
    num_checkpoints_to_keep=None,
    enable_monitor_memory=False,
    enable_elasticity=False):
  """Returns common callbacks."""
  callbacks = []
  if log_steps is not None:
    if batch_size is None or model_dir is None:
      raise ValueError("TimeHistory callback requires batch size and model_dir")
    callbacks.append(keras_utils.TimeHistory(
        batch_size=batch_size,
        log_steps=log_steps,
        logdir=model_dir,
    ))
  if enable_summaries:
    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    callbacks.append(summary_callback)
  if enable_checkpoints:
    checkpoint_path = os.path.join(model_dir, 'checkpoint')
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
      checkpoint_path, save_weights_only=True))
    callbacks.append(virtual_helper.DeleteOldCheckpointsCallback(
      model_dir, num_checkpoints_to_keep))
  if enable_monitor_memory:
    callbacks.append(virtual_helper.MonitorMemoryCallback())
  if enable_elasticity:
    from virtual.elasticity_callback import ELASTICITY_CALLBACK
    if ELASTICITY_CALLBACK is None:
      raise ValueError("Singleton elasticity callback was None")
    callbacks.append(ELASTICITY_CALLBACK)
  return callbacks


def dtype():
  return flags_core.get_tf_dtype(flags.FLAGS)


def use_float16():
  return flags_core.get_tf_dtype(flags.FLAGS) == tf.float16


def use_graph_rewrite():
  return flags.FLAGS.fp16_implementation == 'graph_rewrite'


def get_loss_scale():
  return flags_core.get_loss_scale(flags.FLAGS, default_for_fp16='dynamic')
