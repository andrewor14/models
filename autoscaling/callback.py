#!/usr/bin/env python3

import os

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from autoscaling.agent import log_exceptions
from autoscaling.params import *


class AutoscalingCallback(keras.callbacks.Callback):
  """
  A `keras.callbacks.Callback` that keeps track of autoscaling state for this process.
  """
  def __init__(self, agent):
    self.agent = agent
    self.model = None
    self.num_batches_per_epoch = 0
    self.num_epochs_total = 0
    self.num_batches_processed_this_epoch =\
      int(os.getenv(AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH, 0))
    self.num_epochs_processed = int(os.getenv(AUTOSCALING_NUM_EPOCHS_PROCESSED, 0))
    # Run this callback on all the workers
    self._chief_worker_only = False
    # Expose our progress method to the autoscaling service through our agent
    self.agent.get_progress_method = self.get_progress
    self.agent.bootstrap_progress_method = self.bootstrap_progress
    # In 'checkpoint-restart' mode, we need to load the model from checkpoints
    # However, we cannot load the model in a distribution strategy scope.
    # See https://github.com/tensorflow/tensorflow/issues/30850.
    # As a workaround, we will load it here and transfer the weights later.
    self.loaded_model = self.maybe_load_model()

  def set_model(self, model):
    self.model = model

  def reset(self):
    self.model = None

  def get_progress(self):
    """
    Return a 3-tuple:
      (1) Number of batches processed in this epoch so far,
      (2) Number of epochs processed so far, and
      (3) Number of batches per epoch
    """
    return (
      self.num_batches_processed_this_epoch,
      self.num_epochs_processed,
      self.num_batches_per_epoch)

  def bootstrap_progress(self):
    """
    Bootstrap this worker so it can start on the same step as everyone else.

    We do this by fetching the progress from the master autoscaling server.
    Note: the caller should ensure that we do this after the master is READY_TO_SYNC,
    otherwise the progress may be wrong if the master is still running batches.

    Return a 3-tuple:
      (1) Number of batches processed in this epoch so far,
      (2) Number of epochs processed so far, and
      (3) Number of batches per epoch
    """
    progress = self.agent.client.master_server.get_progress()
    if progress is not None:
      self.num_batches_processed_this_epoch = progress[0]
      self.num_epochs_processed = progress[1]
      self.num_batches_per_epoch = progress[2]
      log_fn("Fetched progress from master server = (%s steps, %s epochs)" %\
        (self.num_batches_processed_this_epoch, self.num_epochs_processed))
    else:
      log_fn("Warning: unable to fetch progress from master server")
    return progress

  def do_on_batch_begin(self, batch, logs):
    """
    Restore saved variables from memory, if any, before running the first step.
    """
    if self.agent.saved_variables is not None:
      self.agent.restore_variables(self.get_trainable_variables(), self.get_session())
    elif self.loaded_model is not None:
      self.model.set_weights(self.loaded_model.get_weights())
      self.loaded_model = None

  def do_on_batch_end(self, batch, logs):
    """
    Listen for changes in cluster membership and react by restarting the server.
    """
    # Update counters
    self.num_batches_processed_this_epoch += 1
    if self.num_batches_processed_this_epoch == self.num_batches_per_epoch:
      self.num_epochs_processed += 1
      self.num_batches_processed_this_epoch = 0
    # Check if we need to restart
    restarting = self.agent.step_end()
    if restarting:
      self.model.stop_training = True

  def do_on_train_end(self, logs):
    """
    Save our variables for the next restart if we are not terminating.
    """
    if self.num_epochs_processed == self.num_epochs_total:
      self.agent.status = AutoscalingStatus.TERMINATED
    if self.agent.status != AutoscalingStatus.TERMINATED:
      self.agent.save_variables(self.get_trainable_variables(), self.get_session())
    else:
      self.maybe_save_model()

  def get_trainable_variables(self):
    """
    Return a list of trainable variables.
    """
    if self.model is None:
      raise ValueError("Model must be set before the first step.")
    return self.model.trainable_variables

  def maybe_load_model(self):
    """
    Restore model from checkpoint if we are running in 'checkpoint-restart' mode
    and the checkpoint file exists.
    """
    model = None
    if is_checkpoint_restart_mode():
      checkpoint_dir = os.getenv(AUTOSCALING_CHECKPOINT_DIR)
      if checkpoint_dir is not None and os.path.exists(checkpoint_dir):
        checkpoint_file = os.path.join(checkpoint_dir, AUTOSCALING_CHECKPOINT_FILE_NAME)
        if os.path.isfile(checkpoint_file):
          log_fn("Restoring checkpoint from %s" % checkpoint_file)
          model = keras.models.load_model(checkpoint_file)
          log_fn("Checkpoint restored")
    return model

  def maybe_save_model(self):
    """
    Save model to checkpoint if we are running in 'checkpoint-restart' mode and a
    new number of workers is specified.
    """
    checkpoint_dir = os.getenv("TRAIN_DIR")
    if is_checkpoint_restart_mode() and\
        os.path.exists(checkpoint_dir) and\
        self.agent.status != AutoscalingStatus.TERMINATED:
      checkpoint_file = os.path.join(checkpoint_dir, AUTOSCALING_CHECKPOINT_FILE_NAME)
      checkpoint_metadata_file = os.path.join(
        checkpoint_dir, AUTOSCALING_CHECKPOINT_METADATA_FILE_NAME)
      log_fn("Saving checkpoint to %s" % checkpoint_file)
      log_fn("Saving checkpoint metadata to %s" % checkpoint_metadata_file)
      self.model.save(checkpoint_file)
      with open(checkpoint_metadata_file, "w") as f:
        metadata = {
          "NUM_WORKERS": self.agent.checkpoint_restart_num_workers,
          AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH: self.num_batches_processed_this_epoch,
          AUTOSCALING_NUM_EPOCHS_PROCESSED: self.num_epochs_processed
        }
        for k, v in metadata.items():
          f.write("%s %s\n" % (k, v))
      log_fn("Checkpoint saved")

  # ================== HELPER METHODS ==================

  def on_batch_begin(self, batch, logs=None):
    log_exceptions(lambda: self.do_on_batch_begin(batch, logs))

  def on_batch_end(self, batch, logs=None):
    log_exceptions(lambda: self.do_on_batch_end(batch, logs))

  def on_train_end(self, logs=None):
    log_exceptions(lambda: self.do_on_train_end(logs))

  def get_session(self):
    return None if tf.executing_eagerly() else K.get_session()

def log_fn(msg):
  tf.logging.info("[Autoscaling callback] %s" % msg)

