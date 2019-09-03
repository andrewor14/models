#!/usr/bin/env python3

import os

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import backend as K

from autoscaling import autoscaling_helper
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
    """
    progress = self.agent.client.master_server.get_progress()
    if progress is not None:
      self.num_batches_processed_this_epoch = progress[0]
      self.num_epochs_processed = progress[1]
      self.num_batches_per_epoch = progress[2]
      # Tell tensorflow which step and epoch to restart from
      autoscaling_helper.STEP_NUMBER = self.num_batches_processed_this_epoch
      autoscaling_helper.EPOCH_NUMBER = self.num_epochs_processed
      log_fn("Fetched progress from master server = (%s steps, %s epochs)" %\
        (self.num_batches_processed_this_epoch, self.num_epochs_processed))
    else:
      log_fn("Warning: unable to fetch progress from master server")

  def do_on_batch_begin(self, batch, logs):
    """
    Restore saved variables from memory, if any, before running the first step.
    """
    if self.loaded_model is not None:
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
    # Check if we need to reinitialize
    is_new_worker = not self.agent.joined
    should_initialize = self.agent.step_end()
    # If we are removed from the cluster, restart training in detached mode
    if self.agent.status == AutoscalingStatus.TERMINATED:
      self.num_batches_processed_this_epoch = 0
      self.num_epochs_processed = 0
      autoscaling_helper.STEP_NUMBER = 0
      autoscaling_helper.EPOCH_NUMBER = 0
      self.agent.detach_from_cluster()
    # If we are reinitializing, the new worker should restore variables from existing workers
    if should_initialize:
      if not is_new_worker and not self.agent.detached_mode:
        self.agent.save_variables(self.get_trainable_variables(), for_new_worker=True)
      # This call gathers `self.agent.saved_variables` across the cluster
      # Therefore, we save the variables before this call and restore them after this call
      self.agent.initialize()
      if is_new_worker:
        self.bootstrap_progress()
        self.agent.restore_variables(self.get_trainable_variables())
      autoscaling_helper.initialize_horovod(self.agent.mpi_communicator, restarting=True)

  def do_on_train_end(self, logs):
    """
    Save our variables for the next restart.
    """
    self.agent.status = AutoscalingStatus.TERMINATED
    if self.num_epochs_processed != self.num_epochs_total:
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
        self.agent.mpi_communicator.rank == 0:
      checkpoint_file = os.path.join(checkpoint_dir, AUTOSCALING_CHECKPOINT_FILE_NAME)
      checkpoint_metadata_file = os.path.join(
        checkpoint_dir, AUTOSCALING_CHECKPOINT_METADATA_FILE_NAME)
      log_fn("Saving checkpoint to %s" % checkpoint_file)
      log_fn("Saving checkpoint metadata to %s" % checkpoint_metadata_file)
      self.model.save(checkpoint_file)
      with open(checkpoint_metadata_file, "w") as f:
        metadata = {
          AUTOSCALING_NUM_BATCHES_PROCESSED_THIS_EPOCH: self.num_batches_processed_this_epoch,
          AUTOSCALING_NUM_EPOCHS_PROCESSED: self.num_epochs_processed
        }
        metadata.update(self.agent.checkpoint_restart_variables)
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

def log_fn(msg):
  tf.logging.info("[Autoscaling callback] %s" % msg)

