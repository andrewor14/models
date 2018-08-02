# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Synchronize replicas for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.util.tf_export import tf_export


# Please note that the gradients from replicas are averaged instead of summed
# (as in the old k_sync_optimizer) so you need to increase the learning
# rate according to the number of replicas. This change is introduced to be
# consistent with how gradients are aggregated (averaged) within a batch in a
# replica.
@tf_export("train.KSyncOptimizer")
class KSyncOptimizer(optimizer.Optimizer):
  """Class to synchronize, aggregate gradients and pass them to the optimizer.

  Note: This optimizer borrows heavily from SyncReplicasOptimizer.

  In a typical asynchronous training environment, it's common to have some
  stale gradients. For example, with a N-replica asynchronous training,
  gradients will be applied to the variables N times independently. Depending
  on each replica's training speed, some gradients might be calculated from
  copies of the variable from several steps back (N-1 steps on average). This
  optimizer avoids stale gradients by collecting gradients from all replicas,
  averaging them, then applying them to the variables in one shot, after
  which replicas can fetch the new variables and continue.

  The following accumulators/queue are created:
  <empty line>
  * N `gradient accumulators`, one per variable to train. Gradients are pushed
    to them and the chief worker will wait until enough gradients are collected
    and then average them before applying to variables. The accumulator will
    drop all stale gradients (more details in the accumulator op).
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all variables are updated.

  The following local variable is created:
  * `sync_rep_local_step`, one per replica. Compared against the global_step in
    each accumulator to check for staleness of the gradients.

  The optimizer adds nodes to the graph to collect gradients and pause the
  trainers until variables are updated.
  For the Parameter Server job:
  <empty line>
  1. An accumulator is created for each variable, and each replica pushes the
     gradients into the accumulators instead of directly applying them to the
     variables.
  2. Each accumulator averages once enough gradients (replicas_to_aggregate)
     have been accumulated.
  3. Apply the averaged gradients to the variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, pushes `global_step` in the `token_queue`, once for
     each worker replica. The workers can now fetch the global step, use it to
     update its local_step variable and start the next batch.

  For the replicas:
  <empty line>
  1. Start a step: fetch variables and compute gradients.
  2. Once the gradients have been computed, push them into gradient
     accumulators. Each accumulator will check the staleness and drop the stale.
  3. After pushing all the gradients, dequeue an updated value of global_step
     from the token queue and record that step to its local_step variable. Note
     that this is effectively a barrier.
  4. Start the next batch.

  ### Usage

  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Wrap the optimizer with k_sync_optimizer with 50 replicas: at each
  # step the optimizer collects 50 gradients before applying to variables.
  # Note that if you want to have 2 backup replicas, you can change
  # total_num_replicas=52 and make sure this number matches how many physical
  # replicas you started in your job.
  opt = tf.train.KSyncOptimizer(opt, replicas_to_aggregate=50,
                                 total_num_replicas=50)

  # Some models have startup_delays to help stabilize the model but when using
  # k_sync training, set it to 0.

  # Now you can call `minimize()` or `compute_gradients()` and
  # `apply_gradients()` normally
  training_op = opt.minimize(total_loss, global_step=self.global_step)


  # You can create the hook which handles initialization and queues.
  k_sync_hook = opt.make_session_run_hook(is_chief)
  ```

  In the training program, every worker will run the train_op as if not
  synchronized.

  ```python
  with training.MonitoredTrainingSession(
      master=workers[worker_id].target, is_chief=is_chief,
      hooks=[k_sync_hook]) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(training_op)
  ```

  To use KSyncOptimizer with an `Estimator`, you need to send
  k_sync_hook while calling the fit.
  ```python
  my_estimator = DNNClassifier(..., optimizer=opt)
  my_estimator.fit(..., hooks=[k_sync_hook])
  ```
  """

  def __init__(self,
               opt,
               starting_replicas_to_aggregate,
               total_num_replicas=None,
               variable_averages=None,
               variables_to_average=None,
               use_locking=False,
               scaling_duration=10000,
               name="k_sync"):
    """Construct a k_sync optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      starting_replicas_to_aggregate: starting number of replicas to aggregate
        for each variable update.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate. Must be > replicas_to_aggregate.
      scaling_duration: How many steps before the number of replicas to aggregate
        reaches its final value (total_num_replicas). If scaling_duration is -1,
        there will be no scaling, i.e. the number of replicas to aggregate stays
        at starting_replicas_to_aggregate.
      variable_averages: Optional `ExponentialMovingAverage` object, used to
        maintain moving averages for the variables passed in
        `variables_to_average`.
      variables_to_average: a list of variables that need to be averaged. Only
        needed if variable_averages is passed in.
      use_locking: If True use locks for update operation.
      name: string. Optional name of the returned operation.
    """
    if total_num_replicas is None:
      total_num_replicas = starting_replicas_to_aggregate

    if starting_replicas_to_aggregate > total_num_replicas:
      raise ValueError("Number of replicas to aggregate (%s) exceeds total number of replicas (%s)"
                       % (starting_replicas_to_aggregate, total_num_replicas))

    super(KSyncOptimizer, self).__init__(use_locking, name)
    logging.info(
        "KSyncReplicaOptimizer: starting_replicas_to_aggregate=%s; total_num_replicas=%s",
        starting_replicas_to_aggregate, total_num_replicas)
    self._opt = opt
    self._starting_replicas_to_aggregate = starting_replicas_to_aggregate
    self._replicas_to_aggregate = None
    self._gradients_applied = False
    self._variable_averages = variable_averages
    self._variables_to_average = variables_to_average
    self._total_num_replicas = total_num_replicas
    self._scaling_duration = scaling_duration
    self._local_step = None
    self._global_step = None
    self._sync_token_queue = None
    self.chief_init_op = None
    self.local_step_init_op = None
    self.ready_for_local_init_op = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

  def _log(self, msg):
    logging.info("ANDREW(k_sync): %s" % msg)

  def compute_gradients(self, *args, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    self._log("Computing gradients...")
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    msg = "Applying gradients (%s of them):" % len(grads_and_vars)
    for g, _ in grads_and_vars[:10]:
      msg += "\n    %s" % str(g)
    self._log(msg)

    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    self._replicas_to_aggregate = self.get_num_replicas_to_aggregate()
    train_ops = []
    aggregated_grad = []
    var_list = []

    # local_anchor op will be placed on this worker task by default.
    local_anchor = control_flow_ops.no_op()
    # Colocating local_step variable prevents it being placed on the PS.
    with ops.colocate_with(local_anchor):
      self._local_step = variable_scope.variable(
          initial_value=0,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=global_step.dtype.base_dtype,
          name="sync_rep_local_step")

    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())

    with ops.name_scope(None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        with ops.device(var.device):
          # Dense gradients.
          if grad is None:
            aggregated_grad.append(None)  # pass-through.
            continue
          elif isinstance(grad, ops.Tensor):
            grad_accum = data_flow_ops.ConditionalAccumulator(
                grad.dtype,
                shape=var.get_shape(),
                shared_name=var.name + "/grad_accum")
            # Set the local step as global step here to avoid dropping stale gradients
            train_ops.append(grad_accum.apply_grad(
                grad, local_step=self._global_step))
            aggregated_grad.append(grad_accum.take_grad(
                self._replicas_to_aggregate))
          else:
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError("Unknown grad type!")
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
                grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
            train_ops.append(grad_accum.apply_indexed_slices_grad(
                grad, local_step=self._global_step))
            aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((grad_accum, var.device))

      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""):
        update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                              global_step)

      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()
          self._log("Dequeue token op: %s" % token)
        train_op = state_ops.assign(self._local_step, token)

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          tokens = array_ops.fill([self._total_num_replicas], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))
          self._log("Enqueue token op: %s" % str(sync_op))

        if self._variable_averages is not None:
          with ops.control_dependencies([sync_op]), ops.name_scope(""):
            sync_op = self._variable_averages.apply(
                self._variables_to_average)

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
              accum.set_global_step(
                  global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True
      return train_op

  def get_num_replicas_to_aggregate(self):
    """Returns a schedule for the number of replicas to aggregate in each step.

    Returns:
      A 0-D Tensor whose value depends on the schedule and the global step.
    """
    start = self._starting_replicas_to_aggregate
    end = self._total_num_replicas
    if self._scaling_duration < 0 or start == end:
      return tf.Variable(start, tf.int32)
    num_boundaries = end - start
    num_values = num_boundaries + 1
    n = self._scaling_duration / num_boundaries
    values = [start + i for i in range(num_values)]
    boundaries = [i * n for i in range(1, num_values)]
    self._log("Schedule for number of replicas to aggregate: "
              "boundaries = %s, values = %s, interval = %s" % (boundaries, values, n))
    return tf.train.piecewise_constant(self._global_step, boundaries, values)

  def get_chief_queue_runner(self):
    """Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: aggregate gradients,
    apply to variables, increment global step, insert tokens to token queue.

    Note that this can only be called after calling apply_gradients() which
    actually generates this queuerunner.

    Returns:
      A `QueueRunner` for chief to execute.

    Raises:
      ValueError: If this is called before apply_gradients().
    """
    if self._gradients_applied is False:
      raise ValueError("Should be called after apply_gradients().")

    return self._chief_queue_runner

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def variables(self):
    """Fetches a list of optimizer variables in the default graph.

    This wraps `variables()` from the actual optimizer. It does not include
    the `KSyncOptimizer`'s local step.

    Returns:
      A list of variables.
    """
    return self._opt.variables()

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def get_init_tokens_op(self, num_tokens=-1):
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    tokens_needed = self._starting_replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._starting_replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    self._log("Initializing with %s token(s)" % num_tokens)
    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        self._log("Initial enqueue token op: %s" % str(tokens))
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens

  def make_session_run_hook(self, is_chief, num_tokens=-1):
    """Creates a hook to handle SyncReplicasHook ops such as initialization."""
    return _KSyncOptimizerHook(self, is_chief, num_tokens)


class _KSyncOptimizerHook(session_run_hook.SessionRunHook):
  """A SessionRunHook handles ops related to KSyncOptimizer."""

  def __init__(self, sync_optimizer, is_chief, num_tokens):
    """Creates hook to handle KSyncOptimizer initialization ops.

    Args:
      sync_optimizer: `KSyncOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      num_tokens: Number of tokens to add to the queue.
    """
    self._sync_optimizer = sync_optimizer
    self._is_chief = is_chief
    self._num_tokens = num_tokens
    self._epoch = 0
    self._session = None
    self._local_init_op = None
    self._ready_for_local_init_op = None
    self._q_runner = None
    self._init_tokens_op = None

  def _log(self, msg):
    self._sync_optimizer._log(msg)

  def begin(self):
    if self._sync_optimizer._gradients_applied is False:  # pylint: disable=protected-access
      raise ValueError(
          "KSyncOptimizer.apply_gradient should be called before using "
          "the hook.")
    if self._is_chief:
      self._local_init_op = self._sync_optimizer.chief_init_op
      self._ready_for_local_init_op = (
          self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = self._sync_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._sync_optimizer.get_init_tokens_op(
          self._num_tokens)
    else:
      self._local_init_op = self._sync_optimizer.local_step_init_op
      self._ready_for_local_init_op = (
          self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    """Runs KSyncOptimizer initialization ops."""
    local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
        self._ready_for_local_init_op, session,
        "Model is not ready for KSyncOptimizer local init.")
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for KSyncOptimizer "
          "local_init. Init op: %s, error: %s" %
          (self._local_init_op.name, msg))
    self._session = session
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(
          session, coord=coord, daemon=True, start=True)

  def after_run(self, run_context, run_values):
    self._epoch += 1
    opt = self._sync_optimizer
    gs = int(self._session.run(opt._global_step))
    ls = int(self._session.run(opt._local_step))
    rta = int(self._session.run(opt._replicas_to_aggregate))
    self._log("AFTER RUN epoch = %s, gs = %s, ls = %s, rta = %s" % (self._epoch, gs, ls, rta))
