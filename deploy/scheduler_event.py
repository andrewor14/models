#!/usr/bin/env python3

import threading

class Event:
  """
  Abstract base class for all scheduler events.
  """

  def process(self, scheduler):
    """
    Process this event.

    This assumes the caller holds `scheduler.lock`.
    Return whether the event was processed successfully.
    A return value of False signals the caller to try again later.
    """
    raise ValueError("Not implemented")

  def __str__(self):
    return self.__class__.__name__

class ScheduleEvent(Event):
  """
  An event that triggers the scheduler to adjust its allocations.
  """

  def process(self, scheduler):
    return scheduler.schedule()

class RunJobEvent(Event):
  """
  An event that attempts to run the job.
  """

  def __init__(self, job, initial_allocation=None):
    self.job = job
    self.initial_allocation = initial_allocation or job.workload.gpu_demand

  def process(self, scheduler):
    """
    If there are enough available GPUs, run the job.
    """
    # If the job is already running, there is no need to run it
    if self.job in scheduler.running_jobs:
      return True

    # Check GPU availability
    job_id = self.job.job_id
    available_gpus = scheduler.get_available_gpus(self.initial_allocation)
    should_run = len(available_gpus) == self.initial_allocation
    if should_run:
      # There are enough GPUs, assign them
      for host, gpu_index in available_gpus:
        scheduler.gpu_assignment[host][gpu_index] = job_id
      master_host = available_gpus[0][0]
      scheduler.elasticity_master_hosts[job_id] = master_host
      # Run the job in the background
      # On exit, release all GPUs assigned to this job
      scheduler.running_jobs.append(self.job)
      p = self.job.workload.run(\
        scheduler, job_id, master_host, self.initial_allocation)
      scheduler.log("Job %s started with %s GPUs (PID = %s)" %\
        (job_id, self.initial_allocation, p.pid))
      def wait_for_process(process):
        process.wait()
        scheduler.log("Job %s finished" % job_id)
        with scheduler.lock:
          # Unassign GPUs
          for host in scheduler.gpu_assignment.keys():
            for i, j in enumerate(scheduler.gpu_assignment[host]):
              if j == job_id:
                scheduler.gpu_assignment[host][i] = None
          del scheduler.elasticity_master_hosts[job_id]
          scheduler.running_jobs.remove(self.job)
          # If we are the last job, tell the scheduler to exit
          # Else, trigger schedule event so new jobs can be scheduled
          scheduler.num_jobs_completed += 1
          if scheduler.num_jobs_completed == scheduler.max_num_jobs:
            scheduler.log("Reached maximum number of %s jobs" % scheduler.max_num_jobs)
            scheduler.done = True
          else:
            scheduler.event_queue.append(ScheduleEvent())
      t = threading.Thread(target=wait_for_process, args=(p,))
      t.setDaemon(True)
      t.start()
    return should_run

  def __str__(self):
    return "RunJobEvent(job=%s, initial_allocation=%s)" %\
      (self.job, self.initial_allocation)
  
class ResizeEvent(Event):
  """
  An event that triggers a job to adjust its allocations.
  """

  def __init__(self, job_id, target_num_workers):
    self.job_id = job_id
    self.target_num_workers = target_num_workers
    self.last_exception_str = None

  def process(self, scheduler):
    """
    Request the job to resize to the target number of workers.

    If this is a size up request, we only send the request if we can assign the
    desired number of GPUs to the job. Otherwise, try again later.
    """
    # If the job is no longer running, there is no need to resize
    if self.job_id not in [j.job_id for j in scheduler.running_jobs]:
      return True

    # For size up request, first try to allocate the desired number of GPUs
    # If this is successful, send the request to the job
    allocated_num_workers = scheduler.get_current_allocation(self.job_id)
    if self.target_num_workers > allocated_num_workers:
      new_num_workers = self.target_num_workers - allocated_num_workers
      available_gpus = scheduler.get_available_gpus(new_num_workers)
      # If there are enough available GPUs, assign them to this job
      if len(available_gpus) == new_num_workers:
        for host, gpu_index in available_gpus:
          scheduler.gpu_assignment[host][gpu_index] = self.job_id
      else:
        # There are not enough GPUs yet, so try again later
        return False

    # Send the set num workers request to the job asynchronously
    from virtual.elasticity_callback import get_elasticity_client
    master_host = scheduler.elasticity_master_hosts[self.job_id]
    elasticity_client = get_elasticity_client(master_host, self.job_id)
    t = threading.Thread(
      target=self._send_resize_request,
      args=(elasticity_client, scheduler))
    t.setDaemon(True)
    t.start()
    return True

  def _send_resize_request(self, elasticity_client, scheduler):
    """
    Send a `set_num_workers` request to the job, catching any exception that was thrown.
    This should be called asynchronously.
    """
    try:
      elasticity_client.set_num_workers(self.target_num_workers)
    except Exception as e:
      from deploy.scheduler import DEBUG
      if DEBUG:
        e_str = "(same as above)" if str(e) == self.last_exception_str else str(e)
        scheduler.log("... set_num_workers exception from job %s: %s" % (self.job_id, e_str))
        self.last_exception_str = str(e)
      # Add this event back into the queue manually because this was called asynchronously
      with scheduler.lock:
        scheduler.failed_events.append(self)
      return None

  def __str__(self):
    return "ResizeEvent(job_id=%s, target_num_workers=%s)" %\
      (self.job_id, self.target_num_workers)

