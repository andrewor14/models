#/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

throughputs = {
  4: [411.462730],
  5: [514.361304],
  6: [604.306791],
  7: [685.393553],
  8: [790.372526],
  9: [898.704328]
}
#throughputs = {
#  4: [400],
#  5: [500],
#  6: [600],
#  7: [700],
#  8: [800],
#  9: [900]
#}

curve_fitting_min_throughputs = 1
curve_fitting_min_points = 5
throughput_increase_threshold = 0.1

def check_spawn_thresholds(old_throughput, new_throughput):
  """ 
  Return whether the increase in throughput exceeds the threshold.
  """
  ratio = (new_throughput - old_throughput) / old_throughput
  print("Old = %s, new = %s, ratio = %s" % (old_throughput, new_throughput, ratio))
  return ratio > throughput_increase_threshold

def get_num_workers_to_spawn(current_num_workers, num_additional_workers):
  """ 
  Return a number of workers to add that satisfies the throughput and cost thresholds.

  The number of workers returned is <= `num_additional_workers`.
  A negative number means that many workers should be removed instead.
  """
  num_workers = []
  average_throughputs = []
  for n, t in throughputs.items():
    if len(t) >= curve_fitting_min_throughputs:
      num_workers.append(n)
      average_throughputs.append(np.mean(t))
  # If we only have 1 data point, always allow adding 1 worker
  if len(average_throughputs) == 1:
    return 1
  # If we don't have enough points to use curve fitting, add 1 worker if the previous
  # increase in throughput satisfies our thresholds
  if len(average_throughputs) < curve_fitting_min_points:
    current_throughput = average_throughputs[num_workers.index(current_num_workers)]
    previous_throughput = average_throughputs[num_workers.index(current_num_workers - 1)]
    return 1 if check_spawn_thresholds(previous_throughput, current_throughput) else 0
  # Otherwise, use curve fitting to estimate the largest additional number of workers
  # under `num_additional_workers` that still satisfies the thresholds
  num_workers = np.array(num_workers)
  average_throughputs = np.array(average_throughputs)
  functions_to_try = [
    lambda x, a, b: a * x + b,
    lambda x, a, b: a - b / x,
    lambda x, a, b: a * np.log(x) + b
  ]
  min_fit_error = None
  best_func = None
  best_fitted_vars = None
  for i, func in enumerate(functions_to_try):
    fitted_vars, _ = curve_fit(func, num_workers, average_throughputs)
    fitted_func = lambda x: func(x, *fitted_vars)
    fit_error = np.mean((average_throughputs - fitted_func(num_workers)) ** 2)
    print("Fitted function %s, variables = %s, error = %s" % (i, fitted_vars, fit_error))
    if min_fit_error is None or fit_error < min_fit_error:
      print("Found new best function %s with error %s" % (i, fit_error))
      min_fit_error = fit_error
      best_func = func
      best_fitted_vars = fitted_vars
  best_fitted_func = lambda x: best_func(x, *best_fitted_vars)

  # Visualize things
  x = num_workers
  y = average_throughputs
  y2 = best_fitted_func(x)
  out_file = "output.pdf"
  fig = plt.figure()
  ax1 = fig.add_subplot(1, 1, 1)
  ax1.errorbar(x, y, fmt=".")
  ax1.errorbar(x, y2, fmt="-")
  plt.xlim(xmin=x[0]-1, xmax=x[-1]+1)
  fig.set_tight_layout({"pad": 1.5})
  fig.savefig(out_file)

  for i in range(num_additional_workers): 
    print("Checking %s additional workers" % i)
    current_throughput = best_fitted_func(current_num_workers + i)
    next_throughput = best_fitted_func(current_num_workers + i + 1)
    # Stop as soon as the thresholds are not satisfied
    if not check_spawn_thresholds(current_throughput, next_throughput):
      return i
  return num_additional_workers 

print(get_num_workers_to_spawn(9, 10))

