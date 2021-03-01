#!/usr/bin/env python3

global_batch_size = 512
b1 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
b2 = [1, 2, 4, 8, 16, 32, 64]
t1 = [1.0, 0.5, 0.25, 0.125, 0.067, 0.037, 0.022, 0.015, 0.013]
t2 = [4.0, 2.0, 1.0, 0.5, 0.286, 0.2, 0.167]
n1 = [1, 2, 4, 8]
n2 = [1, 2, 4, 8, 16]

min_time = float("inf")
min_points = []
for i1, _b1 in enumerate(b1):
  for i2, _b2 in enumerate(b2):
    for _n1 in n1:
      for _n2 in n2:
        if _b1 * _n1 + _b2 * _n2 != global_batch_size:
          continue
        step_time = max(t1[i1], t2[i2])
        if step_time < min_time:
          min_points = []
        min_points.append((i1, i2, _n1, _n2))

for i1_selected, i2_selected, n1_selected, n2_selected in min_points:
  batch_size_sum = n1_selected * b1[i1_selected] + n2_selected * b2[i2_selected]
  min_step_time = max(t1[i1_selected], t2[i2_selected])
  if batch_size_sum != global_batch_size:
    print("Error: incorrect batch size sum %s" % batch_size_sum)
    break
  print("===========================================================")
  print("Batch size 1 = %s, num GPUs 1 = %s" % (b1[i1_selected], n1_selected))
  print("Batch size 2 = %s, num GPUs 2 = %s" % (b2[i2_selected], n2_selected))
  print("Batch sizes: (%s * %s) + (%s * %s) = %s" %\
    (n1_selected, b1[i1_selected], n2_selected, b2[i2_selected], global_batch_size))
  print("Step time: max(%s, %s) = %s" %\
    (t1[i1_selected], t2[i2_selected], min_step_time))
  
