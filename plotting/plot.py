#!/usr/bin/env python

import matplotlib.pyplot as plt 
import math
import numpy as np
import os
import re
import scipy.stats as stats
from subprocess import Popen, PIPE
import sys 


args = sys.argv
if len(args) <= 2:
  print "Usage: plot.py [title] [log_file1] [log_file2] ..."
  sys.exit(1)
title = args[1]
log_files = args[2:]
out_file = "output.png"

# Prepare plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Plot data from each file
for log_file in log_files:
  Popen(['./parse.py', log_file]).communicate()
  csv_file = re.sub("\..*$", "", log_file) + ".csv"
  step = []
  loss = []
  learning_rate = []
  cross_entropy = []
  train_accuracy = []
  global_step_per_sec = []
  with open(csv_file, "r") as f:
    lines = f.readlines()[1:] # skip header
    for line in lines:
      s = line.split(",")
      step.append(int(s[0]))
      loss.append(float(s[1]))
      learning_rate.append(float(s[2]))
      cross_entropy.append(float(s[3]))
      train_accuracy.append(float(s[4]))
      global_step_per_sec.append(float(s[5]))
  name = log_file.lstrip("slurm-")
  name = re.sub("\..*$", "", name)
  name = re.sub("[-0-9]+$", "", name)
  #ax.plot(step, loss, "-x", label="loss (%s)" % name)
  #ax.plot(step, learning_rate, "-x", label="learning_rate (%s)" % name)
  #ax.plot(step, cross_entropy, "-x", label="cross_entropy (%s)" % name)
  ax.plot(step, train_accuracy, "-x", label="train accuracy (%s)" % name)
  #ax.plot(step[1:], global_step_per_sec[1:], "-x", label="global_step_per_sec (%s)" % name)
  os.remove(csv_file)

# Finalize plot
ax.set_xlabel("step")
ax.set_ylabel("value")
#legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1),\
#  fancybox=True, shadow=True)
legend = ax.legend(loc="lower center")
ax.set_title(title)
fig.savefig(out_file, bbox_extra_artists=(legend,), bbox_inches='tight')

print "Wrote to " + out_file

