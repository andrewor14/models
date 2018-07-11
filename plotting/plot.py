#!/usr/bin/env python

import matplotlib.pyplot as plt 
import math
import numpy as np
import scipy.stats as stats
import sys 

args = sys.argv
if len(args) <= 2:
  print "Usage: plot.py [data_file] [name]"
  sys.exit(1)
data_file = args[1]
out_file = data_file.replace(".csv", ".png")
name = args[2]

# Parse data from file
step = []
loss = []
learning_rate = []
cross_entropy = []
train_accuracy = []
global_step_per_sec = []
with open(data_file, "r") as f:
  lines = f.readlines()[1:] # skip header
  for line in lines:
    s = line.split(",")
    step.append(int(s[0]))
    loss.append(float(s[1]))
    learning_rate.append(float(s[2]))
    cross_entropy.append(float(s[3]))
    train_accuracy.append(float(s[4]))
    global_step_per_sec.append(float(s[5]))

# Plot it
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(step, loss, "-x", label="loss")
ax.plot(step, learning_rate, "-x", label="learning_rate")
#ax.plot(step, cross_entropy, "-x", label="cross_entropy")
ax.plot(step, train_accuracy, "-x", label="train_accuracy")
#ax.plot(step[1:], global_step_per_sec[1:], "-x", label="global_step_per_sec")
ax.set_xlabel("step")
ax.set_ylabel("value")
ax.legend()
ax.set_title(name)
plt.savefig(out_file)

