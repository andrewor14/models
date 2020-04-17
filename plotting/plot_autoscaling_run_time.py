#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/autoscaling_run_time.pdf" % OUTPUT_DIR
#STARTING_NUM_GPUS = ["16", "24", "32", "48", "56", "64"]
STARTING_NUM_GPUS = ["16", "32", "48", "64"]
LABELS = ["static", "autoscaling"]
BAR_WIDTH = 0.35
BAR_INDEXES = np.arange(len(STARTING_NUM_GPUS))

# TODO: replace with real values
#static_values = [1.78, 1.18, 0.935, 0.620, 0.558, 0.491]
#autoscaling_values = [0.62, 0.57, 0.54, 0.58, 0.52, 0.53]
static_values = [18.5, 10.5, 7.02, 6]
autoscaling_values = [6.4, 6.3, 6.2, 6.25]

# Plot it
static_bar = plt.bar(BAR_INDEXES - BAR_WIDTH, static_values, BAR_WIDTH, color="red")
autoscaling_bar = plt.bar(BAR_INDEXES, autoscaling_values, BAR_WIDTH, color="green")

plt.ylabel("End-to-end run time (hrs)", fontsize=24, labelpad=20)
plt.xlabel("Starting number of GPUs", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, STARTING_NUM_GPUS, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0, max(static_values)*1.1))
plt.legend([static_bar, autoscaling_bar], LABELS)
plt.margins(0.05, 0)
plt.tight_layout()

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

