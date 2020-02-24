#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/autoscaling_run_time.pdf" % OUTPUT_DIR
STARTING_NUM_GPUS = ["16", "24", "32", "48", "56", "64"]
LABELS = ["static", "autoscaling"]
BAR_WIDTH = 0.35
BAR_INDEXES = np.arange(len(STARTING_NUM_GPUS))

# TODO: replace with real values
static_values = [5.4, 4.2, 2.8, 2.35, 2.0, 1.6]
autoscaling_values = [1.72, 1.69, 1.65, 1.62, 1.68, 1.66]

# Plot it
static_bar = plt.bar(BAR_INDEXES - BAR_WIDTH, static_values, BAR_WIDTH, color="orange")
autoscaling_bar = plt.bar(BAR_INDEXES, autoscaling_values, BAR_WIDTH, color="blue")

plt.ylabel("End-to-end run time (hrs)", fontsize=24, labelpad=20)
plt.xlabel("Starting number of GPUs", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, STARTING_NUM_GPUS, fontsize=20)
plt.yticks(fontsize=20)
plt.legend([static_bar, autoscaling_bar], LABELS)
plt.margins(0.05, 0)
plt.tight_layout()

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

