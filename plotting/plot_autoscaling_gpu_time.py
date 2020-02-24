#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/autoscaling_gpu_time.pdf" % OUTPUT_DIR
STARTING_NUM_GPUS = ["16", "24", "32", "48", "56", "64"]
LABELS = ["static", "autoscaling"]
BAR_WIDTH = 0.35
BAR_INDEXES = np.arange(len(STARTING_NUM_GPUS))

# TODO: replace with real values
static_values = [700, 710, 720, 730, 740, 750]
autoscaling_values = [740, 750, 745, 755, 746, 752]

# Plot it
static_bar = plt.bar(BAR_INDEXES - BAR_WIDTH, static_values, BAR_WIDTH, color="orange")
autoscaling_bar = plt.bar(BAR_INDEXES, autoscaling_values, BAR_WIDTH, color="blue")

plt.ylabel("GPU time (GPU-hours)", fontsize=24, labelpad=20)
plt.xlabel("Starting number of GPUs", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, STARTING_NUM_GPUS, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0, 1000))
plt.legend([static_bar, autoscaling_bar], LABELS)
plt.margins(0.05, 0)
plt.tight_layout()

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

