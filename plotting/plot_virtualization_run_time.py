#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/virtualization_run_time.pdf" % OUTPUT_DIR
LABELS = ["1", "8", "16", "32", "64", "    64 (TF)"]
BAR_WIDTH = 0.5
BAR_INDEXES = np.arange(len(LABELS))

# TODO: replace with real values
all_values = [23.9, 3.37, 1.78, 0.935, 0.491, 0.491]
#all_values = [32.4, 18.5, 10.5, 6, 6]

# Plot it
bars = plt.bar(BAR_INDEXES - BAR_WIDTH / 2, all_values, BAR_WIDTH, color="blue")

plt.ylabel("End-to-end run time (hr)", fontsize=24, labelpad=20)
plt.xlabel("Number of GPUs", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, LABELS, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0, max(all_values)*1.1))
plt.margins(0.1, 0)
plt.tight_layout()

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

