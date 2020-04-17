#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/heterogeneous_run_time.pdf" % OUTPUT_DIR
ALL_COMBINATIONS = ["16x V100", "16x V100 +\n16x K80", "16x V100 +\n16x K80 +\n16x M60"]
BAR_WIDTH = 0.5
BAR_INDEXES = np.arange(len(ALL_COMBINATIONS))

# TODO: replace with real values
#all_values = [1.78, 1.34, 1.13]
all_values = [18.5, 13.3, 10.8]

# Plot it
bars = plt.bar(BAR_INDEXES - BAR_WIDTH / 2, all_values, BAR_WIDTH, color="orange")

plt.ylabel("End-to-end run time (hr)", fontsize=24, labelpad=20)
plt.xlabel("GPU combinations", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, ALL_COMBINATIONS, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0, max(all_values)*1.1))
plt.margins(0.1, 0)
plt.tight_layout()

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

