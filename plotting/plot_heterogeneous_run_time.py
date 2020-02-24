#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/heterogeneous_run_time.pdf" % OUTPUT_DIR
ALL_COMBINATIONS = ["8x V100", "8x V100 +\n8x K80", "8x V100 +\n8x K80 +\n8x M60"]
BAR_WIDTH = 0.5
BAR_INDEXES = np.arange(len(ALL_COMBINATIONS))

# TODO: replace with real values
all_values = [10.2, 6.8, 5.45]

# Plot it
bars = plt.bar(BAR_INDEXES - BAR_WIDTH / 2, all_values, BAR_WIDTH)

plt.ylabel("End-to-end run time (hr)", fontsize=24, labelpad=20)
plt.xlabel("GPU combinations", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, ALL_COMBINATIONS, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0, 11))
plt.margins(0.1, 0)
plt.tight_layout()

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

