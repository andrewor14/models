#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/step_time_overhead.pdf" % OUTPUT_DIR
ALL_MODELS = ["ResNet-50", "VGG-16", "AlexNet", "Transformer", "BERT"]
LABELS = ["orig TF", "1 vnode", "2 vnodes", "4 vnodes", "8 vnodes"]
COLOR_CYCLE = ["red", "blue", "green", "orange", "magenta"]
BAR_WIDTH = 0.1
BAR_INDEXES = np.arange(len(ALL_MODELS))

# TODO: replace with real values
all_values = [
  np.array((660, 880, 1000, 700, 800)),
  np.array((670, 890, 1050, 780, 830)),
  np.array((700, 900, 1100, 800, 840)),
  np.array((720, 920, 1150, 820, 860)),
  np.array((720, 980, 1180, 830, 880))
]

# Plot it
all_bars = []
for i in range(len(LABELS)):
  offset = (i - 2) * BAR_WIDTH  # TODO: express this in terms of len(LABELS)
  all_bars.append(plt.bar(BAR_INDEXES + offset, all_values[i], BAR_WIDTH, color=COLOR_CYCLE[i]))

plt.ylabel("Step time (ms)", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, ALL_MODELS, rotation=30, fontsize=20)
plt.yticks(fontsize=20)
plt.legend(all_bars, LABELS, loc="center left", bbox_to_anchor=(1, 0.5))
plt.margins(0.05, 0)
plt.subplots_adjust(left=0.2, right=0.75, bottom=0.2)

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)

