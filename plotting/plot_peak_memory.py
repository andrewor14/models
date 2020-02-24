#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"
OUTPUT_PATH = "%s/peak_memory.pdf" % OUTPUT_DIR
ALL_MODELS = ["ResNet-50", "VGG-16", "AlexNet", "Transformer", "BERT"]
ALL_CATEGORIES = ["inputs", "model", "gradients", "activations", "gradient_buffer"]
COLOR_CYCLE = ["red", "blue", "green", "orange", "magenta"]
BAR_WIDTH = 0.35
BAR_INDEXES = np.arange(len(ALL_MODELS))

# TODO: replace with real values
all_values = {
  "inputs": np.array((0.015, 0.015, 0.015, 0.02, 0.02)),
  "model": np.array((0.1024, 0.552, 0.244, 0.44, 1.36)),
  "gradients": np.array((0.1024, 0.552, 0.244, 0.44, 1.36)),
  "activations": np.array((8.76, 6.54, 5.88, 9.26, 8.74)),
  "gradient_buffer": np.array((0.1024, 0.552, 0.244, 0.44, 1.36))
}
ALL_CATEGORIES = list(reversed(ALL_CATEGORIES))
COLOR_CYCLE = list(reversed(COLOR_CYCLE))
all_values = [all_values[category] for category in ALL_CATEGORIES]

# Plot it
all_bars = []
for i in range(len(all_values)):
  bottom_bar = np.sum(all_values[:i], axis=0)
  current_bar = plt.bar(BAR_INDEXES, all_values[i], BAR_WIDTH, bottom=bottom_bar, color=COLOR_CYCLE[i])
  all_bars.append(current_bar)

plt.ylabel("Peak memory allocation (GB)", fontsize=24, labelpad=20)
plt.xticks(BAR_INDEXES, ALL_MODELS, rotation=30, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim((0, 16))
plt.legend(reversed(all_bars), reversed(ALL_CATEGORIES), loc="center left", bbox_to_anchor=(1, 0.5))
plt.margins(0.05, 0)
plt.subplots_adjust(left=0.15, right=0.68, bottom=0.2)

print "Saved figure to %s." % OUTPUT_PATH
plt.savefig(OUTPUT_PATH)


