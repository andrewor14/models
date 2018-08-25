#!/usr/bin/env python

import os
import tensorflow as tf
from tensorflow.python.client import device_lib

is_gpu_available = tf.test.is_gpu_available()

print("Tensorflow version: %s" % tf.__version__)
print("All local devices:\n%s" % str(device_lib.list_local_devices()))
print("GPU is available?%s" % is_gpu_available)
print("GPU device name: %s" % str(tf.test.gpu_device_name()))
print("CUDA_VISIBLE_DEVICES = %s" % str(os.environ.get("CUDA_VISIBLE_DEVICES")))

if not is_gpu_available:
  raise RuntimeError("No GPU devices detected!")

