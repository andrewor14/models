import os
import tensorflow as tf
from tensorflow.python.client import device_lib

is_gpu_available = tf.test.is_gpu_available()

print("Tensorflow version: %s" % tf.__version__)
print("All local devices:\n%s" % device_lib.list_local_devices())
print("GPU is available? %s" % is_gpu_available)
print("GPU device name: %s" % (tf.test.gpu_device_name() or "None"))
print("CUDA_VISIBLE_DEVICES = %s" % (os.environ.get("CUDA_VISIBLE_DEVICES") or "N/A"))

if not is_gpu_available:
  raise RuntimeError("No GPU devices detected!")

print ("""==================================================
|                                                |
|        GPU devices detected. Success!          |
|                                                |
==================================================
""")

