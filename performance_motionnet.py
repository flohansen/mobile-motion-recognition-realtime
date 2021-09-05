import time

from numpy.core.numeric import Inf
from modules.motionnet import MotionNet, MotionNetCNN
import tensorflow as tf

network_map = {
    'motionnet_linear_128_3': MotionNet(3, [128]),
    'motionnet_linear_256_3': MotionNet(3, [256]),
    'motionnet_linear_256_128_3': MotionNet(3, [256, 128]),
    'motionnet_cnn_16_32_3': MotionNetCNN(3, [16, 32]),
    'motionnet_cnn_16_32_64_3': MotionNetCNN(3, [16, 32, 64]),
    'motionnet_cnn_16_32_64_128_3': MotionNetCNN(3, [16, 32, 64, 128]),
}

def measure_time(function, *args):
  start_time = time.perf_counter_ns()

  function(*args)

  end_time = time.perf_counter_ns()
  estimated_ms = (end_time - start_time) / 1000000.0
  return estimated_ms


if __name__ == '__main__':
    for model_name in network_map:
        model = network_map[model_name]
        min_time_ms = Inf

        for i in range(10):
            input = tf.random.uniform([1, 60, 17, 3], -1.0, 1.0)
            time_ms = measure_time(model, input)

            if time_ms < min_time_ms:
                min_time_ms = time_ms

        print(f'{model_name}: {min_time_ms}ms')