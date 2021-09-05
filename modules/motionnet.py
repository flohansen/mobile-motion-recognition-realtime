import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization

class MotionNet(tf.keras.Model):
  def __init__(self, classes, number_frames, blocks=[16, 32, 64]):
    super().__init__()
    self.flatten = Flatten()
    self.fc = Dense(classes, activation=tf.keras.activations.softmax)

    self.blocks = tf.keras.Sequential()

    for i, block_channels in enumerate(blocks):
      if i == 0:
        conv = Conv2D(block_channels, kernel_size=3, strides=2, padding='same', use_bias=False, input_shape=(number_frames, 17, 3,))
      else:
        conv = Conv2D(block_channels, kernel_size=3, strides=2, padding='same', use_bias=False)

      self.blocks.add(conv)
      self.blocks.add(BatchNormalization())
      self.blocks.add(tf.keras.layers.ReLU())


  def call(self, x, training=False):
    x = self.blocks(x, training=training)
    x = self.flatten(x)
    x = self.fc(x)
    return x