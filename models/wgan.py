import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

class WGAN():
  def __init__(self, path=None, latent_dim=100, gp_lambda=10):
    if path is not None:
      generator_path = os.path.join(path, 'generator')
      critic_path = os.path.join(path, 'critic')

      self.generator = tf.keras.models.load_model(generator_path, custom_objects={'generator_loss': self.generator_loss})
      self.critic = tf.keras.models.load_model(critic_path, custom_objects={'critic_loss': self.critic_loss})
      self.latent_dim = self.generator.input_shape[1]
    else:
      self.latent_dim = latent_dim
      self.generator = self.build_generator(latent_dim)
      self.critic = self.build_critic()

    self.gp_lambda = gp_lambda

  def build_generator(self, latent_dim=100):
    model = tf.keras.Sequential()
    model.add(Dense(5*6*8*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(LeakyReLU())

    model.add(Reshape((5, 6, 8, 256)))
    assert model.output_shape == (None, 5, 6, 8, 256)

    model.add(Conv3DTranspose(128, (5, 5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 12, 16, 128)
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(64, (5, 5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 24, 32, 64)
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(64, (5, 5, 5), strides=2, padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 48, 64, 64)
    model.add(LeakyReLU())

    model.add(Conv3DTranspose(3, (5, 5, 5), strides=2, padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 80, 96, 128, 3)

    opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
    model.compile(optimizer=opt, loss=self.generator_loss)
    return model

  def build_critic(self):
    model = tf.keras.Sequential()
    model.add(Conv2D(32, (5, 5), strides=2, padding='same', input_shape=[80, 96, 128, 3]))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (5, 5), strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    opt = tf.keras.optimizers.Adam(1e-4, beta_1=0, beta_2=0.9)
    model.compile(optimizer=opt, loss=self.critic_loss)
    return model

  def save_model(self, path):
    generator_path = os.path.join(path, 'generator')
    critic_path = os.path.join(path, 'critic')
    self.generator.save(generator_path)
    self.critic.save(critic_path)

  def gradient_penality(self, fake_sample, real_sample):
    batch_size = fake_sample.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0)
    inter_sample = real_sample * alpha + fake_sample * (1.0 - alpha)

    with tf.GradientTape() as tape_gp:
        tape_gp.watch(inter_sample)
        inter_score = self.critic(inter_sample)

    gp_gradients = tape_gp.gradient(inter_score, inter_sample)
    gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

  def generator_loss(self, fake_score):
    return -tf.reduce_mean(fake_score)

  def critic_loss(self, fake_score, real_score, gp):
    return tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp * self.gp_lambda
