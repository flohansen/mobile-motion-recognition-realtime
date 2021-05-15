import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
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
            self.latent_dim = self.generator.input_shape[0]
        else:
            self.latent_dim = latent_dim

            # Create the generator model
            self.generator = self.build_generator(latent_dim)

            # Create the critic model
            self.critic = self.build_critic()

        self.gp_lambda = gp_lambda

    def build_generator(self, latent_dim=100):
        model = tf.keras.Sequential()
        model.add(Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))

        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)

        model.add(Conv2DTranspose(128, (5, 5), strides=1, padding='same', use_bias=False))
        model.add(LeakyReLU())
        assert model.output_shape == (None, 7, 7, 128)

        model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False))
        model.add(LeakyReLU())
        assert model.output_shape == (None, 14, 14, 64)

        model.add(Conv2DTranspose(1, (5, 5), strides=2, padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        opt = tf.keras.optimizers.Adam(1e-4, beta_1=0, beta_2=0.9)
        model.compile(optimizer=opt, loss=self.generator_loss)
        return model

    def build_critic(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (5, 5), strides=2, padding='same', input_shape=[28, 28, 1]))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=2, padding='same'))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))

        opt = tf.keras.optimizers.Adam(1e-4, beta_1=0, beta_2=0.9)
        model.compile(optimizer=opt, loss=self.critic_loss)
        return model

    def gradient_penality(self, fake_sample, real_sample):
        batch_size = fake_sample.shape[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        inter_sample = fake_sample * alpha + real_sample * (1 - alpha)

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
