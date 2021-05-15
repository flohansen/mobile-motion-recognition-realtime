import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D

class DCGAN():
    def __init__(self, generator=None, discriminator=None, generator_learning_rate=1e-4, discriminator_learning_rate=1e-4, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        if isinstance(generator, tf.keras.Model):
            self.generator = generator
        else:
            self.generator = self.build_generator()

        if isinstance(discriminator, tf.keras.Model):
            self.discriminator = discriminator
        else:
            self.discriminator = self.build_discriminator()

    def load_model(self, path):
        generator_path = os.path.join(path, 'generator')
        discriminator_path = os.path.join(path, 'discriminator')
        self.generator = tf.keras.models.load_model(generator_path, custom_objects={'generator_loss': self.generator_loss})
        self.discriminator = tf.keras.models.load_model(discriminator_path, custom_objects={'discriminator_loss': self.discriminator_loss})

    def save_model(self, path):
        generator_path = os.path.join(path, 'generator')
        discriminator_path = os.path.join(path, 'discriminator')
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(5*12*16*256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((5, 12, 16, 256)))
        assert model.output_shape == (None, 5, 12, 16, 256)

        model.add(Conv3DTranspose(128, (5, 5, 5), strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 10, 24, 32, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv3DTranspose(64, (5, 5, 5), strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 20, 48, 64, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv3DTranspose(64, (5, 5, 5), strides=2, padding='same', use_bias=False))
        assert model.output_shape == (None, 40, 96, 128, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv3DTranspose(3, (5, 5, 5), strides=2, padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 80, 192, 256, 3)

        opt = tf.keras.optimizers.Adam(self.generator_learning_rate)
        model.compile(loss=self.generator_loss, optimizer=opt)
        return model

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(64, (5, 5), strides=2, padding='same', input_shape=[80, 192, 256, 3]))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (5, 5), strides=2, padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1))

        opt = tf.keras.optimizers.Adam(self.discriminator_learning_rate)
        model.compile(loss=self.discriminator_loss, optimizer=opt)
        return model

    def generator_loss(self, y_fake):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(y_fake), y_fake)

    def discriminator_loss(self, y_real, y_fake):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(y_real), y_real)
        fake_loss = cross_entropy(tf.zeros_like(y_fake), y_fake)
        return real_loss + fake_loss

    def export_model_summary(self, target_dir):
        generator_filename = os.path.join(target_dir, 'generator.txt')
        discriminator_filename = os.path.join(target_dir, 'discriminator.txt')

        with open(generator_filename, 'w') as f:
            self.generator.summary(print_fn=lambda x: f.write(x + '\n'))

        with open(discriminator_filename, 'w') as f:
            self.discriminator.summary(print_fn=lambda x: f.write(x + '\n'))

        return generator_filename, discriminator_filename
