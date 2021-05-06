import os
import tensorflow as tf

class WGAN():
    def __init__(self, generator=None, critic=None, generator_learning_rate=1e-5, critic_learning_rate=1e-5, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator_learning_rate = generator_learning_rate
        self.critic_learning_rate = critic_learning_rate

        if isinstance(generator, tf.keras.Model):
            self.generator = generator
        else:
            self.generator = self.build_generator()

        if isinstance(critic, tf.keras.Model):
            self.critic = critic
        else:
            self.critic = self.build_critic()

    def load_model(self, path):
        generator_path = os.path.join(path, 'generator')
        critic_path = os.path.join(path, 'critic')
        self.generator = tf.keras.models.load_model(generator_path, custom_objects={'generator_loss': self.generator_loss})
        self.critic = tf.keras.models.load_model(critic_path, custom_objects={'critic_loss': self.critic_loss})

    def save_model(self, path):
        generator_path = os.path.join(path, 'generator')
        critic_path = os.path.join(path, 'critic')
        self.generator.save(generator_path)
        self.critic.save(critic_path)

    def generator_loss(self, y_fake):
        return -tf.reduce_mean(y_fake)

