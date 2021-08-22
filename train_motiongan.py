import argparse
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import default
import models.motion_gan as mgan

from dataset import read_motion2021_dataset

@tf.function
def train_critic(critic, generator, real_samples, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    with tf.GradientTape() as tape:
        fake_samples = generator(noise, labels)
        fake_scores = critic(fake_samples, labels)
        real_scores = critic(real_samples, labels)

        gp = mgan.gradient_penality(critic, fake_samples, real_samples, labels)
        loss = mgan.critic_loss(fake_scores, real_scores, gp, 10)

    gradient = tape.gradient(loss, critic.trainable_variables)
    critic.optimizer.apply_gradients(zip(gradient, critic.trainable_variables))
    return loss

@tf.function
def train_generator(generator, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    with tf.GradientTape() as tape:
        fake_samples = generator(noise, labels)
        fake_scores = critic(fake_samples, labels)
        loss = mgan.generator_loss(fake_scores)

    gradient = tape.gradient(loss, generator.trainable_variables)
    generator.optimizer.apply_gradients(zip(gradient, generator.trainable_variables))
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    train_x, train_y = read_motion2021_dataset('datasets/motions2021')
    num_classes = train_y.shape[1]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size).shuffle(179)

    generator = mgan.Generator()
    critic = mgan.Critic()

    for epoch in range(args.epochs):
        for x_batch, labels in train_dataset:
            print(x_batch.shape)
            for _ in range(args.n_critic):
                c_loss = train_critic(critic, generator, x_batch, labels)

            g_loss = train_generator(generator, labels)

            print(c_loss, g_loss)