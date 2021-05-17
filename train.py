import os
import datetime
from argparse import ArgumentParser
from models.wgan import WGAN
from dataset import load_video_data
from utils import video_summary
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

@tf.function
def train_step_generator(gan: WGAN, batch_size):
  noise = tf.random.uniform([batch_size, gan.latent_dim], -1.0, 1.0)

  with tf.GradientTape() as tape:
    fake_samples = gan.generator(noise, training=True)
    fake_score = gan.critic(fake_samples, training=True)
    loss = gan.generator_loss(fake_score)

  gradient = tape.gradient(loss, gan.generator.trainable_variables)
  gan.generator.optimizer.apply_gradients(zip(gradient, gan.generator.trainable_variables))
  return loss

@tf.function
def train_step_critic(gan: WGAN, real_samples):
  noise = tf.random.uniform([real_samples.shape[0], gan.latent_dim], -1.0, 1.0)

  with tf.GradientTape() as tape:
    fake_sample = gan.generator(noise)
    real_sample = real_samples

    fake_score = gan.critic(fake_sample)
    real_score = gan.critic(real_sample)

    gp = gan.gradient_penality(fake_sample, real_sample)
    loss = gan.critic_loss(fake_score, real_score, gp)

  gradient = tape.gradient(loss, gan.critic.trainable_variables)
  gan.critic.optimizer.apply_gradients(zip(gradient, gan.critic.trainable_variables))
  return loss

def train(gan, dataset, epochs=100, save_interval=100, batch_size=32, n_critic=5, summary_writer=None, checkpoint_path=None):
  generator_loss = tf.keras.metrics.Mean()
  critic_loss = tf.keras.metrics.Mean()
  z = tf.random.uniform([1, gan.latent_dim], -1.0, 1.0)

  for epoch in range(epochs):
    for batch in dataset:
      for _ in range(n_critic):
        d_loss = train_step_critic(gan, batch)
        critic_loss(d_loss)

      g_loss = train_step_generator(gan, batch_size)
      generator_loss(g_loss)

    if checkpoint_path is not None and (epoch + 1) % save_interval == 0:
        gan.save_model(checkpoint_path)

    if summary_writer is not None:
      frames = gan.generator(z)
      frames = (np.array(frames) * 127.5 + 127.5).astype('uint8')

      with summary_writer.as_default():
        tf.summary.scalar('generator loss', generator_loss.result(), step=epoch)
        tf.summary.scalar('critic loss', critic_loss.result(), step=epoch)
        video_summary('frames sample', frames, fps=20, step=epoch)

    generator_loss.reset_states()
    critic_loss.reset_states()

def read_dataset(buffer_size, batch_size):
  base_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_dir = os.path.join(base_dir, 'dataset')
  train_images = load_video_data(dataset_dir, (128, 96), 80, use_full_videos=True)
  train_images = train_images.astype('float32')
  train_images = (train_images - 127.5) / 127.5
  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
  return train_dataset

def generate_frames(gan, z):
  frames = gan.generator(z)

  output_frames = []
  for i in range(frames.shape[0]):
    for j in range(frames.shape[1]):
      frame = np.array(frames[i, j, :, :, :] * 127.5 + 127.5).astype('uint8')
      output_frames.append(frame)

  return np.array(output_frames)

def generate_video_and_save(gan, z, filename):
  frames = gan.generator(z)

  output_frames = []
  for i in range(frames.shape[0]):
    for j in range(frames.shape[1]):
      frame = np.array(frames[i, j, :, :, :] * 127.5 + 127.5).astype('uint8')
      img = Image.fromarray(frame)
      img = img.resize(size=(120, 90))
      output_frames.append(img)

  output_frames[0].save(filename, save_all=True, append_images=output_frames[1:], duration=30, loop=0)

def main(args):
  dataset = read_dataset(1000, args.batch_size)

  current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
  base_dir = os.path.dirname(os.path.realpath(__file__))
  train_log_dir = os.path.join(base_dir, f'{args.log_dir}/{current_time}/train')
  summary_writer = tf.summary.create_file_writer(train_log_dir)

  wgan = WGAN()
  train(wgan, dataset, epochs=args.epochs, batch_size=args.batch_size, n_critic=args.n_critic, summary_writer=summary_writer)

if __name__ == "__main__":
  # Define arguments of the script
  parser = ArgumentParser()
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--save-interval', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--generator-learning-rate', type=float, default=1e-4)
  parser.add_argument('--discriminator-learning-rate', type=float, default=2e-4)
  parser.add_argument('--n-critic', type=int, default=5)
  parser.add_argument('--dataset-dir', type=str, default='dataset')
  parser.add_argument('--log-dir', type=str, default='logs')

  # Parse arguments from command line
  args = parser.parse_args()
  # Call the main function with arguments from command line
  main(args)
