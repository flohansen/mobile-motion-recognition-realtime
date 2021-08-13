import os
import cv2
import datetime
import glob
from argparse import ArgumentParser
from models.wgan import WGAN
from models.wgan_motion import WGAN_Motion
from dataset import load_video_data
from utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import progressbar
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
def train_step_critic(gan: WGAN, real_sample):
  noise = tf.random.uniform([real_sample.shape[0], gan.latent_dim], -1.0, 1.0)

  with tf.GradientTape() as tape:
    fake_sample = gan.generator(noise)

    fake_score = gan.critic(fake_sample)
    real_score = gan.critic(real_sample)

    gp = gan.gradient_penality(fake_sample, real_sample)
    loss = gan.critic_loss(fake_score, real_score, gp)

  gradient = tape.gradient(loss, gan.critic.trainable_variables)
  gan.critic.optimizer.apply_gradients(zip(gradient, gan.critic.trainable_variables))
  return loss

def train(gan, dataset, start_epoch=0, epochs=100, save_interval=100, batch_size=32, n_critic=5, summary_writer=None, checkpoint_path=None):
  generator_loss = tf.keras.metrics.Mean()
  critic_loss = tf.keras.metrics.Mean()
  z = tf.random.uniform([1, gan.latent_dim], -1.0, 1.0)

  bar = progressbar.ProgressBar(maxval=epochs)
  bar.start()

  for epoch in range(start_epoch, epochs):
    for batch in dataset:
      for _ in range(n_critic):
        d_loss = train_step_critic(gan, batch)
        critic_loss(d_loss)

      g_loss = train_step_generator(gan, batch_size)
      generator_loss(g_loss)

    if checkpoint_path is not None and (epoch + 1) % save_interval == 0:
        gan.save_model(checkpoint_path)

    if summary_writer is not None:
      generated_motion = gan.generator(z)
      generated_motion = ((generated_motion + 1.0) * 127.5) / 255.0

      with summary_writer.as_default():
        tf.summary.scalar('generator loss', generator_loss.result(), step=epoch)
        tf.summary.scalar('critic loss', critic_loss.result(), step=epoch)
        tf.summary.image('motion', generated_motion, step=epoch)
        # video_summary('frames sample', frames, fps=20, step=epoch)

    generator_loss.reset_states()
    critic_loss.reset_states()
    print(f'Epoch {epoch+1} - critic_loss: {critic_loss.result()}, generator_loss: {generator_loss.result()}')
    bar.update(epoch + 1)

  bar.finish()

def read_dataset_videos(path, buffer_size, batch_size):
  train_images = load_video_data(path, (128, 96), 80, use_full_videos=True)
  train_images = train_images.astype('float32')
  train_images = (train_images - 127.5) / 127.5
  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
  return train_dataset

def read_dataset_keypoints(path, buffer_size, batch_size):
  motion_images = glob.glob(os.path.join(path, '*.png'))

  train_images = (np.array([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in motion_images], dtype=np.float32) -127.5) / 127.5
  # train_images = np.array([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in motion_images], dtype=np.float32) / 255.0
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
  current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
  base_dir = os.path.dirname(os.path.realpath(__file__))
  train_log_dir = os.path.join(base_dir, f'{args.log_dir}/{current_time}/train')
  dataset_dir = os.path.join(base_dir, args.dataset_dir)
  checkpoint_path = os.path.join(base_dir, f'{args.checkpoint_dir}/{current_time}')
  start_epoch = 0

  if args.checkpoint is not None:
    checkpoint_path = os.path.join(base_dir, f'{args.checkpoint_dir}/{args.checkpoint}')
    train_log_dir = os.path.join(base_dir, f'{args.log_dir}/{args.checkpoint}/train')
    start_epoch = get_training_epochs(train_log_dir) - 1
    wgan = WGAN(path=checkpoint_path)
    print(f'Loaded model from {checkpoint_path}')
  else:
    wgan = WGAN_Motion(latent_dim=100)

  if isinstance(wgan, WGAN):
    dataset = read_dataset_videos(dataset_dir, 1000, args.batch_size)
  else:
    dataset = read_dataset_keypoints('dataset/push_ups', 1000, 32)

  summary_writer = tf.summary.create_file_writer(train_log_dir)

  train(
    wgan,
    dataset,
    start_epoch=start_epoch,
    epochs=args.epochs,
    batch_size=args.batch_size,
    n_critic=args.n_critic,
    summary_writer=summary_writer,
    checkpoint_path=checkpoint_path,
    save_interval=args.save_interval
  )

if __name__ == "__main__":
  # Define arguments of the script
  parser = ArgumentParser()
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--save-interval', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--n-critic', type=int, default=5)
  parser.add_argument('--log-dir', type=str, default='logs')
  parser.add_argument('--dataset-dir', type=str, default='dataset')
  parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
  parser.add_argument('--checkpoint', type=str, default=None)

  # Parse arguments from command line
  args = parser.parse_args()
  # Call the main function with arguments from command line
  main(args)
