import os
import argparse
import datetime
import progressbar
import tensorflow as tf
from models.dcgan import DCGAN
from dataset import load_video_data

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-i', '--save-interval', type=int, default=100)
parser.add_argument('-b', '--batch-size', type=int, default=25)
parser.add_argument('-g', '--generator-learning-rate', type=float, default=1e-4)
parser.add_argument('-d', '--discriminator-learning-rate', type=float, default=1e-4)
parser.add_argument('--dataset-dir', type=str, default='dataset')
args = parser.parse_args()

tf.get_logger().setLevel('INFO')
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Hyperparameters
noise_dim = 100

base_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(base_dir, args.dataset_dir)
print(f'Using project root {base_dir}')

# Read in training data and normalize its values to be in [0; 1]
train_images = load_video_data(dataset_dir, (480, 360), 40, use_full_videos=False)
train_images = train_images / 127.5 - 1

# Create batches
BUFFER_SIZE = train_images.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(args.batch_size)
del train_images

current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
checkpoint_dir = os.path.join(base_dir, 'checkpoints')

if args.checkpoint:
    train_log_dir = os.path.join(base_dir, f'logs/{args.checkpoint}/train')
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    model = DCGAN().load_model(checkpoint_path)
else:
    train_log_dir = os.path.join(base_dir, f'logs/{current_time}/train')
    checkpoint_path = os.path.join(checkpoint_dir, current_time)
    model = DCGAN(latent_dim=noise_dim, generator_learning_rate=args.generator_learning_rate, discriminator_learning_rate=args.discriminator_learning_rate)

train_gen_loss = tf.keras.metrics.Mean('train_gen_loss', dtype=tf.float32)
train_dis_loss = tf.keras.metrics.Mean('train_dis_loss', dtype=tf.float32)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

model.generator.summary()
model.discriminator.summary()

@tf.function
def train_step(images):
  noise = tf.random.normal([args.batch_size, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
    generated_images = model.generator(noise, training=True)
    real_output = model.discriminator(images, training=True)
    fake_output = model.discriminator(generated_images, training=True)

    gen_loss = model.generator_loss(fake_output)
    dis_loss = model.discriminator_loss(real_output, fake_output)
    train_gen_loss(gen_loss)
    train_dis_loss(dis_loss)

  gradients_of_gen = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
  gradients_of_dis = dis_tape.gradient(dis_loss, model.discriminator.trainable_variables)

  model.generator.optimizer.apply_gradients(zip(gradients_of_gen, model.generator.trainable_variables))
  model.discriminator.optimizer.apply_gradients(zip(gradients_of_dis, model.discriminator.trainable_variables))

def train(dataset, epochs):
    bar = progressbar.ProgressBar(maxval=args.epochs)
    bar.start()

    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        with train_summary_writer.as_default():
            tf.summary.scalar('generator loss', train_gen_loss.result(), step=epoch)
            tf.summary.scalar('discriminator loss', train_dis_loss.result(), step=epoch)

        if (epoch + 1) % 10 == 0:
            dataset = dataset.shuffle(BUFFER_SIZE)

        if (epoch + 1) % args.save_interval == 0:
            model.save_model(checkpoint_path)

        bar.update(epoch + 1)

    bar.finish()

train(train_dataset, args.epochs)
