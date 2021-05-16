from argparse import ArgumentParser
from models.wgan import WGAN
import tensorflow as tf

@tf.function
def train_step_generator(gan: WGAN, batch_size):
  noise = tf.random.uniform([batch_size, gan.latent_dim])

  with tf.GradientTape() as tape:
    fake_samples = gan.generator(noise, training=True)
    fake_score = gan.critic(fake_samples, training=True)
    loss = gan.generator_loss(fake_score)

  gradient = tape.gradient(loss, gan.generator.trainable_variables)
  gan.generator.optimizer.apply_gradients(zip(gradient, gan.generator.trainable_variables))
  return loss

@tf.function
def train_step_critic(gan: WGAN, real_samples):
  noise = tf.random.uniform([real_samples.shape[0], gan.latent_dim])

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

def train(gan: WGAN, dataset, epochs, batch_size, n_critic):
  generator_loss = tf.keras.metrics.Mean()
  critic_loss = tf.keras.metrics.Mean()

  for epoch in range(epochs):
    for batch in dataset:
      for _ in range(n_critic):
        d_loss = train_step_critic(gan, batch)
        critic_loss(d_loss)

      g_loss = train_step_generator(gan, batch_size)
      generator_loss(g_loss)

    print(f'epoch: {epoch}, g_loss: {generator_loss.result():6.3f}, d_loss: {critic_loss.result():6.3f}')
    generator_loss.reset_states()
    critic_loss.reset_states()

def read_dataset(buffer_size, batch_size):
  (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
  x_train = (x_train - 127.5) / 127.5
  train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)
  return train_dataset

def main(args):
  wgan = WGAN()
  dataset = read_dataset(60000, args.batch_size)
  train(wgan, dataset, args.epochs, args.batch_size, args.n_critic)

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

  # Parse arguments from command line
  args = parser.parse_args()
  # Call the main function with arguments from command line
  main(args)
