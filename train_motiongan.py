import os
import datetime
import argparse
import tensorflow as tf
import models.motion_gan as mgan
from dataset import read_motion2021_dataset

@tf.function
def train_critic(critic, critic_optimizer, generator, real_samples, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    with tf.GradientTape() as tape:
        fake_samples = generator((noise, labels))
        fake_scores = critic((fake_samples, labels), training=True)
        real_scores = critic((real_samples, labels), training=True)

        gp = mgan.gradient_penality(critic, fake_samples, real_samples, labels)
        loss = mgan.critic_loss(fake_scores, real_scores, gp, 10)

    gradient = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradient, critic.trainable_variables))
    return loss

@tf.function
def train_generator(generator, generator_optimizer, critic, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    with tf.GradientTape() as tape:
        fake_samples = generator((noise, labels), training=True)
        fake_scores = critic((fake_samples, labels), training=True)
        loss = mgan.generator_loss(fake_scores)

    gradient = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradient, generator.trainable_variables))
    return loss

def summarize_epoch(epoch, generator, generator_loss, critic_loss, latent_vector):
    generated_motion = generator(latent_vector)
    generated_motion = (generated_motion + 1.0) * 0.5

    with summary_writer.as_default():
        tf.summary.scalar('generator_loss', generator_loss, step=epoch)
        tf.summary.scalar('critic_loss', critic_loss, step=epoch)
        tf.summary.image('generated_motion_image', generated_motion, step=epoch)

if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-critic', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    train_x, train_y = read_motion2021_dataset('datasets/motions2021')
    num_classes = train_y.shape[1]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size).shuffle(128)

    generator = mgan.Generator60(num_classes)
    generator_optimizer = tf.keras.optimizers.RMSprop(1e-4)
    generator_save_model_path = os.path.join(args.checkpoint_dir, current_time, 'generator')

    critic = mgan.Critic(num_classes)
    critic_optimizer = tf.keras.optimizers.RMSprop(1e-4)
    critic_save_model_path = os.path.join(args.checkpoint_dir, current_time, 'critic')

    train_log_dir = os.path.join(args.log_dir, current_time, 'train')
    summary_writer = tf.summary.create_file_writer(train_log_dir)
    log_latent_vector = (tf.random.uniform((1, 100), -1.0, 1.0), tf.constant([[1.0, 0.0, 0.0]]))

    for epoch in range(args.epochs):
        for x_batch, labels in train_dataset:
            # Train the critic more than the generator
            for _ in range(args.n_critic):
                c_loss = train_critic(critic, critic_optimizer, generator, x_batch, labels)

            # Train the generator
            g_loss = train_generator(generator, generator_optimizer, critic, labels)

        # Print the losses of the current epoch
        print(f'Epoch {epoch+1}/{args.epochs}, c_loss: {c_loss}, g_loss: {g_loss}')
        summarize_epoch(epoch+1, generator, g_loss, c_loss, log_latent_vector)

        # Save the models if necessary
        if epoch > 0 and (epoch + 1) % args.save_interval == 0:
            generator.save(generator_save_model_path)
            critic.save(critic_save_model_path)