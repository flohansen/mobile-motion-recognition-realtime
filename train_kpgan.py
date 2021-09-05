import os
import datetime
import argparse
import tensorflow as tf
import modules.kpgan as kpgan
from dataset import read_motion2021_dataset

@tf.function
def train_critic(critic, critic_optimizer, generator, real_samples, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    with tf.GradientTape() as tape:
        fake_samples = generator((noise, labels))
        fake_scores = critic((fake_samples, labels), training=True)
        real_scores = critic((real_samples, labels), training=True)

        gp = kpgan.gradient_penality(critic, fake_samples, real_samples, labels)
        loss = kpgan.critic_loss(fake_scores, real_scores, gp, 10)

    gradient = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradient, critic.trainable_variables))
    return loss

@tf.function
def train_generator(generator, generator_optimizer, critic, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    with tf.GradientTape() as tape:
        fake_samples = generator((noise, labels), training=True)
        fake_scores = critic((fake_samples, labels), training=True)
        loss = kpgan.generator_loss(fake_scores)

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
    parser.add_argument('frames', type=int, help='Number of frames of a motion')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size of training data')
    parser.add_argument('--n-critic', type=int, default=5, help='Number of training iteration for critic')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save-interval', type=int, default=10, help='After how many epochs the current model state should be saved')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='The directory where the checkpoints should be stored')
    parser.add_argument('--log-dir', type=str, default='logs', help='The directory of log files. Those can be displayed by Tensorboard')
    parser.add_argument('--dataset-dir', type=str, default='datasets', help='The folder where all datasets are placed')
    args = parser.parse_args()

    if args.frames != 10 and args.frames != 20 and args.frames != 60:
        print(f'Please choose the <frames> parameter to be one of [10, 20, 60]. Given {args.frames}')
        exit()

    motion_dataset_dir = os.path.join(args.dataset_dir, f'motions2021_{args.frames}')

    if not os.path.isdir(motion_dataset_dir):
        print(f"There is no folder named `{motion_dataset_dir}`. Please make sure you've downloaded the dataset.")
        exit()

    train_x, train_y = read_motion2021_dataset(motion_dataset_dir)
    num_classes = train_y.shape[1]
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size).shuffle(128)

    if args.frames == 10:
        generator = kpgan.Generator10(num_classes)
    elif args.frames == 20:
        generator = kpgan.Generator20(num_classes)
    elif args.frames == 60:
        generator = kpgan.Generator60(num_classes)

    generator_optimizer = tf.keras.optimizers.RMSprop(1e-4)
    generator_save_model_path = os.path.join(args.checkpoint_dir, current_time, 'generator')

    critic = kpgan.Critic(num_classes, args.frames)
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