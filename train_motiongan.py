import argparse
import tensorflow as tf
import models.motion_gan as mgan

from dataset import read_motion2021_dataset


def train_critic(critic, generator, labels):
    noise = tf.random.uniform([len(labels), 100], -1.0, 1.0)

    fake_sample = generator(noise, labels)

def train_generator(generator):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    train_x, train_y = read_motion2021_dataset('datasets/motions2021')
    num_classes = train_y.shape[1]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(args.batch_size).shuffle(179)

    image_batch, label_batch = next(iter(train_dataset))

    generator = mgan.Generator(num_classes, 10)
    critic = mgan.Critic(num_classes, 10)

    noise = tf.random.uniform([label_batch.shape[0], 100], -1.0, 1.0)
    y = generator(noise, label_batch)
    print(y)

    exit()


    for epoch in range(args.epochs):
        for _ in range(args.n_critic):
            with tf.GradientTape() as tape:
                critic_loss = train_critic(critic, generator, )

        train_generator(generator)