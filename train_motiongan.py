import argparse
import tensorflow as tf
import models.motion_gan as mgan

def train_critic(critic):
    pass

def train_generator(generator):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    generator = mgan.Generator(3, 10)
    critic = mgan.Critic(3, 10)

    for epoch in range(args.epochs):
        for _ in range(args.n_critic):
            train_critic(critic)

        train_generator(generator)