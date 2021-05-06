import os
from models.dcgan import DCGAN
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
matplotlib.use('Qt5Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str)
args = parser.parse_args()

checkpoint_dir = './checkpoints'

if args.checkpoint:
    checkpoint_name = args.checkpoint
else:
    # Get the list of all checkpoints sorted by name
    checkpoint_names = sorted(os.listdir(checkpoint_dir))
    # Set the checkpoint to the last element, which is the latest checkpoint
    checkpoint_name = checkpoint_names[-1]

checkpoint_path = os.path.join('./checkpoints', checkpoint_name)
print(f'Using checkpoint "{checkpoint_path}"')

model = DCGAN()
model.load_model(checkpoint_path)

noise_dim = 100
noise = tf.random.normal([10, noise_dim])
generated_image = model.generator(noise, training=False)

frames = []
fig = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.axis('off')

for i in range(generated_image.shape[0]):
    for j in range(generated_image.shape[1]):
        frames.append([plt.imshow(np.array(generated_image[i, j, :, :, :] * 127.5 + 127.5).astype('uint8'), animated=True)])

evaluation_dir = f'./evaluation/{checkpoint_name}'
if not os.path.isdir(evaluation_dir):
    os.makedirs(evaluation_dir)

model.export_model_summary(evaluation_dir)

animation_file = os.path.join(evaluation_dir, '10_generated_videos.mp4')
ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True, repeat_delay=0)
ani.save(animation_file, fps=30)
plt.show()
