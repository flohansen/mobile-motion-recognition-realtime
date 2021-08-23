import os
import cv2
import gc
import shutil
import tensorflow as tf
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.keras.optimizers import get
from models.wgan_motion import WGAN_Motion
import pandas as pd
from PIL import Image

connections = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (5, 7),
    (6, 8),
    (6, 12),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    (7, 9),
    (8, 10)
]

def generate_motion_gif(generated_motion, shape):
    print(generated_motion.shape)
    img = np.zeros((generated_motion.shape[0], shape[0], shape[1]), np.float32)

    for i, keypoints in enumerate(generated_motion):
        kps = np.array(keypoints)

        for kp in kps:
            cx = int(kp[1] * shape[1])
            cy = int(kp[0] * shape[0])
            img[i, :, :] = cv2.circle(img[i, :, :], (cx, cy), 5, (255, 0, 0), -1)

        for (p1, p2) in connections:
            p1x = int(kps[p1][1] * shape[1])
            p1y = int(kps[p1][0] * shape[0])
            p2x = int(kps[p2][1] * shape[1])
            p2y = int(kps[p2][0] * shape[0])
            img[i, :, :] = cv2.line(img[i, :, :], (p1x, p1y), (p2x, p2y), (255, 0, 0), 2)

    return img

def get_training_information(path):
    acc = EventAccumulator(path)
    acc.Reload()
    df = pd.DataFrame([(w, s, tf.make_ndarray(t)) for w, s, t in acc.Tensors('generator_loss')], columns=['wall_time', 'step', 'tensor'])

    epochs = int(df.iloc[-1]['step']) + 1
    estimated_time = df.iloc[-1]['wall_time']

    return (epochs, estimated_time)


log_dir = 'logs'
evaluation_dir = 'evaluation'
experiments_dir = './experiments'
experiment_names = sorted(os.listdir(experiments_dir))
shutil.rmtree(evaluation_dir)

template_string  = '| #   | Datensatz | Epochen | Image | Motion |\n'
template_string += '| --- | --------- | ------- | ------------ | ---------- |\n'

z = (tf.random.uniform([1, 100], -1.0, 1.0), tf.constant([[0.0, 0.0, 1.0]]))

for i, experiment_name in enumerate(experiment_names):
    experiment_path = os.path.join(experiments_dir, experiment_name)
    generator_model_path = os.path.join(experiment_path, 'generator')
    generator_log_path = os.path.join(log_dir, experiment_name, 'train')
    experiment_evaluation_dir = os.path.join(evaluation_dir, experiment_name)
    motion_image_path = os.path.join(experiment_evaluation_dir, 'keypoints.png')
    motion_gif_path = os.path.join(experiment_evaluation_dir, 'motion.gif')

    os.makedirs(experiment_evaluation_dir)

    generator = tf.keras.models.load_model(generator_model_path)
    generated_motions = generator(z)

    if generator.layers[-1].activation is tf.keras.activations.tanh:
        generated_motions = (generated_motions + 1.0) / 2.0

    img = np.array(generated_motions[0] * 255.0).astype('uint8')
    img = Image.fromarray(img)
    img.save(motion_image_path)

    motion_gif = generate_motion_gif(generated_motions[0], (200, 300))
    motion_gif = [Image.fromarray(frame) for frame in motion_gif]
    motion_gif[0].save(motion_gif_path, save_all=True, append_images=motion_gif[1:], duration=30, loop=0)

    activation_name = 'tanh' if generator.layers[-1].activation is tf.keras.activations.tanh else 'sigmoid'
    epochs, _ = get_training_information(generator_log_path)
    
    template_string += f'| {i+1} | {experiment_name} <br/> Activation: **{activation_name}** | {epochs} | ![]({motion_image_path.replace(os.sep, "/")}) | ![]({motion_gif_path.replace(os.sep, "/")}) |\n'

    tf.keras.backend.clear_session()
    gc.collect()

# for i, experiment_name in enumerate(experiment_names):
#     print(f'Evaluating {experiment_name}...')
#     checkpoint_path = os.path.join('./checkpoints', experiment_name)
# 
#     log_dir = f'./logs/{experiment_name}/train'
#     evaluation_dir = f'./evaluation/{experiment_name}'
#     animation_file = os.path.join(evaluation_dir, 'results.gif')
#     generate_gif = False
# 
#     if not os.path.isdir(evaluation_dir):
#         os.makedirs(evaluation_dir)
#         generate_gif = True
# 
#     model = DCGAN()
#     model.load_model(checkpoint_path)
#     generator_filename, discriminator_filename = model.export_model_summary(evaluation_dir)
# 
#     frames = model.generator.output_shape[1]
#     width = model.generator.output_shape[3]
#     height = model.generator.output_shape[2]
#     channels = model.generator.output_shape[4]
# 
#     (epochs, estimated_time) = get_training_information(log_dir)
#     template_string += f'| {i+1} | {experiment_name} <br/> [Generator]({generator_filename}) <br/>[Diskriminator]({discriminator_filename}) | {epochs} | Frames: {frames} <br/> Größe: {width}x{height} <br/> Kanäle: {channels} | ![]({animation_file}) |\n'
# 
#     if generate_gif:
#         print(f'Checkpoint {experiment_name} will be evaluated.') 
# 
#         noise_dim = 100
#         noise = tf.random.normal([5, noise_dim])
#         frames = model.generator(noise, training=False)
# 
#         output_frames = []
#         for i in range(frames.shape[0]):
#             for j in range(frames.shape[1]):
#                 frame = np.array(frames[i, j, :, :, :] * 127.5 + 127.5).astype('uint8')
#                 img = Image.fromarray(frame)
#                 img = img.resize(size=(120, 90))
#                 output_frames.append(img)
# 
#         output_frames[0].save(animation_file, save_all=True, append_images=output_frames[1:], duration=30, loop=0)
# 
#     tf.keras.backend.clear_session()
#     gc.collect()
# 

with open('README_template.md', 'r+') as f:
    texts = f.read()
    texts = texts.replace('[EVALUATION_TABLE]', template_string)

with open('README.md', 'w') as f:
    f.write(texts)
