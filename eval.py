import os
import gc
from models.dcgan import DCGAN
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from PIL import Image
matplotlib.use('Qt5Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_training_information(path):
    acc = EventAccumulator(path)
    acc.Reload()
    df = pd.DataFrame([(w, s, tf.make_ndarray(t)) for w, s, t in acc.Tensors('generator loss')], columns=['wall_time', 'step', 'tensor'])

    epochs = int(df.iloc[-1]['step']) + 1
    estimated_time = df.iloc[-1]['wall_time']

    return (epochs, estimated_time)

checkpoint_dir = './checkpoints'
checkpoint_names = sorted(os.listdir(checkpoint_dir))

template_string  = '| #   | Datensatz | Epochen | Output-Shape | Ergebnisse |\n'
template_string += '| --- | --------- | ------- | ------------ | ---------- |\n'

for i, checkpoint_name in enumerate(checkpoint_names):
    print(f'Evaluating {checkpoint_name}...')
    checkpoint_path = os.path.join('./checkpoints', checkpoint_name)

    log_dir = f'./logs/{checkpoint_name}/train'
    evaluation_dir = f'./evaluation/{checkpoint_name}'
    animation_file = os.path.join(evaluation_dir, 'results.gif')

    model = DCGAN()
    model.load_model(checkpoint_path)
    generator_filename, discriminator_filename = model.export_model_summary(evaluation_dir)

    with open(generator_filename, 'r+') as f:
        generator_summary = f.read()
    with open(generator_filename, 'r+') as f:
        discriminator_summary = f.read()

    (epochs, estimated_time) = get_training_information(log_dir)
    template_string += f'| {i+1} | {checkpoint_name} | {epochs} | {model.generator.output_shape} | ![]({animation_file}) |\n'

    if not os.path.isdir(evaluation_dir):
        print(f'Checkpoint {checkpoint_name} will be evaluated.') 
        os.makedirs(evaluation_dir)

        noise_dim = 100
        noise = tf.random.normal([5, noise_dim])
        frames = model.generator(noise, training=False)

        output_frames = []
        for i in range(frames.shape[0]):
            for j in range(frames.shape[1]):
                frame = np.array(frames[i, j, :, :, :] * 127.5 + 127.5).astype('uint8')
                img = Image.fromarray(frame)
                img = img.resize(size=(120, 90))
                output_frames.append(img)

        output_frames[0].save(animation_file, save_all=True, append_images=output_frames[1:], duration=30, loop=0)

    tf.keras.backend.clear_session()
    gc.collect()

with open('README_template.md', 'r+') as f:
    texts = f.read()
    texts = texts.replace('[EVALUATION_TABLE]', template_string)

with open('README.md', 'w') as f:
    f.write(texts)
