import argparse
import numpy as np
import random
import tensorflow as tf
import tensorflow_hub as hub
from dataset import read_motion2021_dataset
from modules.motionnet import MotionNet60

def generate_train_samples(num_classes, batch_size, generator):
  train_y = np.zeros((batch_size, num_classes), dtype=np.float32)

  for i in range(batch_size):
    random_label_index = random.randrange(num_classes)
    train_y[i][random_label_index] = 1.0

  random_latent_vectors = tf.random.uniform((batch_size, 100), -1.0, 1.0)
  train_x = generator((random_latent_vectors, train_y))

  return train_x, train_y

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('num_frames', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=50)
  parser.add_argument('--save-interval', type=int, default=100)
  parser.add_argument('--use-dataset-generator', action='store_true')
  args = parser.parse_args()

  with open(f'datasets/motions2021_{args.num_frames}/labels.txt', 'r') as f:
    labels = [l.rstrip("\n") for l in f.readlines()]

  pose_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
  pose_estimator = pose_model.signatures['serving_default']
  generator = tf.keras.models.load_model(f'datasets/kpgan_{args.num_frames}')

  num_classes = len(labels)
  # model = MotionNet(num_classes, [256])
  model = MotionNet60(num_classes, [16, 32, 64, 128])
  real_x, real_y = read_motion2021_dataset(f'datasets/motions2021_{args.num_frames}')
  real_dataset = tf.data.Dataset.from_tensor_slices((real_x, real_y)).batch(args.batch_size)

  num_batches = int(tf.data.experimental.cardinality(real_dataset).numpy())
  classifier_loss = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

  checkpoint_path = 'checkpoints/motionnet'

  model.compile(optimizer=optimizer, loss=classifier_loss, metrics=['accuracy'])

  if args.use_dataset_generator:
    for epoch in range(args.epochs):
      for _ in range(3):
        train_x, train_y = generate_train_samples(num_classes, args.batch_size, generator)
        loss, train_accuracy = model.train_on_batch(real_x, real_y)

      _, test_accuracy = model.test_on_batch(real_x, real_y)
      print(f'Epoch {epoch+1}/{args.epochs}, train_acc: {train_accuracy}, test_acc: {test_accuracy}')

      if epoch > 0 and (epoch + 1) % args.save_interval == 0:
        model.save(checkpoint_path)
        
  else:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      verbose=True,
      save_freq=args.save_interval*num_batches
    )
    model.fit(
      real_dataset,
      epochs=args.epochs,
      shuffle=True,
      callbacks=[cp_callback]
    )
