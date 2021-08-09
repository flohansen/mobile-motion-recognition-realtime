import cv2
import argparse
import numpy as np
import tensorflow as tf
from models.wgan_motion import WGAN_Motion

parser = argparse.ArgumentParser()
parser.add_argument('generator_model_path', type=str)
args = parser.parse_args()

z = tf.random.uniform([1, 100], -1.0, 1.0)
model = tf.keras.models.load_model(args.generator_model_path, { 'generator_loss': WGAN_Motion.generator_loss })
generated_motion = model(z)

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

for keypoints in generated_motion[0]:
    kps = (np.array(keypoints) + 1.0) / 2.0
    img = np.zeros((768, 1024), np.float32)

    for kp in kps:
        cx = int(kp[1] * img.shape[1])
        cy = int(kp[0] * img.shape[0])
        img = cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

    for (p1, p2) in connections:
        p1x = int(kps[p1][1] * img.shape[1])
        p1y = int(kps[p1][0] * img.shape[0])
        p2x = int(kps[p2][1] * img.shape[1])
        p2y = int(kps[p2][0] * img.shape[0])
        img = cv2.line(img, (p1x, p1y), (p2x, p2y), (255, 0, 0), 2)

    cv2.imshow('Keypoint', img)

    cv2.waitKey(33)