import cv2
import numpy as np
import tensorflow as tf
from models.wgan_motion import WGAN_Motion
import matplotlib.pyplot as plt

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

z = tf.random.uniform([1, 100], -1.0, 1.0)
generator = tf.keras.models.load_model('experiments/2021-08-09-151550/generator', { 'generator_loss': WGAN_Motion.generator_loss })
generated_motions = generator(z)

fig, ax = plt.subplots(1, 11, sharex=True, sharey=True)

motion_img = (np.array(generated_motions[0]) + 1.0) / 2.0
ax[0].axis('off')
ax[0].imshow(cv2.resize(motion_img, (96, 360)))

for i, keypoints in enumerate(motion_img):
    kps = np.array(keypoints, dtype=np.float32)
    img = np.zeros((360, 360), np.float32)

    for (p1, p2) in connections:
        p1x = int(kps[p1][1] * img.shape[1])
        p1y = int(kps[p1][0] * img.shape[0])
        p2x = int(kps[p2][1] * img.shape[1])
        p2y = int(kps[p2][0] * img.shape[0])
        print(p1x)
        img = cv2.line(img, (p1x, p1y), (p2x, p2y), (150, 0, 0), 3)

    for kp in kps:
        cx = int(kp[1] * img.shape[1])
        cy = int(kp[0] * img.shape[0])
        img = cv2.circle(img, (cx, cy), 6, (255, 0, 0), -1)

    if i < 10:
        ax[i+1].axis('off')
        ax[i+1].imshow(img, cmap='gray')

plt.savefig('test.png', dpi=600, bbox_inches='tight')