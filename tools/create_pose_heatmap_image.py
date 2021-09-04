import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

def gaussian(img, pt, sigma=4):
    '''
    Source: https://github.com/princeton-vl/pose-hg-train/blob/master/src/pypose/draw.py
    '''
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
      # If not, just return the image as is
      return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']

input_image = cv2.resize(cv2.imread('C:/Users/flhan/Downloads/person.jpg'), (192, 192))
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image_tensor = tf.convert_to_tensor(input_image_rgb, dtype=tf.float32)
input_image_tensor = tf.expand_dims(input_image_tensor, 0)
input_image_tensor = tf.cast(input_image_tensor, dtype=tf.int32)

predictions = movenet(input_image_tensor)
keypoints = predictions['output_0'][0][0]
hmap_img = np.zeros((192, 192))
kp_img = np.copy(input_image)

for i, keypoint in enumerate(keypoints):
    cx = int(keypoint[1] * input_image_rgb.shape[1])
    cy = int(keypoint[0] * input_image_rgb.shape[0])
    mask = gaussian(np.zeros((192, 192)), (cx, cy))
    hmap_img = np.clip(hmap_img + mask, 0, 1)
    kp_img = cv2.circle(kp_img, (cx, cy), 3, (0, 0, 255), -1)

plt.figure()
plt.axis('off')
plt.imshow(input_image_rgb)
plt.imshow(hmap_img, alpha=0.7)

plt.savefig('person_hmap.png', bbox_inches='tight', pad_inches=0)
cv2.imwrite('person.png', input_image)
cv2.imwrite('person_kp.png', kp_img)