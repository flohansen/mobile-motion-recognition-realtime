import cv2
import numpy as np
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

filenames = ['dataset/push_ups/push_ups_5.png', 'dataset/bench_press/bench_press_52.png', 'dataset/dumbbell_laterals/dumbbell_laterals_3.png']

fig, axes = plt.subplots(len(filenames), 11, sharex=True, sharey=True)
plt.subplots_adjust(top=1, bottom=0, hspace=-0.86, wspace=0.02)

for row, filename in enumerate(filenames):
    motion_img = cv2.imread(filename)
    motion_img = cv2.cvtColor(motion_img, cv2.COLOR_BGR2RGB)
    imgs = np.zeros((motion_img.shape[0], 360, 360), dtype=np.float32)

    for i, keypoints in enumerate(motion_img):
        kps = np.array(keypoints) / 255.0
        img = np.zeros(imgs.shape[1:], np.float32)

        for (p1, p2) in connections:
            p1x = int(kps[p1][1] * img.shape[1])
            p1y = int(kps[p1][0] * img.shape[0])
            p2x = int(kps[p2][1] * img.shape[1])
            p2y = int(kps[p2][0] * img.shape[0])
            img = cv2.line(img, (p1x, p1y), (p2x, p2y), (150, 0, 0), 3)

        for kp in kps:
            cx = int(kp[1] * img.shape[1])
            cy = int(kp[0] * img.shape[0])
            img = cv2.circle(img, (cx, cy), 6, (255, 0, 0), -1)

        imgs[i, :, :] = img
        
    axes[row][0].axis('off')
    axes[row][0].imshow(cv2.resize(motion_img, (96, 360)))

    for i in range(10):
        axes[row][i+1].axis('off')
        axes[row][i+1].imshow(imgs[i, :, :], cmap='gray')

plt.savefig('test.png', dpi=600, bbox_inches='tight')