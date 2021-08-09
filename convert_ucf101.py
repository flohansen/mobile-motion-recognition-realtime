import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

export_filename = 'push_ups'
export_frames = 60
export_dir = 'dataset/push_ups'
movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']
current_chunk = 1

videos = glob.glob('C:/Users/flhan/Downloads/UCF-101/PushUps/*.avi')

for filename in videos:
    cap = cv2.VideoCapture(filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_chunks = num_frames // export_frames

    motion_map = np.zeros((export_frames, 17, 3))

    for i in range(num_chunks * export_frames):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x_frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        x_frame = tf.expand_dims(x_frame, 0)
        x_frame = tf.cast(tf.image.resize(x_frame, (192, 192)), dtype=tf.int32)

        outputs = movenet(x_frame)
        keypoints = outputs['output_0'][0][0]

        for idx, kp in enumerate(keypoints):
            print(i%export_frames, idx)
            motion_map[i % export_frames][idx][0] = kp[0]
            motion_map[i % export_frames][idx][1] = kp[1]
            motion_map[i % export_frames][idx][2] = kp[2]

        if i > 0 and i % export_frames == 0:
            motion_map = (motion_map * 255.0).astype(np.uint8)
            im = Image.fromarray(motion_map)
            export_file = os.path.join(export_dir, f'{export_filename}_{current_chunk}.png')
            im.save(export_file)

            motion_map = np.zeros((export_frames, 17, 3))
            current_chunk += 1

    # print(f'Video {idx+1}/{len(videos)}')

    cap.release()