import os
import cv2
import glob
import numpy as np
from numpy.lib.function_base import append
import tensorflow as tf
import tensorflow_hub as hub
import uuid
from PIL import Image

export_filename = 'push_ups'
video_chunk_size = 60
export_dir = 'datasets'
movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']
current_chunk = 1

labels_file = os.path.join(export_dir, 'labels.txt')
annotation_file = os.path.join(export_dir, 'annotations.txt')
data_dir = os.path.join(export_dir, 'data')

os.makedirs(export_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

classes = {
    'PushUps': 'push_ups',
    'BenchPress': 'bench_press',
    'PullUps': 'pull_ups',
}

def append_line(file_path, line):
    with open(file_path, 'a+') as f:
        f.seek(0)
        data = f.read(100)

        if len(data) > 0:
            f.write('\n')

        f.write(line)

for ucf_class in classes:
    videos = glob.glob(f'C:/Users/flhan/Downloads/UCF-101/{ucf_class}/*.avi')

    append_line(labels_file, f'{classes[ucf_class]}')

    for filename in videos:
        cap = cv2.VideoCapture(filename)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_chunks = num_frames // video_chunk_size

        motion_map_buffer = np.zeros((video_chunk_size, 17, 3))

        for i in range(num_chunks * video_chunk_size):
            ret, frame = cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            x_frame = tf.convert_to_tensor(frame, dtype=tf.float32)
            x_frame = tf.expand_dims(x_frame, 0)
            x_frame = tf.cast(tf.image.resize(x_frame, (192, 192)), dtype=tf.int32)

            outputs = movenet(x_frame)
            keypoints = outputs['output_0'][0][0]

            for idx, kp in enumerate(keypoints):
                motion_map_buffer[i % video_chunk_size][idx][0] = kp[0]
                motion_map_buffer[i % video_chunk_size][idx][1] = kp[1]
                motion_map_buffer[i % video_chunk_size][idx][2] = kp[2]

            if i > 0 and i % video_chunk_size == 0:
                motion_map_buffer = (motion_map_buffer * 255.0).astype(np.uint8)
                im = Image.fromarray(motion_map_buffer)

                image_id = uuid.uuid4().hex
                export_file = os.path.join(data_dir, f'{image_id}.png')

                keys = list(classes.keys())
                label_idx = keys.index(ucf_class)
                append_line(annotation_file, f'{export_file} {label_idx}')

                im.save(export_file)

                motion_map_buffer = np.zeros((video_chunk_size, 17, 3))
                current_chunk += 1

        cap.release()