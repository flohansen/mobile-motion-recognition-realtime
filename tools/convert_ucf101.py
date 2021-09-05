import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import uuid
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--number-output-frames', type=int, default=20)
parser.add_argument('ucf101_dir', type=str)
parser.add_argument('export_dir', type=str)
parser.add_argument('classes', nargs='+')
args = parser.parse_args()

movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']

labels_file = os.path.join(args.export_dir, 'labels.txt')
annotation_file = os.path.join(args.export_dir, 'annotations.txt')
data_dir = os.path.join(args.export_dir, 'data')

os.makedirs(args.export_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

def append_line(file_path, line):
    with open(file_path, 'a+') as f:
        f.seek(0)
        data = f.read(100)

        if len(data) > 0:
            f.write('\n')

        f.write(line)

for ucf_class in args.classes:
    videos = glob.glob(os.path.join(args.ucf101_dir, f'{ucf_class}/*.avi'))

    append_line(labels_file, f'{ucf_class}')

    for filename in videos:
        cap = cv2.VideoCapture(filename)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_chunks = num_frames // args.number_output_frames

        motion_map_buffer = np.zeros((args.number_output_frames, 17, 3))

        for i in range(num_chunks * args.number_output_frames):
            ret, frame = cap.read()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            x_frame = tf.convert_to_tensor(frame, dtype=tf.float32)
            x_frame = tf.expand_dims(x_frame, 0)
            x_frame = tf.cast(tf.image.resize(x_frame, (192, 192)), dtype=tf.int32)

            outputs = movenet(x_frame)
            keypoints = outputs['output_0'][0][0]

            for idx, kp in enumerate(keypoints):
                motion_map_buffer[i % args.number_output_frames][idx][0] = kp[0]
                motion_map_buffer[i % args.number_output_frames][idx][1] = kp[1]
                motion_map_buffer[i % args.number_output_frames][idx][2] = kp[2]

            if i > 0 and i % args.number_output_frames == 0:
                motion_map_buffer = (motion_map_buffer * 255.0).astype(np.uint8)
                im = Image.fromarray(motion_map_buffer)

                image_id = uuid.uuid4().hex
                export_file = os.path.join('data', f'{image_id}.png')

                label_idx = args.classes.index(ucf_class)
                append_line(annotation_file, f'{export_file} {label_idx}')

                im.save(os.path.join(data_dir, f'{image_id}.png'))

                motion_map_buffer = np.zeros((args.number_output_frames, 17, 3))

        cap.release()