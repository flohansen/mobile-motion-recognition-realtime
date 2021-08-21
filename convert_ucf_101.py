import cv2
import tensorflow as tf
import tensorflow_hub as hub

movenet_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = movenet_model.signatures['serving_default']

def read_video_in_chunks(video_path, chunk_size):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_chunks = num_frames // chunk_size
    num_frames_to_read = num_chunks * chunk_size

    for i in range(num_frames_to_read):
        _, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        x_frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        x_frame = tf.expand_dims(x_frame, 0)
        x_frame = tf.cast(tf.image.resize(x_frame, (192, 192)), dtype=tf.int32)

        outputs = movenet(x_frame)
        keypoints = outputs['output_0'][0][0]

if __name__ == '__main__':
