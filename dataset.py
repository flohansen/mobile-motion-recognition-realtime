import os
import numpy as np
import cv2
import operator
import functools

def foldl(func, acc, xs):
    return functools.reduce(func, xs, acc)

def min_frames(video_paths):
    min = 999999

    for v in video_paths:
        cap = cv2.VideoCapture(v)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if min > num_frames:
            min = num_frames

    return min


def read_frames(video_path, shape, number_frames, use_full_video=False):
    frames = []
    cap = cv2.VideoCapture(video_path)
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if use_full_video:
        number_chunks = max_frames // number_frames
    else:
        number_chunks = min(1, max_frames // number_frames)

    for i in range(number_chunks):
        start_frame_index = i * number_frames
        end_frame_index = start_frame_index + number_frames
        current_frames = []

        for j in range(start_frame_index, end_frame_index):
            _, frame = cap.read()
            frame = cv2.resize(frame, shape)
            current_frames.append(frame)

            if j == (end_frame_index - 1):
                frames.append(current_frames)
                current_frames = []

    cap.release()
    return np.array(frames)

def load_video_data(videos_dir, video_shape, frames, use_full_videos=False):
    video_paths = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir)]

    if frames:
        number_frames = frames
    else:
        number_frames = min_frames(video_paths)

    chunks = []
    for video_path in video_paths:
        chunks.extend(read_frames(video_path, video_shape, number_frames, use_full_videos))

    return np.array(chunks)

def load_video_data_images(videos_dir, video_shape):
    video_paths = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir)]

    number_frames = min_frames(video_paths)
    videos = [read_frames(video_path, video_shape, number_frames) for video_path in video_paths]
    videos = foldl(operator.add, [], videos)
    videos = np.array(videos)

    return videos
