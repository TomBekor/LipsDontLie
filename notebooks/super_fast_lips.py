from Video import Video
import matplotlib.pyplot as plt
import numpy as np
import datetime

def show_video_frames(frames, figsize=(15,12), save=None):
    plt.figure(figsize=figsize)
    all_frames = np.concatenate(frames[0:15], axis=1)
    for i in range(15,75,15):
        frames_row = np.concatenate(frames[i:i+15], axis=1)
        all_frames = np.concatenate([all_frames, frames_row], axis=0)
    plt.imshow(all_frames, cmap='gray')
    if save:
        plt.savefig(save)
    else:
        plt.show()

t1 = datetime.datetime.now()
video_path = 'videos/2/bbaf1n.mpg'
video = Video(video_path)
video.super_fast_find_mouth_frames()
t2 = datetime.datetime.now()
print(t2-t1)
print(video.mouth_frames.shape)

show_video_frames(video.mouth_frames, save='super-fast-lips.png')