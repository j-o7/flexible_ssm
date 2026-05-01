import numpy as np
import os
import pandas as pd
import torch
import decord
import timeit
import pickle
import json
import random
import cv2
import ffmpeg

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import Variable
from torchvision import transforms
from decord import VideoReader, cpu, gpu
from datetime import timedelta
from torch.utils.data._utils.collate import default_collate

print('imported!!!')
# os.environ['DECORD_EOF_RETRY_MAX'] = '40960'
decord.bridge.set_bridge('torch')


def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    inputs = [x for x in inputs if x is not None]
    labels = [x for x in labels if x is not None]

    inputs, labels, = (
        default_collate(inputs),
        default_collate(labels),
    )

    return inputs, labels


# test_remove = ['084k_RL3ApU_000109_000119.mp4', '2xWiEVNUvhE_000064_000074.mp4', '305P2f9_lko_004145_004155.mp4',
#                'B4bn9G6__sY_000086_000096.mp4', 'BvBVQmm2RcM_000082_000092.mp4', 'CxjipYE57Yo_000199_000209.mp4',
#                'IhanWvpHGu8_001243_001253.mp4', 'Lw14NH9kAqE_000759_000769.mp4',
#                ' XFkykETgkoo_002967_002977.mp4', 'jJFqy6yiXzQ_000024_000034.mp4',
#                'kinMMqkswUk_000120_000130.mp4', 'y7cYaYX4gdw_000047_000057.mp4']

class HMDBDL(Dataset):
    def __init__(self, data_split, num_frames, resolution, viz=False, found=False):
        self.data = []
        self.labels = []
        self.num_frames = num_frames
        self.data_split = data_splita
        self.root = "/lustre/fs1/groups/data/hmdb51/videos/"
        self.viz = viz
        self.resolution = resolution
        print('resolution: ', resolution)


        if data_split == 'train':
            self.data = open("/lustre/fs1/home/VideoMamba/videomamba/video_sm/csv/trainlist01.txt").readlines()

        else:
            self.data = open(
                "/lustre/fs1/home/VideoMamba/videomamba/video_sm/csv/testlist01.txt").readlines()

        self.labels = open("/lustre/fs1/home/VideoMamba/videomamba/video_sm/csv/classInd.txt", 'r').readlines()
        self.labels = [label.split(' ')[1].strip('\n') for label in self.labels]

        self.data = [x.strip('\n') for x in self.data]
        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([resolution, resolution]),
        ])
        print(len(self.data), self.data[0], len(self.labels), self.labels[0])

        if found:
            self.data = [#'punch/THE_PROTECTOR_punch_f_cm_np1_ba_med_49.avi 1',
                         # 'laugh/Best_Of_Skype_Laughter_Chain_laugh_h_nm_np1_fr_bad_4.avi 1',
                         # 'shake_hands/A_Beautiful_Mind_1_shake_hands_u_nm_np2_le_med_1.avi 1',
                         'smile/YouTube_smiles!_smile_h_cm_np1_fr_med_26.avi 1',
                         'ride_horse/Mylifehorseriding_ride_horse_f_nm_np1_fr_med_5.avi 1',
                         # 'ride_bike/1996_Tour_de_France_-_Indurain_Cracks_ride_bike_f_cm_np1_ba_med_1.avi 1',
                         # 'situp/Personal_Training_Workout_Tips_situp_f_nm_np1_le_goo_1.avi 1',
                         'cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_5.avi 1']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_split == 'train':
            vid, label = self.data[index].split(' ')
            label = int(label) - 1
        else:
            vid = self.data[index]
            label = vid.split('/')[0]
            label = self.labels.index(label)


        vid_path = os.path.join(self.root, vid)
        try:
            vr = VideoReader(vid_path)
            frame_indexer = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
            frames = vr.get_batch(frame_indexer)
        except Exception as e:
            print(e, vid_path, flush=True)
            return None, None

        frames = frames.permute(0, 3, 1, 2).float() / 255.

        if self.viz:
            print(vid_path)
            frames = transforms.Resize([self.resolution, self.resolution])(frames)
            return frames, label, vid_path
        else:
            frames = self.transforms(frames).permute(1, 0, 2, 3)
            # print(frames.shape)
            return frames, label


if __name__ == '__main__':
    shuffle = False
    dataloader_gen = HMDBDL('train', 16, False)
    dataloader = DataLoader(dataloader_gen, num_workers=0, batch_size=1, collate_fn=multiple_samples_collate)
    for frames, label in tqdm(dataloader):
        pass