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

class BkfstDL(Dataset):
    def __init__(self, data_split, num_frames, resolution, viz=False, found=False, stflex=False):
        self.data = []
        self.labels = []
        self.num_frames = num_frames
        self.data_split = data_split
        self.root = "/lustre/fs1/groups/data/Breakfast/videos/"
        self.stflex = stflex
        self.resolution = resolution
        self.viz = viz
        print('resolution: ', resolution)
        
        if data_split == 'train':
            self.data = open("/home/VideoMamba/videomamba/video_sm/csv/breakfast_train.csv").readlines()[1:]
            
        else:
            self.data = open(
                "/home/VideoMamba/videomamba/video_sm/csv/breakfast_test.csv").readlines()[1:]
            
        self.data = [x.strip('\n') for x in self.data]
        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([resolution, resolution], antialias=True),
        ])
        print(len(self.data))

        if found:
            print(self.data[0])
            self.data = [#'P18-stereo-P18_milk_ch0,0,l,0',
                         'P17-webcam02-P17_milk,0,l,0',
                         #P44-cam02-P44_milk,0,l,0',
                         'P40-webcam01-P40_coffee,0,l,0',
                         'P16-webcam02-P16_friedegg,0,l,0']


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_split == 'train' and self.stflex:
            index, num_frames, reso = index
            # print('reso, frames: ', reso, num_frames)
            self.transforms = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize([reso, reso], antialias=True),
                # transforms.CenterCrop(size=(224, 224))
            ])
            self.num_frames = num_frames

        vid, _, action, label = self.data[index].split(',')
        split = vid.split('-')
        vid_path = os.path.join(self.root, split[0], split[1], split[2] + '.avi')
        try:
            vr = VideoReader(vid_path)
            frame_indexer = np.linspace(0, len(vr)-1, self.num_frames).astype(int)
            frames = vr.get_batch(frame_indexer)
        except Exception as e:
            print(e, vid_path, flush=True)
            return None, None

        frames = frames.permute(0, 3, 1, 2).float() / 255.
        if self.viz:
            print(vid_path)
            frames = transforms.Resize([self.resolution, self.resolution])(frames)
            print(frames.shape)
            return frames, int(label), vid_path
        else:
            frames = self.transforms(frames).permute(1, 0, 2, 3)
            return frames, int(label)


if __name__ == '__main__':
    shuffle = False
    dataloader_gen = BkfstDL('test', 2, False)
    dataloader = DataLoader(dataloader_gen, num_workers=6, batch_size=4, collate_fn=multiple_samples_collate)
    for frames, label in tqdm(dataloader):
        print(frames.shape, label)