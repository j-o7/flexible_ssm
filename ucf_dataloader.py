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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate


# print('imported!!!')
# os.environ['DECORD_EOF_RETRY_MAX'] = '40960'
decord.bridge.set_bridge('torch')
# test_remove = ['084k_RL3ApU_000109_000119.mp4', '2xWiEVNUvhE_000064_000074.mp4', '305P2f9_lko_004145_004155.mp4',
#                'B4bn9G6__sY_000086_000096.mp4', 'BvBVQmm2RcM_000082_000092.mp4', 'CxjipYE57Yo_000199_000209.mp4',
#                'IhanWvpHGu8_001243_001253.mp4', 'Lw14NH9kAqE_000759_000769.mp4',
#                ' XFkykETgkoo_002967_002977.mp4', 'jJFqy6yiXzQ_000024_000034.mp4',
#                'kinMMqkswUk_000120_000130.mp4', 'y7cYaYX4gdw_000047_000057.mp4']




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


class CustomBatchSampler(DistributedSampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data, num_replicas, rank, batch_size, shuffle):
        super().__init__(data, num_replicas, rank, shuffle)
        self.batch_size = batch_size
        self.data = data
        if shuffle:
            random.shuffle(self.data)


    def __iter__(self):
        batch = []
        num_frames = random.choice([4, 8, 16, 32, 64])
        for index, sample in enumerate(self.data):
            batch.append([index, num_frames])

            if len(batch) == self.batch_size:
                yield batch
                num_frames = random.choice([4, 8, 16, 32, 64])
                batch = []

    def __len__(self):
        return len(self.data)


class UCFDL(Dataset):
    def __init__(self, data_split, num_frames, resolution, viz=False, found=False, stflex=False):
        self.labels = []
        self.num_frames = num_frames
        self.root = "/lustre/fs1/groups/data/UCF101/videos/"
        self.data_split = data_split
        self.viz = viz
        self.resolution = resolution
        self.data_split = data_split
        self.stflex = stflex
        print('resolution: ', resolution)

        if data_split == 'train':
            self.data = open("/lustre/fs1/groups/data/UCF101/trainlist01.txt",'r').read().splitlines()

        else:
            self.data = open("/lustre/fs1/groups/data/UCF101/testlist01.txt",'r').read().splitlines()
        # self.data = self.data[:10]
        print(len(self.data))
        self.labels = [folder for folder in os.listdir(self.root)]
        print(len(self.labels))
        

        # self.data = self.data[:100]
        
        print(len(self.data), self.data[0], len(self.labels), self.labels[0])
        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([resolution, resolution]),
        ])

        if found:
            print(self.data[0])
            self.data = ["Typing/v_Typing_g11_c05.avi 1", "Fencing/v_Fencing_g10_c03.avi 1", "PlayingFlute/v_PlayingFlute_g13_c04.avi 1"]
            print(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_split == 'train' and self.stflex:
            index, num_frames, reso = index
            # print('reso, frames: ', reso, num_frames)
            self.transforms = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize([reso, reso]),
                # transforms.CenterCrop(size=(224, 224))
            ])
        else:
            num_frames = self.num_frames
        v_path = self.data[index].split(' ')[0]
        label = v_path.split('/')[0]

        vid_path = os.path.join(self.root, v_path)
        # print(vid_path)
        try:
            vr = VideoReader(vid_path)
        except Exception:
            return None, None
        frame_indexer = np.linspace(0, len(vr) - 1, num_frames)
        frames = vr.get_batch(frame_indexer)
        frames = frames.permute(0, 3, 1, 2) / 255.
        if self.viz:
            print(v_path)
            frames = transforms.Resize([self.resolution, self.resolution])(frames)

            label = self.labels.index(label)
            return frames, label, v_path
        else:
            frames = self.transforms(frames).permute(1, 0, 2, 3)
            # torch.save(frames, 'sample_vid-96.pt')
            # exit()
            label = self.labels.index(label)

            return frames, label


if __name__ == '__main__':
    shuffle = False
    dataloader_gen = UCFDL('test', 8, 224)
    # cb_sampler = CustomBatchSampler(dataloader_gen.data, batch_size=8, num_replicas=1, rank=0, shuffle=True)
    dataloader = DataLoader(dataloader_gen, num_workers=0, batch_size=8, collate_fn=multiple_samples_collate)
    for frames, label in tqdm(dataloader):
        print(frames.shape)