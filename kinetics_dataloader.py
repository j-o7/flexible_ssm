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

# import logging
# logging.basicConfig(filename="error.txt",level=logging.DEBUG)
# logging.captureWarnings(True)


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

    def __init__(self, data, num_replicas, rank, batch_size, shuffle, spatial_flex, static_tokens):
        # super().__init__(data, num_replicas, rank, shuffle)
        self.batch_size = batch_size
        self.data = data
        self.spatial_flex = spatial_flex
        self.static_tokens = static_tokens
        if shuffle:
            random.shuffle(self.data)


    def __iter__(self):
        batch = []
        num_frames = random.choice([4, 8, 16])
        if self.spatial_flex:
            reso = random.choice([96, 128, 224, 384])#, 512, 640])
        elif self.static_tokens:  # need resos divisible by 14 to determine patch size that results in 14 tokens
            reso = random.choice([98, 126, 224, 392])
        for index, sample in enumerate(self.data):
            if self.spatial_flex or self.static_tokens:
                batch.append([index, num_frames, reso])
            else:
                batch.append([index, num_frames])


            if len(batch) == self.batch_size:
                yield batch
                num_frames = random.choice([4, 8, 16])
                if self.spatial_flex:
                    reso = random.choice([96, 128, 224, 384])
                elif self.static_tokens:  # need resos divisible by 14 to determine patch size that results in 14 tokens
                    reso = random.choice([98, 126, 224, 392])
                batch = []

    def __len__(self):
        return len(self.data) // self.batch_size
        # return 24 // self.batch_size


class KineticsDL(Dataset):
    def __init__(self, data_split, num_frames, flexible=False, spatial_flex=False, flexivit=False, flex_all=False, static_tokens=False, resolution=224):
        self.labels = []
        self.num_frames = num_frames
        self.flexible = flexible
        self.spatial_flex = spatial_flex
        self.flexivit = flexivit
        self.flex_all = flex_all
        self.static_tokens = static_tokens
        self.resolution = resolution

        assert (self.flexivit and self.spatial_flex) != 1, print('spatial flex and flexivit cannot both be true')


        self.root = '/groups/data/k400_320p/'
        self.data_split = data_split
        # val_videos = os.listdir('/home/c3-0/datasets/kinetics400_dataset/val')
        # test_videos = os.listdir('/home/c3-0/datasets/kinetics400_dataset/test')
        # train_videos = os.listdir('/home/c3-0/datasets/kinetics400_dataset/train')
        # print(len(train_videos), len(val_videos), len(test_videos))
        err = open('/home/VideoMamba/videomamba/video_sm/error.txt', 'r').readlines()[2:]
        remove = []
        for line in err:
            if 'Error' in line:
                vid_id = line.split(' ')[2].split('/')[6].strip('\n')[:11]
                remove.append(vid_id)

        if data_split == 'train':
            self.data = pickle.load(open("/home/VideoMamba/videomamba/video_sm/train.pkl", 'rb'))

        else:
            remove = []
            self.data = open("/lustre/fs1/home/VideoMamba/videomamba/video_sm/csv/val.csv", 'r').readlines()[1:]
            self.data = [x.strip('\n') for x in self.data]
            print(len(self.data))

            err = open('/home/VideoMamba/videomamba/video_sm/kin_val_error.txt', 'r').readlines()
            for i, line in enumerate(err):
                if 'Error' in line:
                    # print(line)
                    index = line.split(' ')[3]
                    # print(index)
                    remove.append(int(index))
            remove.append(3470)
            print('remove: ', remove)
            for index in sorted(remove, reverse=True):
                print(index)
                del self.data[index]
            self.data.remove('"dying hair",I0luMKjIZyg,422,432,val,0')
            self.data.remove('"rock climbing",_6uq-NBo3Bk,12,22,val,0')
            self.data.remove('"skipping rope",sAA809R_u1E,77,87,val,0')

        print(len(self.data))
        # self.data = self.data
        self.labels = open("/lustre/fs1/home/VideoMamba/videomamba/video_sm/csv/clsIdx.csv", 'r').readlines()[1:]
        self.labels = [x.split(',')[0] for x in self.labels]
        # self.data = self.data[:50]
        print(len(self.data), self.data[0], len(self.labels), self.labels[0])
        self.data = [x for x in self.data if x.split(',')[1] not in remove]
        print(len(self.data), self.data[0], len(self.labels), self.labels[0])

        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(256, antialias=True), # why not [256, 256]?
            transforms.CenterCrop(size=(self.resolution, self.resolution))

        ])
        # exit()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.flexible and self.data_split == 'train' and (self.spatial_flex or self.flex_all or self.static_tokens):
            index, num_frames, reso = index
            # print('reso, frames: ', reso, num_frames)
            self.transforms = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize([reso, reso], antialias=True),
                # transforms.CenterCrop(size=(224, 224))
            ])

        elif self.flexible and self.data_split == 'train' and self.flexivit:
            index, num_frames = index
            # print('reso, frames: ', reso, num_frames)
            self.transforms = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Resize([256, 256], antialias=True),
                transforms.CenterCrop(size=(240, 240))
            ])

        elif self.flexible and self.data_split == 'train':
            index, num_frames = index

        else:
            num_frames = self.num_frames
            index = index

        if self.data_split == 'train':
            label, vid, start, stop, _, _ = self.data[index].split(',')
            vid = f'{vid}_{start.zfill(6)}_{stop.zfill(6)}'
            vid_path = os.path.join(self.root + 'train', vid+'.mp4')
            # print(vid_path)
            # exit()
            try:
                vr = VideoReader(vid_path)
            except Exception as e:
                print(e, flush=True)
                # print('entered')
                # print()
                # print()
                label, vid, start, stop, _, _ = self.data[index-random.randrange(5, 10)].split(',')
                vid = f'{vid}_{start.zfill(6)}_{stop.zfill(6)}'
                vid_path = os.path.join(self.root + 'train', vid + '.mp4')
                vr = VideoReader(vid_path)
                # return None, None
            frame_indexer = np.linspace(0, len(vr) - 1, num_frames)
            frames = vr.get_batch(frame_indexer)
            frames = frames.permute(0, 3, 1, 2) / 255.
            # torch.save(frames, 'kin_frame.pt')
            # exit()
            frames = self.transforms(frames).permute(1, 0, 2, 3)
            label = self.labels.index(label.strip('\"'))


        elif self.data_split == 'test':
            label, vid, start, stop, _, _ = self.data[index].split(',')
            vid = f'{vid}_{start.zfill(6)}_{stop.zfill(6)}'
            vid_path = os.path.join(self.root + 'val', vid + '.mp4')
            try:
                vr = VideoReader(vid_path)
            except Exception as e:
                print(e, index, self.data[index])
                # label, vid, start, stop, _, _ = self.data[index-1].split(',')
                # vid = f'{vid}_{start.zfill(6)}_{stop.zfill(6)}'
                # vid_path = os.path.join(self.root + 'val', vid + '.mp4')
                # vr = VideoReader(vid_path)
                return None, None
            frame_indexer = np.linspace(0, len(vr) - 1, num_frames)
            frames = vr.get_batch(frame_indexer)
            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = self.transforms(frames).permute(1, 0, 2, 3)
            label = self.labels.index(label.strip('\"'))

        return frames, label


if __name__ == '__main__':
    shuffle = False
    spatial_flex = False
    flexivit=False
    tr_dataloader_gen = KineticsDL('train', 8, False, spatial_flex, flexivit, False, False)
    cb_sampler = CustomBatchSampler(tr_dataloader_gen.data, batch_size=4, num_replicas=1, rank=0, shuffle=True, spatial_flex=spatial_flex, static_tokens=False)
    trdataloader = DataLoader(tr_dataloader_gen, num_workers=0, batch_sampler=cb_sampler, collate_fn=multiple_samples_collate)

    # for frames, label in tqdm(trdataloader):
    #     print(frames.shape)
    # exit()


    ts_dataloader_gen = KineticsDL('test', 1, False, spatial_flex, flexivit)
    tsdataloader = DataLoader(ts_dataloader_gen, num_workers=8, batch_size=1, collate_fn=multiple_samples_collate)

    for i, (frames, label) in enumerate(tqdm(tsdataloader)):
        # print(frames.shape)
        pass