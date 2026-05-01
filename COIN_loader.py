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

# print('imported!!!')
# os.environ['DECORD_EOF_RETRY_MAX'] = '40960'
decord.bridge.set_bridge('torch')


# test_remove = ['084k_RL3ApU_000109_000119.mp4', '2xWiEVNUvhE_000064_000074.mp4', '305P2f9_lko_004145_004155.mp4',
#                'B4bn9G6__sY_000086_000096.mp4', 'BvBVQmm2RcM_000082_000092.mp4', 'CxjipYE57Yo_000199_000209.mp4',
#                'IhanWvpHGu8_001243_001253.mp4', 'Lw14NH9kAqE_000759_000769.mp4',
#                ' XFkykETgkoo_002967_002977.mp4', 'jJFqy6yiXzQ_000024_000034.mp4',
#                'kinMMqkswUk_000120_000130.mp4', 'y7cYaYX4gdw_000047_000057.mp4']

class COINDL(Dataset):
    def __init__(self, data_split, num_frames, resolution, viz=False, found=False):
        self.data = []
        self.labels = []
        self.num_frames = num_frames
        self.data_split = data_split
        self.root = '/squash/COIN/videos/'
        self.resolution = resolution
        self.viz = viz
        print('resolution: ', resolution)

        videos = os.listdir(self.root)
        data = json.load(open("/home/ny525072/VideoMamba/videomamba/video_sm/COIN.json", 'r'))['database']
        del data['iXRhbOqVtJE']
        del data['yT7RlBGsAEs']
        del data['1sNwO2P6j7M']
        del data['AfiVmAjfTNs']
        remove = []

        if data_split == 'train':
            for k, v in data.items():
                if f'{k}.mp4' not in videos or v['subset'] == 'testing':
                    remove.append(k)

            for ele in remove:
                del data[ele]


            for k, v in data.items():
                self.data.append([k, v])

        else:
            for k, v in data.items():
                if f'{k}.mp4' not in videos or v['subset'] == 'training':
                    remove.append(k)

            for ele in remove:
                del data[ele]


            for k, v in data.items():
                self.data.append([k, v])


        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([resolution, resolution]),
        ])
        self.labels = pd.read_excel("/home/ny525072/VideoMamba/videomamba/video_sm/taxonomy.xlsx")["Targets"].values.tolist()
        print(len(self.data), len(self.labels), len(remove))
        if found:
            print(self.data[0])
            self.data = [['i8NJo7jYnV4', {'class': 'PractisePoleVault'}], ['o0jerJWhfSs', {'class': 'PractisePoleVault'}], ['yiT8Z11rkMY', {'class': 'PractisePoleVault'}]]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, anno = self.data[index]
        vid_path = os.path.join(self.root, vid + '.mp4')
        try:
            vr = VideoReader(vid_path)
            frame_indexer = np.linspace(0, len(vr)-1, self.num_frames).astype(int)
            frames = vr.get_batch(frame_indexer)
        
            # probe = ffmpeg.probe(vid_path)
            # video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            # width = int(video_stream['width'])
            # height = int(video_stream['height'])
            # out, _ = (
            #     ffmpeg
            #         .input(vid_path, loglevel="quiet")
            #         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            #         .run(capture_stdout=True)
            # )
            # frames = (
            #     np
            #         .frombuffer(out, np.uint8)
            #         .reshape([-1, height, width, 3])
            # )
            # frames = torch.from_numpy(frames)
            # frame_indexer = np.linspace(0, frames.shape[0] - 1, self.num_frames).astype(int)
            # frames = frames[frame_indexer]
        
        except Exception as e:
            print(e, vid_path)
            if self.data_split == 'train':
                return torch.randn([3, self.num_frames, self.resolution, self.resolution]), 0
            else:
                return None, None
        frames = frames.permute(0, 3, 1, 2).float() / 255.
        if self.viz:
            print(vid)
            frames = transforms.Resize([self.resolution, self.resolution])(frames)

            label = self.labels.index(anno['class'])

            return frames, int(label), vid

        else:
            frames = self.transforms(frames).permute(1, 0, 2, 3)

            label = self.labels.index(anno['class'])

            return frames, int(label)


if __name__ == '__main__':
    shuffle = False
    dataloader_gen = COINDL('train', 2, 224)
    dataloader = DataLoader(dataloader_gen, num_workers=8, batch_size=2, collate_fn=None)
    for frames, label in tqdm(dataloader):
        pass