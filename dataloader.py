import numpy as np
import os
import pandas as pd
import numpy as np
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
# os.environ['DECORD_EOF_RETRY_MAX'] = '40960'
decord.bridge.set_bridge('torch')


def replace_actor(text):
    # print('pos: ', text.replace(" c ", " the actor ").replace("C ", "The actor ").replace("C'", "The actor'").replace("c'", "the actor'").replace('the character', ''))
    return text.replace(" c ", " the actor ").replace("C ", "The actor ").replace("C'", "The actor'").replace("c'", "the actor'")

def prep_vqa_data(videos_folder, question_data, answer_data):
    video_question_pairs = []
    for answer in answer_data:
        q_uid = answer
        cur_answer = answer_data[q_uid]
        for item in question_data:
            if item['q_uid'] == q_uid:
                cur_question, option_0, option_1, option_2, option_3, option_4 = item['question'], item['option 0'], item['option 1'], item['option 2'], item['option 3'], item['option 4']
                break
        if cur_question is None:
            continue
        cur_video = f'{videos_folder}{q_uid}.mp4'
        formatted_question = format_question(cur_question, option_0, option_1, option_2, option_3, option_4)
        video_question_pairs.append([cur_answer, q_uid, cur_video, formatted_question])
    return video_question_pairs


def format_question(question, option_0, option_1, option_2, option_3, option_4):
    question = replace_actor(question)
    option_0 = replace_actor(option_0)
    option_1 = replace_actor(option_1)
    option_2 = replace_actor(option_2)
    option_3 = replace_actor(option_3)
    option_4 = replace_actor(option_4)



    ####### VIDEO LLAVA ########
    # # prompt 1
    # #formatted_question = f'Answer the following multiple choice question with an answer, either 0, 1, 2, 3, or 4.  {question} Option 0: {option_0} Option 1: {option_1} Option 2: {option_2} Option 3: {option_3} Option 4: {option_4} Respond with the selected answer choice.'
    # # prompt 2
    # N = "\n"
    # formatted_question = f': Answer the following question about the video by picking the best option Question: {question} {N} is it Option 0: {option_0} is it Option 1: {option_1} is it Option 2: {option_2} is it Option 3: {option_3} is it option 4: {option_4} Reply with the chosen option as a single character answer. Answer (0/1/2/3/4).'
    #
    # # prompt 1 ABCDE
    # #formatted_question = f'Answer the following multiple choice question with an answer, either A, B, C, D, or E.  {question} Option A: {option_0} Option B: {option_1} Option C: {option_2} Option D: {option_3} Option E: {option_4} Respond with the selected answer choice.'
    #
    #
    # # print(formatted_question)
    #return formatted_question
    ####### VIDEO LLAVA ########

    return [question, option_0, option_1, option_2, option_3, option_4]




class omniDataLoader(Dataset):
    def __init__(self, data_split, num_frames):
        self.data_split = data_split
        self.num_frames = num_frames
        print('EgoSchema val frames: ', self.num_frames)
        self.data = []
        if data_split == "train":
            self.videos_folder = '/home/c3-0/datasets/Ego4D/videos/h264/' # '/home/c3-0/datasets/Ego4D/Ego4D/ego4d_data/v2/video_540ss/'
            self.videos = sorted([x for x in os.listdir(self.videos_folder) if x.endswith('.mp4')])
            with open("/home/c3-0/datasets/Ego4D/Ego4D/narration.json", 'r') as f:
                self.narrations = json.load(f)
                for vid in self.videos:
                    keys = self.narrations[vid[:-4]].keys()
                    if 'narration_pass_1' not in keys or 'narration_pass_2' not in keys:
                        self.videos.remove(vid)
                        continue
                    if not self.narrations[vid[:-4]]['narration_pass_1']['summaries']:
                        for summary in self.narrations[vid[:-4]]['narration_pass_2']['summaries']:
                            self.data.append([vid, summary])
                    elif not self.narrations[vid[:-4]]['narration_pass_2']['summaries']:
                        for summary in self.narrations[vid[:-4]]['narration_pass_1']['summaries']:
                            self.data.append([vid, summary])
                    else:
                        for summary in self.narrations[vid[:-4]]['narration_pass_1']['summaries']:
                            self.data.append([vid, summary])
                        for summary in self.narrations[vid[:-4]]['narration_pass_2']['summaries']:
                            self.data.append([vid, summary])

            print(len(self.data), len(self.narrations), self.data[0])
            ######## Somehow corrupted data #######
            # self.data.remove(['73257595-8d1a-40b6-a0f6-03818bf7390b.mp4', {'start_sec': 1620.0333333333333, 'end_sec': 1811.2666666666667, 'summary_text': '#Summary C stood in a kitchen, wiped a gas stove with a cloth and swept a room with a broom.', 'annotation_uid': '2925e3aa-dd2a-4e83-81bc-902a92e7616b'}])
            # self.data.remove(['73257595-8d1a-40b6-a0f6-03818bf7390b.mp4', {'start_sec': 1620.0333333333333, 'end_sec': 1811.2666666666667, 'summary_text': '#Summary c wiped the table, wiped the cooker , took out a bucket from the cabinet placed it on the table , swept the room', 'annotation_uid': '45958d69-04ab-4dc7-80a7-1d83b739aa69'}])
            self.data.remove(['b029658f-073b-4df7-a74d-f72753277ba0.mp4', {'start_sec': 1890.0, 'end_sec': 1947.2333333333333, 'summary_text': '#Summary C sat at a table, played Robo rally and packed the game inside a box.', 'annotation_uid': '630c030c-a44c-4ac1-b950-119bdfd6429f'}])
            print(len(self.data), len(self.narrations))
            ######## Somehow corrupted data #######


            ################################ REMOVE AFTER PRELIM RESULTS #####################################
            random.shuffle(self.data)
            self.data = self.data[:5000]
            print(len(self.data))

        else:
            self.videos_folder = '/home/c3-0/datasets/EgoSchema/EgoSchema/videos/videos/'
            question_json_path = '/home/c3-0/datasets/EgoSchema/EgoSchema/questions.json'
            answer_json_path = '/home/c3-0/datasets/EgoSchema/EgoSchema/subset_answers.json'
            with open(answer_json_path, 'r') as f:
                answer_data = json.load(f)
            with open(question_json_path, 'r') as f:
                question_data = json.load(f)
            vqa_data = prep_vqa_data(self.videos_folder, question_data, answer_data)
            self.data = vqa_data

            print(len(self.data))

        self.transforms = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize([224, 224])
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        frames = []
        if self.data_split == 'train':
            vid, summary = self.data[index]
            vid_path = os.path.join(self.videos_folder, vid)
            # probe = ffmpeg.probe(vid_path)
            # video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            # width = int(video_stream['width'])
            # height = int(video_stream['height'])
            # test = '/home/c3-0/datasets/Ego4D/videos/h264/509160ad-d4be-4e15-87d1-fd739a18fb14.mp4'
            start, end, sum = summary['start_sec'], summary['end_sec'], summary['summary_text']
            vr = VideoReader(vid_path)
            frame_indexer = np.linspace(0, len(vr) - 1, 64)
            frames = vr.get_batch(frame_indexer).to(torch.bfloat16)


            # print(start, end)
            # start = str(timedelta(seconds=start))
            # end = str(timedelta(seconds=end))
            # # print(start, end, type(start))
            #
            # out, err = (
            #     ffmpeg
            #     .input(vid_path, ss=start, to=end)
            #     .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='quiet')
            #     .run(capture_stdout=True)
            # )
            # video = (
            #     np
            #     .frombuffer(out, np.uint8)
            #     .reshape([-1, height, width, 3])
            # )
            # print(video.shape)
            # frames = video[np.linspace(0, video.shape[0] - 1, 64).astype(int)]
            # frames = torch.stack(torch.from_numpy(frames))
            caption = sum.replace('#Summary ', '').replace(" c ", " the actor ").replace("C ", "The actor ").replace(
                "C'", "The actor'").replace("c'", "the actor'")

            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = self.transforms(frames)




                # start = int(start * 30)
                # end = int(end * 30)
                #
                #
                # cap = cv2.VideoCapture(vid_path)
                # for i in frame_indexer:
                #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                #     res, frame = cap.read()
                #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     frames.append(torch.from_numpy(frame))
                # frames = torch.stack(frames)

            return frames, caption

        else:
            gt, vid_id, vid_path, qas = self.data[index]
            vr = VideoReader(vid_path)
            frame_indexer = np.linspace(0, len(vr) - 1, self.num_frames)
            frames = vr.get_batch(frame_indexer)
            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = self.transforms(frames).to(torch.bfloat16)
            question, answers = qas[0], qas[1:]
            
            return frames, answers, gt


if __name__ == '__main__':
    shuffle = False
    dataloader_gen = omniDataLoader('train')
    dataloader = DataLoader(dataloader_gen, num_workers=0, batch_size=1)
    for frames, caption in tqdm(dataloader):
        print(frames.shape, ''.join(caption))