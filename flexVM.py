import timeit

import cv2
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import decord
import statistics
import argparse
import hiera
import pickle
import os

from decord import VideoReader

decord.bridge.set_bridge('torch')
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.video import r3d_18, R3D_18_Weights, swin3d_b, Swin3D_B_Weights, mvit_v1_b
from tqdm import tqdm
from timm.models import create_model

from models import *


from kinetics_dataloader import KineticsDL, multiple_samples_collate
from COIN_loader import COINDL
from ucf_dataloader import UCFDL
from breakfast_loader import BkfstDL
from hmdb_loader import HMDBDL

from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils
from collections import OrderedDict

import time
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import numpy as np


# run retrieval evaluation on the validation set



# extract train and test features and compute retrieval accuracy
def run_train_test_ret(model, train_loader, val_loader, args):
    model.cuda()
    model.eval()
    train_features = []
    test_features = []
    train_labels = []
    labels_list = []
    
    for frames, labels in tqdm(train_loader):
        frames, labels = frames.cuda(), labels.cuda()
        if args.model == 'videomamba':
            # print(frames.shape)
            feat = model.forward_features(frames)


        zipped = zip(feat, labels)
        for feature, lbl in zipped:
            train_features.append(feature.detach().cpu())
            train_labels.append(lbl.detach().cpu().item())
    
    for frames, labels in tqdm(val_loader):
        frames, labels = frames.cuda(), labels.cuda()
        if args.model == 'videomamba':
            feat = model.forward_features(frames)

        zipped = zip(feat, labels)
        for feature, lbl in zipped:
            test_features.append(feature.detach().cpu())
            labels_list.append(lbl.detach().cpu().item())

    train_features = torch.stack(train_features)
    test_features = torch.stack(test_features)



    correct = 0
    print(train_features.shape, test_features.shape)
    # dist_mat = []
    for i, probe in enumerate(tqdm(test_features)):
        probe_sim = torch.nn.CosineSimilarity()(probe.unsqueeze(0).detach().cpu(), train_features.detach().cpu())
        # dist_mat.append(probe_sim.cuda())
        first, arg = torch.topk(probe_sim.flatten(), 2).indices
        # print(labels_list[i] == labels_list[arg.item()])
        if labels_list[i] == train_labels[first.item()]:
            correct += 1
    # dist_mat = torch.stack(dist_mat)
    # print(dist_mat.shape)
    # torch.save(dist_mat, 'u-96-base-distmat.pt')
    print(i)
    accuracy = correct / i
    print(correct, i)
    print(f'Test Accuracy: {accuracy}')
    return accuracy



            


# load the model and run flexible evaluation or retrieval
def eval_flexVM(args):
    num_frames = int(args.frames)
    resolution = int(args.resolution)
    ######################### LOAD VM MODEL ####################################
    if args.model == 'videomamba':
        model_name = 'videomamba_middle'
        cpath = args.ckpt
        checkpoint = torch.load(cpath)
        static_tokens = True if 'static-tokens' in cpath else False
        flex_all = True if 'flex_all' in cpath else False
        flexivit = True if 'flexivit' in cpath else False
#        if '.pth' in cpath:
#            model_frames = 8
#            checkpoint = checkpoint['model']
#            nb_classes = 101
        if 'k400' in cpath: #only do this for vm baseline weights
            ch_frames = int(cpath.split('_')[4][1:])
            print(ch_frames)
            model_frames = ch_frames
            nb_classes = 400
        else:
            model_frames = 64
            nb_classes = 400
        #model_frames = 64 if 'f16' not in cpath[:-4] else 16
        spatial_flex = False if (resolution == 224 or static_tokens or flexivit) else True

        model = create_model(
            model_name,
            img_size=224,
            pretrained=None,
            num_classes=nb_classes,
            fc_drop_rate=0,
            drop_path_rate=0.4,
            kernel_size=1,
            num_frames=model_frames,
            use_checkpoint=True,
            checkpoint_num=0,
            flexible=True,
            spatial_flex=spatial_flex,
            flexivit=flexivit,
            flex_all=flex_all,
            static_tokens=static_tokens,
        )



        new_dict = OrderedDict()
        all_keys = list(checkpoint.keys())

        for key in all_keys:
            if key.startswith('module.'):
                new_dict[key[7:]] = checkpoint[key]
            else:
                new_dict[key] = checkpoint[key]
        checkpoint = new_dict
        model.load_state_dict(checkpoint, strict=False)
        print('weights loaded: ', cpath)

        ####################################### Interpolate Temporal Emebeddings ###################################
        if model_frames != num_frames:
            print(f"Temporal interpolate from {model_frames} to {num_frames}")
            temp_pos_embed = model.temporal_pos_embedding.permute(0, 2, 1)
            temp_pos_embed = torch.nn.functional.interpolate(
                temp_pos_embed, size=(num_frames,), mode='linear', align_corners=False
            )
            temp_pos_embed = temp_pos_embed.permute(0, 2, 1)
            model.temporal_pos_embedding = nn.Parameter(temp_pos_embed)
        else:
            print('no interp needed: ', model_frames, num_frames)
        ####################################### Interpolate Temporal Emebeddings ###################################


    
    if num_frames < 32 and resolution < 384:
        bs = 4
    elif num_frames == 32 and resolution < 288:
        bs = 2
    else:
        bs = 1
    #bs = 2
    ####################################### Interpolate Positional Emebeddings ###################################
    # if not static_tokens:
    #     orig_size = 14
    #     # height (== width) for the new position embedding
    #     new_size = int(resolution / 16)
    #     num_extra_tokens = 1  # appended cls token  = +1
    #     print(orig_size, new_size)
    #     embedding_size = model.pos_embed.shape[-1]
    #     if orig_size != new_size:
    #         print("Position interpolate from %d to %d" % (orig_size, new_size))
    #         extra_tokens = model.pos_embed[:, :num_extra_tokens]
    #         # only the position tokens are interpolated
    #         pos_tokens = model.pos_embed[:, num_extra_tokens:]
    #         # B, L, C -> B, H, W, C -> B, C, H, W
    #         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    #         pos_tokens = torch.nn.functional.interpolate(
    #             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    #         # B, C, H, W -> B, H, W, C ->  B, H, W, C
    #         pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size, new_size, embedding_size)
    #         pos_tokens = pos_tokens.flatten(1, 2)  # B, L, C
    #         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    #         model.pos_embed = nn.Parameter(new_pos_embed)

    ####################################### Interpolate Positional Emebeddings ###################################

    model.to('cuda')
    
    
    ######################### Init Data ####################################
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")  # 74,503,818
    model.eval()
    print('model loaded, all layers frozen')
    if args.dataset == 'kinetics':
        train_dataset = KineticsDL('train', num_frames=num_frames, resolution=resolution)
        test_dataset = KineticsDL('test', num_frames=num_frames, resolution=resolution)
    elif args.dataset == 'coin':
        train_dataset = COINDL('train', num_frames=num_frames, resolution=resolution)
        test_dataset = COINDL('test', num_frames=num_frames, resolution=resolution)
    elif args.dataset == 'ucf':
        train_dataset = UCFDL('train', num_frames=num_frames, resolution=resolution)
        test_dataset = UCFDL('test', num_frames=num_frames, resolution=resolution)

    elif args.dataset == 'breakfast':
        train_dataset = BkfstDL('train', num_frames=num_frames, resolution=resolution)
        test_dataset = BkfstDL('test', num_frames=num_frames, resolution=resolution)
    elif args.dataset == 'hmdb':
        train_dataset = HMDBDL('train', num_frames=num_frames, resolution=resolution)
        test_dataset = HMDBDL('test', num_frames=num_frames, resolution=resolution)



    test_loader = DataLoader(test_dataset, num_workers=8, batch_size=bs, shuffle=False, collate_fn=multiple_samples_collate, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=bs, shuffle=False, collate_fn=multiple_samples_collate, drop_last=True)
    ######################### Init Data ####################################

    ######################### Eval ####################################

    run_train_test_ret(model, train_loader, test_loader, args)
    print('frames + reso: ', num_frames, args.resolution, args.dataset, args.model)
    print('weights loaded: ', cpath)




    ######################### Eval ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('--dataset', choices=['kinetics', 'coin', 'smthsmth', 'ucf', 'breakfast', 'hmdb'], required=True)
    parser.add_argument('--model', choices=['videomamba'], required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--resolution', required=True)
    parser.add_argument('--ckpt', required=True)

    args = parser.parse_args()
    eval_flexVM(args)