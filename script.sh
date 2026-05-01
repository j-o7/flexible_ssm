#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -n8
#SBATCH --constraint=gpu80
#SBATCH -p highgpu
#SBATCH --job-name=flex-mvit-kinetics
#SBATCH --mem-per-cpu=20G
#SBATCH --output=./logs/%x.out



# Load modules
module load cuda
module load ffmpeg
module list
source activate /home/anaconda3/envs/mamba/

# "/home/VideoMamba/videomamba/video_sm/videomamba_m16_k400_f16_res224.pth"
# "checkpoints/st-static-tokens_epoch_12_74.52332072539392.pt"

#director  genre  like_ratio  relationship  scene  view_count  way_speaking  writer  year

python3 flexVM.py --dataset lvu --model videomamba --frames 32 --resolution 288 --genre scene --ckpt "/home/VideoMamba/videomamba/video_sm/videomamba_m16_k400_f16_res224.pth" # --train "probe"
#bash /home/VideoMamba/videomamba/video_sm/exp/k400/videomamba_middle/run_f8x224.sh
#bash /home/VideoMamba/videomamba/video_sm/exp/ssv2/videomamba_middle/run_f16x224.sh


