# Evaluation and Training

This repository provides scripts for **evaluating** and **training** VideoMamba-based models, including support for several forms of **spatio-temporal flexible training**.

---

## 1. Evaluation (`flexVM.py`)

Runs **video retrieval evaluation** using a pretrained VideoMamba checkpoint.

### Usage

```bash
python flexVM.py \
    --dataset <dataset_name> \
    --model videomamba \
    --frames <num_frames> \
    --resolution <height> \
    --ckpt <checkpoint_path>
```

### Arguments

| Flag | Purpose |
|------|---------|
| `--dataset` | One of: `coin`, `smthsmth`, `ucf`, `breakfast`, `hmdb`. |
| `--model` | Currently supports only `videomamba`. |
| `--frames` | Number of frames per clip. |
| `--resolution` | Input spatial resolution. |
| `--ckpt` | Path to pretrained checkpoint. |

### What It Does
1. Loads VideoMamba + checkpoint.  
2. Interpolates temporal embeddings if frame count differs from checkpoint.  
3. Builds train/test splits for the chosen dataset.  
4. Extracts features for all clips.  
5. Computes nearest-neighbor retrieval accuracy and prints the final score.

### Example

```bash
python flexVM.py \
    --dataset kinetics \
    --model videomamba \
    --frames 16 \
    --resolution 224 \
    --ckpt checkpoints/vm_k400.pth
```

---

## 2. Training (`run_class_finetuning.py`)

Script for classification fine-tuning on datasets like Kinetics.  
Supports multiple flexible training modes. **NOTE: A lot of the training code was taken from the original VideoMamba codebase (https://github.com/OpenGVLab/VideoMamba) with some changes made to support flexible training/data loading.**

### Basic Usage (single GPU)

```bash
python run_class_finetuning.py \
    --data_set Kinetics \
    --data_path /path/to/data \
    --model videomamba \
    --num_frames 16 \
    --input_size 224 \
    --batch_size 8 \
    --epochs 30 \
    --finetune /path/to/checkpoint.pth \
    --output_dir outputs/run1
```

### Important Options

#### Dataset / Input
| Flag | Meaning |
|------|---------|
| `--data_set` | Dataset name (e.g., `Kinetics`, `SSV2`, `UCF101`, `HMDB51`, etc.). |
| `--data_path` | Root directory containing videos. |
| `--num_frames` | Number of frames per clip. |
| `--input_size` | Spatial resolution. |

#### Model / Finetuning
| Flag | Meaning |
|------|---------|
| `--model` | Model variant (e.g., `videomamba`). |
| `--finetune` | Load pretrained checkpoint for fine-tuning. |

#### Training
| Flag | Meaning |
|------|---------|
| `--batch_size` | Training batch size. |
| `--epochs` | Number of epochs. |

---

## Flexibility Modes

Enable one of the following:

| Flag | Description |
|------|-------------|
| `--flexible` | Temporal flexibility (varies frame count / sampling). |
| `--spatial_flex` | Spatial flexibility (varies image resolution). |
| `--flexivit` | FlexiViT-style (flexible patch size + pos-emb). |
| `--flex_all` | Full spatio-temporal flexibility (resolution + patch size + pos-emb). |
| `--static_tokens` | Flexible image/patch size but static positional tokens. |

Inside training, these simply toggle attributes on the model (e.g., `model.module.flex_all = True`), enabling the chosen flexibility behavior.

---

## Example Training Commands

### A. Standard Fine-Tuning (no flexibility)

```bash
python run_class_finetuning.py \
    --data_set Kinetics \
    --data_path /path/to/K400 \
    --model videomamba \
    --num_frames 16 \
    --input_size 224 \
    --batch_size 8 \
    --epochs 30 \
    --finetune checkpoints/vm_k400.pth \
    --output_dir outputs/static_run
```

### B. Temporal Flexible Training

```bash
python run_class_finetuning.py \
    --data_set Kinetics \
    --data_path /path/to/K400 \
    --model videomamba \
    --num_frames 16 \
    --input_size 224 \
    --batch_size 8 \
    --epochs 30 \
    --finetune checkpoints/vm_k400.pth \
    --flexible \
    --output_dir outputs/temporal_flex
```

### C. Full Spatio-Temporal Flexible Training

```bash
python run_class_finetuning.py \
    --data_set Kinetics \
    --data_path /path/to/K400 \
    --model videomamba \
    --num_frames 16 \
    --input_size 224 \
    --batch_size 8 \
    --epochs 30 \
    --finetune checkpoints/vm_k400.pth \
    --flex_all \
    --output_dir outputs/flex_all_run
```

### D. FlexiViT-Style

```bash
python run_class_finetuning.py \
    --data_set Kinetics \
    --data_path /path/to/K400 \
    --model videomamba \
    --num_frames 16 \
    --input_size 224 \
    --batch_size 8 \
    --epochs 30 \
    --finetune checkpoints/vm_k400.pth \
    --flexivit \
    --output_dir outputs/flexivit_run
```
