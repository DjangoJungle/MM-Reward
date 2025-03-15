# Multi-Modal Reward Model (MMRM)

This repository contains a pipeline of training reward model for multi-modal preference learning, built on top of the Qwen2-VL-2B-Instruct model.

## Overview

Reward models are critical components in aligning large language models with human preferences. This project explores innovative architectures to enhance the performance of multi-modal reward models, particularly for image-text understanding tasks.

> More details about this repo is in the "report.pdf" document

## Training & Evaluation

The models were trained on the MMPR (Multi-Modal Preference Ranking) dataset using LoRA fine-tuning with the following settings:

- Batch size: 1
- Gradient accumulation steps: 16
- Learning rate: 5e-6
- Training epochs: 3
- Optimizer: AdamW
- LR scheduler: Cosine annealing
- Mixed precision: FP16
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05

> For detailed steps on how to run it, you can look at the individual bash files in the first level directory

## Requirements

```txt
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
datasets>=2.14.0
peft>=0.6.0
trl>=0.7.4
wandb
pillow
matplotlib
tqdm 
```

