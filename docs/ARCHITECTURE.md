# FusionXDrive Architecture

## Overview

FusionXDrive is a multi-modal VLM-based system for autonomous driving scene understanding and trajectory planning. It fuses RGB camera, LiDAR, and 4D mmWave Radar data through learnable bridges into a frozen LLM, producing structured scene captions and future trajectory waypoints.

## Pipeline

```
RGB Image ──► DINOv3 ──────────────┐
LiDAR PCD ──► VoxelNeXt ──────────┼──► Q-Former/MoRo-Former ──► Qwen2.5-0.5B ──┬──► Scene Caption (JSON)
Radar PCD ──► RadarNeXt ──────────┘                                               │
                                                                                  └──► Diffusion Planner ──► Waypoints [8×3]
```

## Component Details

### 1. Vision Encoder: DINOv3-base
- **Model**: `facebook/dinov3-base`
- **Output**: [B, 256, 768] patch features (CLS token dropped)
- **Status**: Frozen during all training stages
- **Rationale**: DINOv3 provides robust visual features pre-trained on diverse data

### 2. LiDAR Encoder: VoxelNeXt
- **Reference**: Chen et al., CVPR 2023
- **Process**: Points → Pillarization → BEV pseudo-image → 6-stage backbone
- **Backbone**: SparseConv blocks with strides {1,2,4,8,16,32}
- **Input**: (x, y, z, intensity) = 4 channels
- **Pillar size**: [0.2, 0.2, 8.0]m
- **Output**: [B, 256] via GAP on multi-scale fused features

### 3. Radar Encoder: RadarNeXt
- **Reference**: Jia et al., 2025
- **Process**: Points → Pillarization → BEV pseudo-image → Rep-DWC backbone + MDFEN neck
- **Backbone**: 3-stage re-parameterizable depthwise convolution blocks
- **Neck**: MDFEN (Multi-path Deformable Foreground Enhancement Network) with DCNv3
- **Input**: (x, y, z, doppler, power, recoveredSpeed) = 6 channels
- **Pillar size**: [0.32, 0.32, 8.0]m (larger for sparser radar)
- **Output**: [B, 256] via GAP

### 4. Bridge Module

#### Q-Former (Standard)
- **Reference**: BLIP-2 (Li et al., 2023)
- 64 learnable query tokens, 4 transformer blocks
- Cross-attends to concatenated [image + LiDAR + radar] features
- Output: [B, 64, 896] projected to LLM space

#### MoRo-Former (Task-Aware)
- 9 task-specific query groups × 8 queries each = 72 internal queries
- 4 expert branches: i (image), il (image+LiDAR), ir (image+Radar), ilr (trimodal)
- 3 task groups:
  - **Image-dominant** (weather, traffic_light, traffic_sign): → image expert
  - **Adaptive-routing** (participants, hazard, drivability, lane, advice): → learned router
  - **Global** (explanation): → trimodal expert
- Lightweight router with Gumbel-softmax training

### 5. LLM: Qwen2.5-0.5B-Instruct
- Hidden size: 896
- Frozen base + LoRA adapters (r=16, alpha=32) on q/k/v/o projections
- `<|image_pad|>` placeholder tokens replaced with Q-Former output

### 6. Planner: Truncated Diffusion Planner
- **Reference**: Inspired by DiffusionDrive
- **Output**: 8 waypoints at 0.5s intervals (0.5s–4.0s)
- **Anchors**: Speed × curvature grid, initialized via farthest-point sampling
- **Denoiser**: 4-block MLP with FiLM conditioning (trained for t=1..5)
- **Inference**: Single-step anchor correction at t=1
- **Loss**: Reconstruction + ADE + anchor scoring + direct regression + smoothness

## Data Flow

### Training (Stage 1: VQA)
```
Image → DINOv3 ────────────┐
LiDAR → VoxelNeXt ────────┼──► Q-Former ──► LLM → CrossEntropy(pred, gt_caption)
Radar → RadarNeXt ────────┘
                                    ↑
                          LoRA (trainable)
```

### Training (Stage 2: Planning)
```
(Frozen VQA pipeline) → planning_tokens → Diffusion Planner → L1/L2(pred, gt_waypoints)
```

### Inference
```
All 3 modalities → Q-Former → LLM → {caption tokens, planning tokens}
                                              │
                              ┌───────────────┘
                              ▼
                    Diffusion Planner → waypoints [8×3]
```

## Key Design Decisions

1. **Frozen vision + LLM**: Reduces trainable parameters and prevents catastrophic forgetting
2. **LoRA over full fine-tuning**: Parameter-efficient, ~2M extra params vs ~500M
3. **Anchor-based diffusion**: Faster inference (single-step) than full diffusion
4. **Three-stage progressive training**: Captioning warm-up → routing learning → planner training, decouples VQA quality from planning accuracy
5. **Modality routing**: MoRo-Former adaptively selects relevant sensors per task

## Learning Scheme

FusionXDrive is trained in three progressive stages:

1. **Captioning Warm-up**: Modality experts and structured captioning branch are warmed up with hard routing disabled. Each adaptive subtask is decoded by all four experts in parallel. Optimized by autoregressive captioning loss only.

2. **Routing Learning** (TODO — not fully open-sourced): Experts and captioning branch are frozen. Only the modality router is trained with sensor dropout (LiDAR/Radar dropped with p=0.5) and pseudo-label supervision (expert with lowest captioning loss per subtask).

3. **Planner Training**: Truncated diffusion-based planner is trained while freezing all upstream modules. GT trajectories are matched to nearest anchors; the denoiser learns residual corrections. Jointly optimized for anchor scoring and residual denoising.

See [TRAINING.md](TRAINING.md) for full training commands and hyperparameters.
