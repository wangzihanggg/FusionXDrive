<p align="center">
  <img src="assets/logo_fusionxdrive.png" alt="FusionXDrive" width="300">
</p>

A multi-modal VLM for driving scene understanding and trajectory planning, fusing RGB, LiDAR, and 4D Radar through learnable bridges into a frozen LLM.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FusionXDrive Pipeline                         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Left RGB    │  │  LiDAR PCD   │  │  Radar PCD   │          │
│  │  (720×540)    │  │  (x,y,z,i)   │  │(x,y,z,d,p,s)│          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│         ▼                  ▼                  ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   DINOv3     │  │  VoxelNeXt   │  │  RadarNeXt   │          │
│  │  (frozen)    │  │  Encoder     │  │  Encoder     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                   │
│         └──────────────────┼──────────────────┘                  │
│                            ▼                                     │
│                   ┌────────────────┐                             │
│                   │   Q-Former /   │                             │
│                   │   MoRo-Former  │                             │
│                   │   Bridge       │                             │
│                   └───────┬────────┘                             │
│                           │                                      │
│                           ▼                                      │
│                  ┌─────────────────┐                             │
│                  │   Qwen2.5-0.5B  │                             │
│                  │   + LoRA        │                             │
│                  │   (frozen LLM)  │                             │
│                  └────────┬────────┘                             │
│                           │                                      │
│              ┌────────────┴────────────┐                         │
│              ▼                         ▼                          │
│     ┌─────────────────┐    ┌──────────────────┐                  │
│     │  Scene Caption   │    │  Trajectory       │                  │
│     │  (VQA JSON)      │    │  Waypoints (8×3)  │                  │
│     └─────────────────┘    └──────────────────┘                  │
│                              ↑                                    │
│                     ┌────────────────────┐                        │
│                     │ Truncated Diffusion │                       │
│                     │ Planner             │                       │
│                     └────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Multi-modal fusion**: RGB (DINOv3) + LiDAR (VoxelNeXt) + 4D Radar (RadarNeXt) → Q-Former → LLM
- **Two bridge variants**: Standard Q-Former + MoRo-Former (Mixture-of-Experts routing)
- **Three-stage progressive training**: Captioning warm-up → Routing learning → Planner training
- **LoRA fine-tuning**: Efficient LLM adaptation with minimal trainable parameters
- **Multi-dataset support**: UNISCP（待发布）

## Project Structure

```
FusionXDrive/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
│
├── fusionxdrive/                    # Core Python package
│   ├── __init__.py                  # Package init, version, exports
│   ├── dataset.py                   # UNISCP dataset loader
│   ├── dataset_planning.py          # Planning data utilities
│   ├── model_vqa.py                 # MultiModalVLM (VQA model)
│   ├── model_planning.py            # MultiModalVLM + Planning
│   ├── point_cloud_encoders.py      # VoxelNeXt + RadarNeXt
│   ├── qformer.py                   # Q-Former bridge
│   ├── moro_former.py               # MoRo-Former (MoE bridge)
│   ├── diffusion_planner.py         # Truncated Diffusion Planner
│   └── metrics.py                   # Evaluation metrics
│
├── scripts/                         # Executable scripts
│   ├── train_vqa.py                 # VQA training
│   ├── train_planning.py            # Two-stage training (VQA + Planning)
│   ├── evaluate_vqa.py              # VQA evaluation
│   ├── evaluate_planning.py         # VQA + Planning evaluation
│   └── inference_vqa.py             # Single-frame inference
│
├── configs/                         # Configuration files
│   ├── inference_config.py
│   └── inference_config_batch.py
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md
│   └── TRAINING.md
│
└── assets/                          # Static assets
    └── logo_fusionxdrive.png
```

## Installation

```bash
git clone https://github.com/your-org/FusionXDrive.git
cd FusionXDrive
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Data Preparation

> **Note**: The UniSCP dataset is currently **待发布** (to be released). The dataset structure below is provided for reference.

Expected dataset structure under `UNISCP/`:

```
UNISCP/
├── CALIBRATION/
│   ├── 1_CAMERA/
│   ├── 3_CAMERA_LIDAR/
│   └── 4_CAMERA_RADAR/
├── RURAL_A0/
│   ├── 1_IMAGE/1_IMAGE/LEFT/
│   ├── 2_LIDAR/2_LIDAR/PCD/
│   ├── 3_RADAR/3_RADAR/PCD/
│   └── 6_CAPTION/
└── ...
```

### Training (Three-Stage Progressive)

FusionXDrive uses a three-stage progressive training protocol. **Note: Stage 2 (routing learning) is not yet fully open-sourced** — the current code combines it with Stage 3. See [docs/TRAINING.md](docs/TRAINING.md) for the complete paper design.

**Stage 1: Captioning Warm-up**
```bash
python scripts/train_vqa.py \
    --data_root ./UNISCP \
    --output_dir save/stage1_captioning \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_epochs 10 \
    --sequences RURAL_A0 RURAL_A1 RURAL_A2
```

**Stage 2: Routing Learning** (TODO — not fully public)
- Trains only the MoRo-Former router with sensor dropout + pseudo-label supervision
- LiDAR/Radar independently dropped with probability 0.5

**Stage 3: Planner Training** (code: `--stage 2`)
```bash
python scripts/train_planning.py --stage 2 \
    --data_root ./UNISCP \
    --output_dir save/stage3_planning \
    --vqa_checkpoint save/stage1_captioning/final_model \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --planning_loss_weight 2.0
```

| Stage | Paper | Public Code |
|-------|-------|-------------|
| 1. Captioning warm-up | Hard routing disabled, all experts | `scripts/train_vqa.py` |
| 2. Routing learning | Sensor dropout + pseudo labels | Not public |
| 3. Planner training | Truncated diffusion | `scripts/train_planning.py --stage 2` |

### Inference

```bash
python scripts/inference_vqa.py \
    --model_path save/stage1_captioning/final_model \
    --data_root ./UNISCP \
    --sequence RURAL_A0 \
    --frame_idx 000100
```

### Evaluation

```bash
# VQA only (after Stage 1)
python scripts/evaluate_vqa.py \
    --model_path save/stage1_captioning/final_model \
    --data_root ./UNISCP \
    --sequences RURAL_A0

# VQA + Planning (after Stage 3)
python scripts/evaluate_planning.py \
    --model_path save/stage3_planning/final_model \
    --data_root ./UNISCP \
    --sequences RURAL_A0
```

## Model Details

### Encoders
- **DINOv3-base** (ViT-B/14): Frozen vision encoder, extracts patch features [B, 256, 768]
- **VoxelNeXt** (LiDAR): Pillarization + 6-stage sparse CNN backbone, outputs [B, 256]
- **RadarNeXt** (4D Radar): Pillarization + Rep-DWC backbone + MDFEN neck, outputs [B, 256]

### Bridge
- **Q-Former**: 64 learnable query tokens cross-attend to multi-modal features, 4 transformer blocks
- **MoRo-Former**: Task-aware modality routing with 4 expert branches and learnable router

### LLM
- **Qwen2.5-0.5B-Instruct**: Frozen base with LoRA adapters on attention projections

### Planner
- **Truncated Diffusion Planner**: Anchor-based, 8-waypoint trajectory prediction at 0.5s intervals
- Output: [B, 8, 3] waypoints in ego frame (x=forward, y=left, z=up)

## Trainable Parameters

| Component | Parameters |
|-----------|-----------|
| VoxelNeXt encoder | ~1.5M |
| RadarNeXt encoder | ~1.2M |
| Q-Former bridge | ~8M |
| LoRA adapters (r=16) | ~2M |
| Diffusion planner | ~6M |
| **Total** | **~18.7M** |

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{fusionxdrive2025,
  title={FusionXDrive: Multi-Modal VLM for Driving Scene Understanding and Trajectory Planning},
  year={2025},
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgements

This project builds upon several excellent works:
- [BLIP-2](https://arxiv.org/abs/2301.12597) - Q-Former architecture
- [VoxelNeXt](https://arxiv.org/abs/2303.11301) - LiDAR point cloud encoding
- [RadarNeXt](https://arxiv.org/abs/2025) - 4D radar point cloud encoding
- [DiffusionDrive](https://arxiv.org/abs/2025) - Truncated diffusion for trajectory planning
