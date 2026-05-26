# Training Guide

## Three-Stage Training Protocol (Paper)

FusionXDrive is designed with a three-stage progressive training strategy:

### Stage 1: Captioning Warm-up

Warm up modality experts and the structured captioning branch. Hard routing in MoRo-Former is **disabled** — each adaptive subtask is decoded by all four experts in parallel, concatenated, and fed into the LLM.

**What's trainable**: LoRA adapters, Q-Former/MoRo-Former, LiDAR encoder, Radar encoder
**What's frozen**: DINOv3 vision encoder, Qwen2.5 LLM base weights
**Loss**: Autoregressive captioning loss only

```bash
python scripts/train_vqa.py \
    --data_root ./UNISCP \
    --output_dir save/stage1_captioning \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --num_epochs 10 \
    --lora_r 16 --lora_alpha 32 \
    --sequences RURAL_A0 RURAL_A1 RURAL_A2 RURAL_B0 RURAL_B1 RURAL_B2
```

### Stage 2: Routing Learning

> **TODO — NOT FULLY OPEN-SOURCED.** The routing stage in the paper trains only the MoRo-Former routing module, with sensor dropout augmentation and pseudo-label supervision. The current public code combines stages 2–3 in a simplified pipeline.

In the paper's full design:
- Freeze expert decoders and captioning branch
- Train only the modality routing module
- LiDAR and 4D Radar are independently dropped with probability 0.5 → 4 sensor settings (camera-only, camera+LiDAR, camera+Radar, trimodal)
- For each adaptive-routing subtask, the expert with the lowest captioning loss serves as the pseudo routing label
- Optimize: cross-entropy between router prediction and pseudo label

### Stage 3: Planner Training

Train the truncated diffusion-based planner while freezing all multimodal encoders and routing. Each GT trajectory is matched to its nearest anchor, and the planner learns the residual around this anchor.

**What's trainable**: Diffusion planner only (~6M params)
**What's frozen**: Everything from Stage 1 (encoders, bridge, LLM, LoRA)
**Loss**: Anchor scoring (CE) + residual denoising (MSE)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_planning.py --stage 2 \
    --data_root ./UNISCP \
    --output_dir save/stage3_planning \
    --vqa_checkpoint save/stage1_captioning/final_model \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --planning_loss_weight 2.0 \
    --sequences RURAL_A0
```

## Open-Source Status

| Stage | Paper Description | Public Code | Notes |
|-------|------------------|-------------|-------|
| Stage 1 | Captioning warm-up | `scripts/train_vqa.py` | Full implementation |
| Stage 2 | Routing learning (sensor dropout + pseudo labels) | NOT public | Simplified into Stage 3 |
| Stage 3 | Diffusion planner training | `scripts/train_planning.py --stage 2` | Full implementation |

The current public code combines the routing and planning stages by directly training the planner on top of the captioning checkpoint without explicit routing supervision. We plan to release the standalone routing stage in a future update.

## Hyperparameters

| Parameter | Stage 1 (Captioning) | Stage 3 (Planning) |
|-----------|---------------------|---------------------|
| Learning rate | 5e-6 | 1e-4 |
| Batch size | 4 | 2 |
| Grad accumulation | 8 | 16 |
| Effective batch | 32 | 32 |
| LoRA rank | 16 | N/A (frozen) |
| LoRA alpha | 32 | N/A |
| Max grad norm | 1.0 | 1.0 |
| Warmup steps | 200 | 100 |
| LR schedule | cosine | cosine |
| Epochs | 10 | 10 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |

## Dataset Configuration

### Supported Datasets

| Dataset | Modalities | Waypoints | Symbolic Link | Status |
|---------|-----------|-----------|---------------|
| UNISCP | RGB + LiDAR + 4D Radar | 1s, 2s, 5s, 10s | `ln -s /data/UNISCP ./UNISCP` | 待发布 |

## Evaluation

### VQA-only Evaluation

```bash
python scripts/evaluate_vqa.py \
    --model_path save/stage1_captioning/final_model \
    --data_root ./UNISCP \
    --sequences RURAL_A0 RURAL_A1 RURAL_A2 \
    --num_samples 100 \
    --output_file results/stage1_vqa_eval.json
```

Metrics: BLEU-1/2/3/4, BERTScore, Field Accuracy, METEOR, ROUGE-L, CIDEr

### VQA + Planning Evaluation

```bash
python scripts/evaluate_planning.py \
    --model_path save/stage3_planning/final_model \
    --data_root ./UNISCP \
    --sequences RURAL_A0 RURAL_A1 RURAL_A2 \
    --num_samples 100 \
    --output_file results/stage2_full_eval.json
```

Metrics: BLEU, BERTScore, Field Accuracy + ADE, FDE, per-step error, smoothness

## Monitoring

Training logs are saved to the output directory. Use TensorBoard:

```bash
tensorboard --logdir save/
```

Key metrics to watch:
- `train/loss` - Should decrease steadily
- `eval/bleu` - Should increase (Stage 1)
- `eval/ade` - Should decrease (Stage 3)
