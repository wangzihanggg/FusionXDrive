"""
Training script for Multi-Modal VLM on UNISCP dataset.

Follows wild-drive training strategy exactly:
  - Vision encoder: frozen
  - LLM: frozen (freeze_llm_model=True, like wild-drive default)
  - Q-Former + point cloud encoders: trainable
  - learning_rate=5e-6  (wild-drive: 5e-6, NOT 1e-4)
  - ddp_find_unused_parameters=True
  - max_grad_norm=1.0
  - label masking: pad_token_id (same as wild-drive)

Usage:
    torchrun --nproc_per_node=2 train.py \
        --data_root ./UNISCP \
        --output_dir save/multimodal_vlm_v2 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --num_epochs 10 \
        --num_query_tokens 64 \
        --qformer_layers 4 \
        --sequences RURAL_A0 RURAL_A1 RURAL_A2 RURAL_B0 RURAL_B1 RURAL_B2
"""

import os
import sys
import argparse
import logging

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import (
    AutoTokenizer, AutoImageProcessor,
    TrainingArguments, Trainer,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from fusionxdrive.model_vqa import MultiModalVLM, MultiModalVLMConfig
from fusionxdrive.dataset import UniscpDataset, UniscpDataCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Modal VLM (wild-drive strategy)")
    parser.add_argument("--data_root",         type=str, required=True)
    parser.add_argument("--output_dir",        type=str, default="save/multimodal_vlm_v2")
    parser.add_argument("--llm_model_path",    type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--vision_model_path", type=str, default="facebook/dinov3-base")
    # Training — following wild-drive hyperparams
    parser.add_argument("--batch_size",                  type=int,   default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=16)
    parser.add_argument("--learning_rate",               type=float, default=5e-6,
                        help="wild-drive uses 5e-6 (NOT 1e-4)")
    parser.add_argument("--num_epochs",                  type=int,   default=10)
    parser.add_argument("--warmup_steps",                type=int,   default=200)
    parser.add_argument("--save_steps",                  type=int,   default=500)
    parser.add_argument("--logging_steps",               type=int,   default=50)
    parser.add_argument("--num_workers",                 type=int,   default=4)
    parser.add_argument("--max_grad_norm",               type=float, default=1.0)
    parser.add_argument("--fp16",  action="store_true", default=True)
    parser.add_argument("--bf16",  action="store_true", default=False)
    # Data
    parser.add_argument("--max_lidar_points", type=int, default=40000)
    parser.add_argument("--max_radar_points", type=int, default=16000)
    parser.add_argument("--num_query_tokens", type=int, default=64)
    parser.add_argument("--qformer_layers",   type=int, default=4)
    parser.add_argument("--sequences",  nargs="+", default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main    = local_rank in (-1, 0)

    if is_main:
        logger.info("=" * 60)
        logger.info("Multi-Modal VLM Training  (full LLM fine-tuning)")
        logger.info("=" * 60)

    # ── config ───────────────────────────────────────────────────────────────
    config = MultiModalVLMConfig(
        llm_model_path=args.llm_model_path,
        vision_model_path=args.vision_model_path,
        freeze_vision_model=True,
        freeze_llm_model=False,     # False = full LLM fine-tuning
        qformer_dim=512,
        qformer_layers=args.qformer_layers,
        qformer_heads=8,
        num_query_tokens=args.num_query_tokens,
        lidar_output_dim=256,
        radar_output_dim=256,
        image_pad_num=args.num_query_tokens,
    )

    if is_main:
        logger.info(
            f"lr={args.learning_rate}  batch={args.batch_size}  "
            f"grad_acc={args.gradient_accumulation_steps}  "
            f"epochs={args.num_epochs}"
        )

    # ── model ────────────────────────────────────────────────────────────────
    model = MultiModalVLM(config)
    if torch.cuda.is_available():
        model = model.cuda()
        if is_main:
            logger.info(f"GPU: {torch.cuda.get_device_name()}")

    # ── tokenizer / processor ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

    if '<|image_pad|>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
        model.llm_model.resize_token_embeddings(len(tokenizer))
        model.tokenizer = tokenizer
        if is_main:
            logger.info(f"Added <|image_pad|>, vocab={len(tokenizer)}")

    # ── dataset ──────────────────────────────────────────────────────────────
    train_dataset = UniscpDataset(
        data_root=args.data_root,
        sequences=args.sequences,
        tokenizer=tokenizer,
        processor=processor,
        image_pad_num=config.image_pad_num,
        max_lidar_points=args.max_lidar_points,
        max_radar_points=args.max_radar_points,
        lidar_pc_range=config.lidar_pc_range,
        radar_pc_range=config.radar_pc_range,
    )
    if is_main:
        logger.info(f"Dataset: {len(train_dataset)} samples")
    if len(train_dataset) == 0:
        logger.error("No samples found!")
        return

    # ── training args — mirroring wild-drive exactly ─────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,          # 5e-6
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,          # 1.0  (wild-drive)
        fp16=args.fp16 and not args.bf16,
        bf16=args.bf16,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        logging_first_step=True,                   # wild-drive
        report_to='tensorboard',
        dataloader_pin_memory=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        lr_scheduler_type='cosine',
        ddp_find_unused_parameters=True,           # wild-drive: essential for frozen LLM + DDP
    )

    data_collator = UniscpDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if is_main:
        logger.info("Starting training ...")
    resume = args.resume_from if args.resume_from else False
    trainer.train(resume_from_checkpoint=resume)

    if is_main:
        final_dir = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_dir)
        trainer.save_state()
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved to {final_dir}")


if __name__ == '__main__':
    main()

