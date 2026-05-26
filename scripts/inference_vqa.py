"""
Inference script for Multi-Modal VLM.

Usage:
    python inference.py \
        --model_path save/multimodal_vlm/final_model \
        --data_root /path/to/UNISCP \
        --sequence RURAL_A0 \
        --frame_idx 000100
"""

import os
import sys
import json
import argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from fusionxdrive.model_vqa import MultiModalVLM, MultiModalVLMConfig
from fusionxdrive.dataset import (
    read_pcd, get_rural_calibration, load_timestamps, find_nearest_idx,
    SYSTEM_PROMPT, USER_PROMPT,
)
from pathlib import Path


def load_single_sample(data_root, sequence, frame_idx):
    """Load a single frame's data for inference."""
    data_root = Path(data_root)
    seq_dir = data_root / sequence
    
    calib = get_rural_calibration()
    
    # Find subdirs
    def find_subdir(prefix):
        top = seq_dir / prefix
        nested = top / prefix
        return nested if nested.exists() else top
    
    img_dir = find_subdir('1_IMAGE')
    lidar_dir = find_subdir('2_LIDAR')
    radar_dir = find_subdir('3_RADAR')
    
    # Load image
    img_path = img_dir / 'LEFT' / f'{frame_idx}.png'
    image = Image.open(str(img_path)).convert('RGB')
    
    # Load timestamps and find matching lidar/radar
    ts_img = load_timestamps(str(img_dir / 'timestamp_image_left.txt'))
    ts_lidar = load_timestamps(str(lidar_dir / 'timestamp_lidar.txt'))
    ts_radar = load_timestamps(str(radar_dir / 'timestamp_radar.txt'))
    
    # Find image timestamp
    img_ts = None
    for idx_str, ts in ts_img:
        if idx_str == frame_idx:
            img_ts = ts
            break
    
    if img_ts is None:
        raise ValueError(f"Frame {frame_idx} not found in timestamps")
    
    lidar_idx = find_nearest_idx(img_ts, ts_lidar)
    radar_idx = find_nearest_idx(img_ts, ts_radar)
    
    # Load LiDAR
    lidar_path = lidar_dir / 'PCD' / f'{lidar_idx}.pcd'
    lidar_data = read_pcd(str(lidar_path), ['x', 'y', 'z', 'intensity'])
    lidar_points = np.stack([
        lidar_data['x'], lidar_data['y'], lidar_data['z'],
        lidar_data.get('intensity', np.zeros_like(lidar_data['x']))
    ], axis=-1).astype(np.float32)
    valid = np.all(np.isfinite(lidar_points), axis=-1) & np.any(lidar_points[:, :3] != 0, axis=-1)
    lidar_points = lidar_points[valid]
    lidar_points = calib.crop_lidar_to_fov(lidar_points)
    
    # Load Radar
    radar_path = radar_dir / 'PCD' / f'{radar_idx}.pcd'
    radar_data = read_pcd(str(radar_path), ['x', 'y', 'z', 'doppler', 'power', 'recoveredSpeed'])
    radar_points = np.stack([
        radar_data['x'], radar_data['y'], radar_data['z'],
        radar_data.get('doppler', np.zeros_like(radar_data['x'])),
        radar_data.get('power', np.zeros_like(radar_data['x'])),
        radar_data.get('recoveredSpeed', np.zeros_like(radar_data['x']))
    ], axis=-1).astype(np.float32)
    valid = np.all(np.isfinite(radar_points), axis=-1)
    radar_points = radar_points[valid]
    radar_points = calib.crop_radar_to_fov(radar_points)
    
    # Load caption (ground truth)
    caption_path = seq_dir / '6_CAPTION' / f'{frame_idx}.json'
    gt_caption = None
    if caption_path.exists():
        with open(caption_path) as f:
            gt_caption = json.load(f)
    
    return image, lidar_points, radar_points, gt_caption


def inference(model, tokenizer, processor, image, lidar_points, radar_points, 
              num_query_tokens=64, device='cuda'):
    """Run inference on a single sample."""
    
    # Process image
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    
    # Build prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "<image>\n" + USER_PROMPT},
    ]
    q_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace('<image>', '<|image_pad|>' * num_query_tokens)
    
    prompt_ids = tokenizer(q_text, return_tensors="pt")["input_ids"].to(device)
    
    # Point clouds to tensors
    lidar_tensor = [torch.from_numpy(lidar_points).float().to(device)]
    radar_tensor = [torch.from_numpy(radar_points).float().to(device)]
    
    # Generate
    output_ids = model.generate(
        pixel_values=pixel_values,
        lidar_points=lidar_tensor,
        radar_points=radar_tensor,
        prompt_ids=prompt_ids,
        max_new_tokens=512,
        temperature=0.1,
    )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Try to parse JSON from generated text
    try:
        # Find JSON in the output
        start = generated_text.find('{')
        end = generated_text.rfind('}') + 1
        if start >= 0 and end > start:
            prediction = json.loads(generated_text[start:end])
        else:
            prediction = {"raw_output": generated_text}
    except json.JSONDecodeError:
        prediction = {"raw_output": generated_text}
    
    return prediction, generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sequence", type=str, default="RURAL_A0")
    parser.add_argument("--frame_idx", type=str, default="000100")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    config = MultiModalVLMConfig.from_pretrained(args.model_path)
    model = MultiModalVLM(config).to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)
    
    # Load data
    print(f"Loading frame {args.frame_idx} from {args.sequence}...")
    image, lidar_points, radar_points, gt_caption = load_single_sample(
        args.data_root, args.sequence, args.frame_idx
    )
    print(f"  Image: {image.size}")
    print(f"  LiDAR points: {lidar_points.shape}")
    print(f"  Radar points: {radar_points.shape}")
    
    # Inference
    print("Running inference...")
    prediction, raw_text = inference(
        model, tokenizer, processor,
        image, lidar_points, radar_points,
        num_query_tokens=config.num_query_tokens,
        device=device,
    )
    
    print("\n" + "=" * 60)
    print("PREDICTION:")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))
    
    if gt_caption:
        print("\nGROUND TRUTH:")
        print(json.dumps(gt_caption, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
