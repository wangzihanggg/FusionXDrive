"""
Dataset additions for planning support in UNISCP.

This file shows the KEY CHANGES needed in dataset.py to support
trajectory planning alongside VQA. It is NOT a full dataset file —
it shows only the new/modified parts.

Key changes:
  1. Add <|plan_pad|> special token
  2. Load waypoint JSON from 7_PLANNING/WAYPOINTS/
  3. Append plan_pad tokens AFTER the answer in input_ids
  4. Mask plan_pad tokens from LM loss (set labels = pad_token_id)
  5. Return gt_waypoints in the sample dict
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional


# Waypoint keys in order (matching the JSON format)
WAYPOINT_LABELS = ["t+1s", "t+2s", "t+5s", "t+10s"]


def load_waypoints(waypoint_path: str, num_waypoints: int = 4) -> np.ndarray:
    """
    Load waypoints from a JSON file.

    Expected JSON format (from 7_PLANNING/WAYPOINTS/):
    {
        "waypoints": [
            {"label": "t0", "x": 0.0, "y": 0.0, "z": 0.0},
            {"label": "t+1s", "x": 4.04, "y": 0.03, "z": 0.007, "available": true},
            {"label": "t+2s", "x": 8.60, "y": 0.00, "z": -0.014, "available": true},
            {"label": "t+5s", "x": 23.82, "y": 0.28, "z": -0.043, "available": true},
            {"label": "t+10s", "x": 55.63, "y": 0.38, "z": -0.211, "available": true}
        ],
        "coordinate_frame": "ego (x=forward, y=left, z=up)"
    }

    Returns:
        waypoints: [4, 3] numpy array (x, y, z) for t+1s, t+2s, t+5s, t+10s
    """
    waypoints = np.zeros((num_waypoints, 3), dtype=np.float32)

    try:
        with open(waypoint_path, 'r') as f:
            data = json.load(f)

        wp_list = data.get('waypoints', [])

        # Build a lookup by label
        wp_dict = {wp['label']: wp for wp in wp_list}

        for i, label in enumerate(WAYPOINT_LABELS[:num_waypoints]):
            if label in wp_dict:
                wp = wp_dict[label]
                if wp.get('available', True):
                    waypoints[i] = [wp['x'], wp['y'], wp['z']]

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        pass  # Return zeros if file missing or malformed

    return waypoints


# =============================================================================
# Modified __getitem__ — shows ONLY the changes needed
# =============================================================================

def build_text_with_planning(
    tokenizer,
    messages: list,
    answer_text: str,
    image_pad_num: int,
    num_planning_tokens: int,
    use_planning: bool = True,
) -> tuple:
    """
    Build input_ids and labels WITH planning token support.

    The sequence structure is:
        [system + user (with <|image_pad|>)] [answer + EOS] [<|plan_pad|> × N]

    Labels:
        [pad...] [answer + EOS] [pad...]
        ^question masked         ^plan tokens masked from LM loss

    This mirrors wild-drive's _build_text() exactly.
    """
    # Build question text with image padding
    q_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace('<image>', '<|image_pad|>' * image_pad_num)

    q_ids = tokenizer(q_text)['input_ids']
    a_text = answer_text + tokenizer.eos_token
    a_ids = tokenizer(a_text)['input_ids']

    # Planning tokens: appended AFTER the answer
    if use_planning and num_planning_tokens > 0:
        # We add (num_planning_tokens + 1) tokens because the causal shift
        # (input_ids = all_ids[:-1]) will remove the last one,
        # leaving exactly num_planning_tokens tokens in input_ids
        plan_text = '<|plan_pad|>' * (num_planning_tokens + 1)
        plan_ids = tokenizer(plan_text)['input_ids']
    else:
        plan_ids = []

    # Assemble full sequence
    all_ids = q_ids + a_ids + plan_ids
    all_labels = (
        [tokenizer.pad_token_id] * len(q_ids)    # mask question
        + a_ids                                     # supervise answer
        + [tokenizer.pad_token_id] * len(plan_ids)  # mask plan tokens
    )

    # Causal shift: predict next token
    input_ids = all_ids[:-1]
    labels = all_labels[1:]

    return input_ids, labels


# =============================================================================
# Modified __getitem__ example (pseudo-code showing the diff)
# =============================================================================

def getitem_with_planning(self, index):
    """
    Shows the modifications needed in UniscpDataset.__getitem__().

    ADDITIONS compared to the VQA-only version:
      1. Load waypoints from 7_PLANNING/WAYPOINTS/
      2. Use build_text_with_planning() instead of manual text building
      3. Return gt_waypoints in the output dict
    """
    sample = self.samples[index]

    # ── Original: load image, lidar, radar, caption (unchanged) ──
    # image = ...
    # lidar_points = ...
    # radar_points = ...
    # caption = ...
    # pixel_values = ...
    # answer_text = json.dumps(caption, ensure_ascii=False)

    # ── NEW: load waypoints ──
    # Waypoint path: same index as image, in 7_PLANNING/WAYPOINTS/ directory
    waypoint_path = sample.get('waypoint_path', None)
    if waypoint_path and Path(waypoint_path).exists():
        gt_waypoints = load_waypoints(waypoint_path, num_waypoints=4)
    else:
        gt_waypoints = np.zeros((4, 3), dtype=np.float32)

    # ── MODIFIED: build text with planning tokens ──
    from dataset import SYSTEM_PROMPT, USER_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "<image>\n" + USER_PROMPT},
    ]

    input_ids, labels = build_text_with_planning(
        tokenizer=self.tokenizer,
        messages=messages,
        answer_text="...",  # json.dumps(caption)
        image_pad_num=self.image_pad_num,
        num_planning_tokens=getattr(self, 'num_planning_tokens', 4),
        use_planning=True,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": None,  # pixel_values
        "lidar_points": None,  # lidar_points
        "radar_points": None,  # radar_points
        "n_lidar": 0,
        "n_radar": 0,
        "gt_waypoints": gt_waypoints,  # [4, 3] numpy → NEW
    }


# =============================================================================
# Modified DataCollator
# =============================================================================

class UniscpDataCollatorWithPlanning:
    """
    Extended data collator that also handles gt_waypoints.

    Adds gt_waypoints stacking to the original UniscpDataCollator.
    """
    def __init__(self, tokenizer, use_planning=True):
        self.tokenizer = tokenizer
        self.use_planning = use_planning

    def __call__(self, features: List[Dict]) -> Dict:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        labels = []
        pixel_values = []
        lidar_points_list = []
        radar_points_list = []
        gt_waypoints_list = []

        for f in features:
            # Pad text
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(f["labels"] + [self.tokenizer.pad_token_id] * pad_len)

            pixel_values.append(f["pixel_values"])

            # Point clouds
            lidar_pts = torch.from_numpy(f["lidar_points"]).float()
            radar_pts = torch.from_numpy(f["radar_points"]).float()
            n_l = f["n_lidar"]
            n_r = f["n_radar"]
            lidar_points_list.append(lidar_pts[:n_l] if n_l > 0 else lidar_pts[:1] * 0)
            radar_points_list.append(radar_pts[:n_r] if n_r > 0 else radar_pts[:1] * 0)

            # Waypoints (NEW)
            if self.use_planning and "gt_waypoints" in f:
                gt_waypoints_list.append(
                    torch.from_numpy(f["gt_waypoints"]).float()
                )

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values, dim=0),
            "lidar_points": lidar_points_list,
            "radar_points": radar_points_list,
        }

        if self.use_planning and gt_waypoints_list:
            result["gt_waypoints"] = torch.stack(gt_waypoints_list, dim=0)

        return result


# =============================================================================
# Sample discovery: how to find waypoint files
# =============================================================================

def build_sample_list_with_planning(data_root, sequences):
    """
    Shows how to extend the sample list to include waypoint paths.

    Directory structure expected:
        UNISCP/
        ├── RURAL_A0/
        │   ├── 1_CAMERA_LEFT/
        │   │   ├── 000000.png
        │   │   └── ...
        │   ├── 3_LIDAR/
        │   ├── 4_RADAR/
        │   ├── 6_CAPTION/
        │   └── 7_PLANNING/
        │       └── WAYPOINTS/
        │           ├── 000000.json
        │           ├── 000001.json
        │           └── ...
    """
    data_root = Path(data_root)
    samples = []

    for seq_name in sequences:
        seq_dir = data_root / seq_name
        waypoint_dir = seq_dir / '7_PLANNING' / 'WAYPOINTS'

        # ... (existing code to find image, lidar, radar, caption) ...

        # For each frame, add waypoint path
        # img_idx_str = "000000"
        # sample['waypoint_path'] = str(waypoint_dir / f"{img_idx_str}.json")

    return samples


# =============================================================================
# Tokenizer setup: add <|plan_pad|> token
# =============================================================================

def setup_tokenizer_with_planning(tokenizer, model):
    """
    Add both <|image_pad|> and <|plan_pad|> special tokens.

    Call this ONCE before training.
    """
    special_tokens = {'additional_special_tokens': ['<|image_pad|>', '<|plan_pad|>']}
    num_added = tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.llm_model.resize_token_embeddings(len(tokenizer))
        model.tokenizer = tokenizer
        print(f"Added {num_added} special tokens, vocab size = {len(tokenizer)}")
    return tokenizer
