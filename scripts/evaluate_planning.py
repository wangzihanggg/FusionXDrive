# """
# Evaluation script for Multi-Modal VLM + Trajectory Planning on UNISCP.

# Evaluates:
#   1. Scene Captioning (VQA):
#      - BLEU-1/2/3/4 (corpus-level)
#      - BERTScore P/R/F1
#      - Field Accuracy (per-field exact match)

#   2. Trajectory Planning:
#      - ADE (Average Displacement Error) — mean over 4 waypoints
#      - FDE (Final Displacement Error) — error at t+10s
#      - Per-step ADE at t+1s, t+2s, t+5s, t+10s
#      - minADE (best-case ADE)
#      - Longitudinal / Lateral error breakdown
#      - Anchor selection accuracy

# Usage:
#     python evaluate_with_planning.py \
#         --model_path save/vlm_diffplan/final_model \
#         --data_root ./UNISCP \
#         --sequences RURAL_A0 RURAL_A1 RURAL_A2 \
#         --num_samples 100 \
#         --output_file results/eval_planning.json \
#         --batch_size 4

#     # Skip BERTScore for faster evaluation:
#     python evaluate_with_planning.py \
#         --model_path save/vlm_diffplan/final_model \
#         --data_root ./UNISCP \
#         --no_bert \
#         --output_file results/eval_fast.json
# """

# import os
# import sys
# import json
# import argparse
# import logging
# from pathlib import Path
# from tqdm import tqdm

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import torch
# import numpy as np
# from PIL import Image
# from transformers import AutoTokenizer, AutoImageProcessor

# import math
# from collections import Counter

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# # Import from the planning-enabled model
# from model_with_planning import MultiModalVLM, MultiModalVLMConfig
# from dataset import (
#     UniscpDataset, read_pcd, get_rural_calibration,
#     load_timestamps, find_nearest_idx,
#     SYSTEM_PROMPT, USER_PROMPT,
# )
# from dataset_planning_additions import load_waypoints as _load_waypoints_orig

# def load_waypoints(path, num_waypoints=None):
#     """Load dense 8-point waypoints from JSON."""
#     LABELS = ["t+0.5s","t+1.0s","t+1.5s","t+2.0s","t+2.5s","t+3.0s","t+3.5s","t+4.0s"]
#     n = num_waypoints or 8
#     wp = np.zeros((n, 3), dtype=np.float32)
#     with open(path) as f:
#         data = json.load(f)
#     d = {w['label']: w for w in data.get('waypoints', [])}
#     for i, lab in enumerate(LABELS[:n]):
#         if lab in d and d[lab].get('available', True):
#             wp[i] = [d[lab]['x'], d[lab]['y'], d[lab].get('z', 0.0)]
#     return wp

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# logger = logging.getLogger(__name__)

# RURAL_SEQUENCES = ['RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B0', 'RURAL_B1', 'RURAL_B2']
# OTHER_SEQUENCES = ['FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO', 'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE']
# ALL_SEQUENCES   = RURAL_SEQUENCES + OTHER_SEQUENCES

# WAYPOINT_KEYS = ["t+0.5s","t+1.0s","t+1.5s","t+2.0s","t+2.5s","t+3.0s","t+3.5s","t+4.0s"]
# WAYPOINT_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]


# # =============================================================================
# # Checkpoint loading
# # =============================================================================

# def load_checkpoint_weights(model, model_path: str):
#     """Load trained checkpoint weights into the model."""
#     ckpt_path = Path(model_path)
#     weight_files = sorted(ckpt_path.glob("*.safetensors")) + sorted(ckpt_path.glob("*.bin"))

#     if not weight_files:
#         logger.warning(f"No weight files found in {ckpt_path}.")
#         return

#     state_dict = {}
#     for wf in weight_files:
#         logger.info(f"  Loading weights from {wf.name} ...")
#         if wf.suffix == ".safetensors":
#             from safetensors.torch import load_file
#             state_dict.update(load_file(str(wf)))
#         elif wf.suffix == ".bin":
#             loaded = torch.load(str(wf), map_location="cpu", weights_only=False)
#             if isinstance(loaded, dict):
#                 state_dict.update(loaded)

#     if not state_dict:
#         logger.warning(f"No valid state_dict loaded from {ckpt_path}.")
#         return

#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     logger.info(f"Checkpoint loaded from {ckpt_path}")
#     logger.info(f"  Total keys in checkpoint : {len(state_dict)}")
#     logger.info(f"  Missing keys : {len(missing)}")
#     logger.info(f"  Unexpected keys : {len(unexpected)}")
#     if missing:
#         logger.info(f"  First missing: {missing[:10]}")
#     if unexpected:
#         logger.info(f"  First unexpected: {unexpected[:10]}")


# # =============================================================================
# # Data loading
# # =============================================================================

# def load_sample_from_record(record: dict, calib, max_lidar=40000, max_radar=16000):
#     """Load image, lidar, radar, GT caption, and GT waypoints from a sample record."""
#     image = Image.open(record['img_path']).convert('RGB')

#     # LiDAR
#     data = read_pcd(record['lidar_path'], ['x', 'y', 'z', 'intensity'])
#     x, y, z = data.get('x', np.zeros(0)), data.get('y', np.zeros(0)), data.get('z', np.zeros(0))
#     intensity = data.get('intensity', np.zeros_like(x))
#     lidar = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
#     valid = np.all(np.isfinite(lidar), axis=-1) & np.any(lidar[:, :3] != 0, axis=-1)
#     lidar = lidar[valid]
#     lidar = calib.crop_lidar_to_fov(lidar)
#     if len(lidar) > max_lidar:
#         idx = np.random.choice(len(lidar), max_lidar, replace=False)
#         lidar = lidar[idx]

#     # Radar
#     data = read_pcd(record['radar_path'], ['x', 'y', 'z', 'doppler', 'power', 'recoveredSpeed'])
#     x, y, z = data.get('x', np.zeros(0)), data.get('y', np.zeros(0)), data.get('z', np.zeros(0))
#     doppler = data.get('doppler', np.zeros_like(x))
#     power   = data.get('power',   np.zeros_like(x))
#     speed   = data.get('recoveredSpeed', np.zeros_like(x))
#     radar = np.stack([x, y, z, doppler, power, speed], axis=-1).astype(np.float32)
#     valid = np.all(np.isfinite(radar), axis=-1)
#     radar = radar[valid]
#     radar = calib.crop_radar_to_fov(radar)
#     if len(radar) > max_radar:
#         idx = np.random.choice(len(radar), max_radar, replace=False)
#         radar = radar[idx]

#     # GT caption
#     with open(record['caption_path'], 'r') as f:
#         gt_caption = json.load(f)

#     # GT waypoints
#     gt_waypoints = None
#     waypoint_path = record.get('waypoint_path', None)
#     if waypoint_path and Path(waypoint_path).exists():
#         gt_waypoints = load_waypoints(waypoint_path)

#     return image, lidar, radar, gt_caption, gt_waypoints


# # =============================================================================
# # Inference
# # =============================================================================

# def build_prompt(tokenizer, num_query_tokens):
#     """Build the chat prompt string (same for every sample)."""
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user",   "content": "<image>\n" + USER_PROMPT},
#     ]
#     return tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     ).replace('<image>', '<|image_pad|>' * num_query_tokens)


# def parse_prediction(generated_text: str):
#     """Extract JSON dict from generated text."""
#     try:
#         start = generated_text.find('{')
#         end   = generated_text.rfind('}') + 1
#         if start >= 0 and end > start:
#             return json.loads(generated_text[start:end]), generated_text
#     except json.JSONDecodeError:
#         pass
#     return {}, generated_text


# def run_inference_batch(model, tokenizer, processor, images, lidar_list, radar_list,
#                         num_query_tokens=64, device='cuda', max_new_tokens=512,
#                         use_planning=True):
#     """
#     Run inference on a batch of samples.
#     Returns VQA predictions and (optionally) planning predictions.
#     """
#     prompt_str = build_prompt(tokenizer, num_query_tokens)

#     # Image
#     pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

#     # Text
#     tokenizer.padding_side = 'left'
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     B = len(images)
#     encoded = tokenizer([prompt_str] * B, return_tensors="pt", padding=True)
#     prompt_ids = encoded["input_ids"].to(device)

#     # Point clouds
#     lidar_tensors = [torch.from_numpy(p).float().to(device) for p in lidar_list]
#     radar_tensors = [torch.from_numpy(p).float().to(device) for p in radar_list]

#     with torch.no_grad():
#         if use_planning and hasattr(model, 'planner'):
#             text_outputs, planner_out = model.generate(
#                 pixel_values=pixel_values,
#                 lidar_points=lidar_tensors,
#                 radar_points=radar_tensors,
#                 prompt_ids=prompt_ids,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.1,
#                 return_planning=True,
#             )
#             pred_waypoints = planner_out['pred_waypoints'].cpu().numpy()  # [B, K, 3] K=8 or 4
#             all_waypoints = planner_out.get('all_waypoints', None)
#             if all_waypoints is not None:
#                 all_waypoints = all_waypoints.cpu().numpy()
#             anchor_scores = planner_out.get('anchor_scores', None)
#             if anchor_scores is not None:
#                 anchor_scores = anchor_scores.cpu().numpy()
#         else:
#             text_outputs = model.generate(
#                 pixel_values=pixel_values,
#                 lidar_points=lidar_tensors,
#                 radar_points=radar_tensors,
#                 prompt_ids=prompt_ids,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.1,
#             )
#             pred_waypoints = None
#             all_waypoints = None
#             anchor_scores = None

#     # Decode text
#     predictions, raw_texts = [], []
#     for i in range(B):
#         text = tokenizer.decode(text_outputs[i], skip_special_tokens=True)
#         pred, raw = parse_prediction(text)
#         predictions.append(pred)
#         raw_texts.append(raw)

#     return predictions, raw_texts, pred_waypoints, all_waypoints, anchor_scores


# # =============================================================================
# # VQA Metrics (same as evaluate_fixed.py)
# # =============================================================================

# def flatten_json_to_text(obj) -> str:
#     if isinstance(obj, dict):
#         return ' '.join(flatten_json_to_text(v) for v in obj.values())
#     elif isinstance(obj, list):
#         return ' '.join(flatten_json_to_text(i) for i in obj)
#     else:
#         return str(obj)


# def _ngrams(tokens, n):
#     return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# def _modified_precision(references, hypotheses, n):
#     clipped_count = 0
#     total_count = 0
#     for refs, hyp in zip(references, hypotheses):
#         hyp_ngrams = Counter(_ngrams(hyp, n))
#         max_ref_counts = Counter()
#         for ref in refs:
#             ref_ngrams = Counter(_ngrams(ref, n))
#             for ng, count in ref_ngrams.items():
#                 max_ref_counts[ng] = max(max_ref_counts[ng], count)
#         for ng, count in hyp_ngrams.items():
#             clipped_count += min(count, max_ref_counts.get(ng, 0))
#             total_count += count
#     if total_count == 0:
#         return 0.0
#     return clipped_count / total_count


# def compute_bleu_scores(references, hypotheses):
#     refs_tok = [[ref.lower().split()] for ref in references]
#     hyps_tok = [hyp.lower().split() for hyp in hypotheses]

#     hyp_len = sum(len(h) for h in hyps_tok)
#     ref_len = 0
#     for refs, hyp in zip(refs_tok, hyps_tok):
#         closest = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
#         ref_len += closest

#     if hyp_len == 0:
#         bp = 0.0
#     elif hyp_len >= ref_len:
#         bp = 1.0
#     else:
#         bp = math.exp(1.0 - ref_len / hyp_len)

#     weights_map = {
#         'BLEU-1': [1.0],
#         'BLEU-2': [0.5, 0.5],
#         'BLEU-3': [1/3, 1/3, 1/3],
#         'BLEU-4': [0.25, 0.25, 0.25, 0.25],
#     }
#     results = {}
#     for name, weights in weights_map.items():
#         log_avg = 0.0
#         for i, w in enumerate(weights):
#             p = _modified_precision(refs_tok, hyps_tok, i + 1) + 1e-5
#             log_avg += w * math.log(p)
#         results[name] = round(bp * math.exp(log_avg), 4)
#     return results


# def compute_bert_scores(references, hypotheses, lang='en', device='cuda'):
#     from bert_score import score as bert_score_fn
#     P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, device=device, verbose=False)
#     return {
#         'BERTScore-P':  round(P.mean().item(), 4),
#         'BERTScore-R':  round(R.mean().item(), 4),
#         'BERTScore-F1': round(F1.mean().item(), 4),
#     }


# def compute_field_accuracy(gt_list, pred_list):
#     field_paths = [
#         ('weather',           'condition'),
#         ('weather',           'illumination'),
#         ('traffic_light',     'present'),
#         ('traffic_light',     'state'),
#         ('traffic_sign',      'present'),
#         ('traffic_sign',      'category'),
#         ('forward_drivability','status'),
#         ('lane_keeping',      'status'),
#         ('lane_keeping',      'deviation'),
#         ('driving_advice',    'action'),
#         ('hazard_region',     'present'),
#         ('hazard_region',     'type'),
#     ]
#     counts = {f'{a}/{b}': {'correct': 0, 'total': 0} for a, b in field_paths}
#     overall_correct = overall_total = 0

#     for gt, pred in zip(gt_list, pred_list):
#         for top, sub in field_paths:
#             gt_val   = gt.get(top, {}).get(sub, None)
#             pred_val = pred.get(top, {}).get(sub, None)
#             if gt_val is not None:
#                 counts[f'{top}/{sub}']['total'] += 1
#                 overall_total += 1
#                 if str(gt_val).lower() == str(pred_val).lower():
#                     counts[f'{top}/{sub}']['correct'] += 1
#                     overall_correct += 1

#     result = {}
#     for key, v in counts.items():
#         if v['total'] > 0:
#             result[key] = round(v['correct'] / v['total'], 4)
#     result['overall_field_acc'] = round(overall_correct / overall_total, 4) if overall_total else 0.0
#     return result


# # =============================================================================
# # Planning Metrics
# # =============================================================================

# def compute_planning_metrics(all_pred, all_gt):
#     """
#     Compute planning metrics. Auto-adapts to 4-point or 8-point trajectories.
#     Handles mixed-length inputs by truncating to the minimum common length.
#     """
#     if not all_pred or not all_gt:
#         return {}

#     # Stack — all must be [N, 8, 3]
#     all_pred = np.array([np.array(p)[:8] for p in all_pred])  # [N, 8, 3]
#     all_gt   = np.array([np.array(g)[:8] for g in all_gt])    # [N, 8, 3]
#     N = len(all_pred)
#     if N == 0:
#         return {}

#     wp_keys = WAYPOINT_KEYS
#     wp_weights = np.array(WAYPOINT_WEIGHTS)

#     # Per-waypoint L2 displacement on (x, y)
#     dx = all_pred[:, :, 0] - all_gt[:, :, 0]
#     dy = all_pred[:, :, 1] - all_gt[:, :, 1]
#     l2_disp = np.sqrt(dx**2 + dy**2 + 1e-6)   # [N, K]

#     # Weighted ADE
#     weighted_disp = l2_disp * wp_weights[None, :]
#     weighted_ade_per_sample = weighted_disp.mean(axis=1)

#     # Standard ADE / FDE
#     ade_per_sample = l2_disp.mean(axis=1)
#     fde_per_sample = l2_disp[:, -1]

#     # Smooth L1
#     diff_xyz = all_pred - all_gt
#     smooth_l1_per_sample = np.where(
#         np.abs(diff_xyz) < 1.0,
#         0.5 * diff_xyz**2,
#         np.abs(diff_xyz) - 0.5
#     ).mean(axis=(1, 2))

#     results = {}

#     # Loss-aligned metrics
#     results['smooth_l1 (≈rec_loss)'] = round(float(smooth_l1_per_sample.mean()), 4)
#     results['weighted_ADE (≈ade_loss)'] = round(float(weighted_ade_per_sample.mean()), 4)
#     results['estimated_total_loss'] = round(
#         float(smooth_l1_per_sample.mean()) + float(weighted_ade_per_sample.mean()), 4
#     )

#     # Standard metrics
#     results['ADE'] = round(float(ade_per_sample.mean()), 4)
#     results['FDE'] = round(float(fde_per_sample.mean()), 4)
#     results['minADE'] = round(float(ade_per_sample.min()), 4)
#     results['maxADE'] = round(float(ade_per_sample.max()), 4)
#     results['medianADE'] = round(float(np.median(ade_per_sample)), 4)

#     # Per-step
#     for i, key in enumerate(wp_keys):
#         results[f'L2_{key}'] = round(float(l2_disp[:, i].mean()), 4)
#         results[f'wL2_{key}'] = round(float(weighted_disp[:, i].mean()), 4)

#     # Longitudinal vs Lateral
#     long_err = np.abs(dx)
#     lat_err  = np.abs(dy)
#     results['avg_long_err'] = round(float(long_err.mean()), 4)
#     results['avg_lat_err']  = round(float(lat_err.mean()), 4)
#     for i, key in enumerate(wp_keys):
#         results[f'long_{key}'] = round(float(long_err[:, i].mean()), 4)
#         results[f'lat_{key}']  = round(float(lat_err[:, i].mean()), 4)

#     # Z-axis
#     if all_pred.shape[2] >= 3:
#         dz = all_pred[:, :, 2] - all_gt[:, :, 2]
#         results['avg_z_err'] = round(float(np.abs(dz).mean()), 4)

#     # Thresholds
#     for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
#         pct = float((ade_per_sample < t).mean() * 100)
#         results[f'ADE<{t}m'] = round(pct, 1)

#     results['num_samples'] = N
#     results['num_waypoints'] = 8
#     return results


# def compute_multimodal_diversity(all_waypoints_list):
#     """
#     Compute trajectory diversity metric (from DiffusionDriveV2).

#     When the planner generates multiple trajectories per sample
#     (one per anchor), this measures how diverse they are.

#     Args:
#         all_waypoints_list: list of [N_anchor, 4, 3] arrays

#     Returns:
#         dict with diversity score
#     """
#     if not all_waypoints_list:
#         return {}

#     all_divs = []
#     for trajs in all_waypoints_list:
#         # trajs: [N_anchor, 4, 3]
#         M = trajs.shape[0]
#         if M < 2:
#             continue

#         # Pairwise diversity per waypoint
#         div_per_wp = []
#         for n in range(trajs.shape[1]):
#             pts = trajs[:, n, :2]  # [M, 2] — only x, y
#             dists = []
#             for i in range(M):
#                 for j in range(i + 1, M):
#                     d = np.linalg.norm(pts[i] - pts[j])
#                     dists.append(d)
#             raw_div = np.mean(dists) if dists else 0.0

#             # Normalize by average trajectory scale
#             avg_scale = np.mean(np.linalg.norm(pts, axis=1)) + 1e-6
#             div_normalized = min(1.0, raw_div / avg_scale)
#             div_per_wp.append(div_normalized)

#         all_divs.append(np.mean(div_per_wp))

#     if not all_divs:
#         return {}

#     return {
#         'trajectory_diversity': round(float(np.mean(all_divs) * 100), 2),
#         'diversity_std':        round(float(np.std(all_divs) * 100), 2),
#     }


# # =============================================================================
# # Main
# # =============================================================================

# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate VLM + Planning')
#     parser.add_argument('--model_path',   type=str, required=True)
#     parser.add_argument('--data_root',    type=str, required=True)
#     parser.add_argument('--sequences',    nargs='+', default=None)
#     parser.add_argument('--num_samples',  type=int, default=None)
#     parser.add_argument('--output_file',  type=str, default='eval_planning_results.json')
#     parser.add_argument('--device',       type=str, default='cuda')
#     parser.add_argument('--max_new_tokens', type=int, default=512)
#     parser.add_argument('--bert_lang',    type=str, default='en')
#     parser.add_argument('--no_bert',      action='store_true')
#     parser.add_argument('--no_planning',  action='store_true',
#                         help='Skip planning evaluation (VQA only)')
#     parser.add_argument('--predictions_cache', type=str, default=None)
#     parser.add_argument('--batch_size',   type=int, default=1)
#     return parser.parse_args()


# def main():
#     args   = parse_args()
#     device = args.device if torch.cuda.is_available() else 'cpu'
#     use_planning = not args.no_planning

#     sequences = args.sequences if args.sequences else ALL_SEQUENCES
#     logger.info(f"Evaluating sequences: {sequences}")
#     logger.info(f"Planning evaluation: {'ON' if use_planning else 'OFF'}")

#     # ── Model ──
#     # IMPORTANT: Load tokenizer FIRST, resize embeddings to match checkpoint,
#     # THEN load checkpoint weights.  This avoids size mismatch errors when
#     # the checkpoint was saved with a different vocab size (e.g. after adding
#     # <|image_pad|> during VQA training).
#     logger.info(f"Loading model from {args.model_path} ...")
#     config = MultiModalVLMConfig.from_pretrained(args.model_path)
#     model  = MultiModalVLM(config)

#     # Step 1: Load the saved tokenizer (which already has <|image_pad|>)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

#     # Step 2: Add any missing special tokens (<|plan_pad|> if not in saved tokenizer)
#     special_tokens_to_add = []
#     if '<|image_pad|>' not in tokenizer.get_vocab():
#         special_tokens_to_add.append('<|image_pad|>')
#     if '<|plan_pad|>' not in tokenizer.get_vocab():
#         special_tokens_to_add.append('<|plan_pad|>')
#     if special_tokens_to_add:
#         tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

#     # Step 3: Resize embeddings to match tokenizer vocab size BEFORE loading weights
#     model.llm_model.resize_token_embeddings(len(tokenizer))
#     model.tokenizer = tokenizer
#     logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

#     # Step 4: NOW load checkpoint weights (embedding sizes will match)
#     load_checkpoint_weights(model, args.model_path)

#     model = model.to(device)
#     model.eval()

#     # ── Load from cache if available ──
#     if args.predictions_cache and Path(args.predictions_cache).exists():
#         logger.info(f"Loading cache from {args.predictions_cache} ...")
#         with open(args.predictions_cache, 'r', encoding='utf-8') as f:
#             cache = json.load(f)
#         _run_all_metrics(
#             args, cache['all_gt_texts'], cache['all_pred_texts'],
#             cache['all_gt_dicts'], cache['all_pred_dicts'],
#             cache.get('all_pred_waypoints'), cache.get('all_gt_waypoints'),
#             cache.get('all_multi_waypoints'),
#             cache['per_sample_results'], sequences, device, use_planning,
#         )
#         return

#     # ── Build sample list ──
#     calib   = get_rural_calibration()
#     dataset = UniscpDataset(
#         data_root=args.data_root,
#         sequences=sequences,
#         tokenizer=tokenizer,
#         processor=processor,
#     )
#     samples = dataset.samples

#     # Attach waypoint paths to samples (original dataset doesn't index them)
#     data_root = Path(args.data_root)
#     n_wp_found = 0
#     for s in samples:
#         seq_dir = data_root / s['seq']
#         wp_path = seq_dir / '7_PLANNING' / 'WAYPOINTS' / f"{s['img_idx']}.json"
#         if wp_path.exists():
#             s['waypoint_path'] = str(wp_path)
#             n_wp_found += 1
#         else:
#             s['waypoint_path'] = None
#     logger.info(f"Waypoint files found: {n_wp_found}/{len(samples)}")

#     if args.num_samples:
#         per_seq   = args.num_samples
#         seq_groups = {}
#         for s in samples:
#             seq_groups.setdefault(s['seq'], []).append(s)
#         samples = []
#         for seq in sequences:
#             group = seq_groups.get(seq, [])
#             step  = max(1, len(group) // per_seq)
#             samples += group[::step][:per_seq]

#     logger.info(f"Total samples to evaluate: {len(samples)}")

#     # ── Inference loop ──
#     all_gt_texts, all_pred_texts     = [], []
#     all_gt_dicts, all_pred_dicts     = [], []
#     all_gt_waypoints, all_pred_waypoints = [], []
#     all_multi_waypoints              = []
#     per_sample_results               = []

#     batch_size = args.batch_size
#     for batch_start in tqdm(range(0, len(samples), batch_size), desc='Inference'):
#         batch_records = samples[batch_start: batch_start + batch_size]

#         batch_images, batch_lidar, batch_radar = [], [], []
#         batch_gt_captions, batch_gt_wp = [], []
#         valid_records = []

#         for record in batch_records:
#             try:
#                 image, lidar_pts, radar_pts, gt_caption, gt_wp = \
#                     load_sample_from_record(record, calib)
#                 batch_images.append(image)
#                 batch_lidar.append(lidar_pts)
#                 batch_radar.append(radar_pts)
#                 batch_gt_captions.append(gt_caption)
#                 batch_gt_wp.append(gt_wp)
#                 valid_records.append(record)
#             except Exception as e:
#                 logger.warning(f"Load failed [{record['seq']} {record['img_idx']}]: {e}")

#         if not batch_images:
#             continue

#         pred_dicts, raw_texts, pred_wp_batch, multi_wp_batch, anchor_scores = \
#             run_inference_batch(
#                 model, tokenizer, processor,
#                 batch_images, batch_lidar, batch_radar,
#                 num_query_tokens=config.num_query_tokens,
#                 device=device,
#                 max_new_tokens=args.max_new_tokens,
#                 use_planning=use_planning,
#             )

#         for idx, (record, gt_caption, pred_dict, raw_text, gt_wp) in enumerate(
#                 zip(valid_records, batch_gt_captions, pred_dicts, raw_texts, batch_gt_wp)):

#             gt_text   = flatten_json_to_text(gt_caption)
#             pred_text = flatten_json_to_text(pred_dict) if pred_dict else raw_text

#             all_gt_texts.append(gt_text)
#             all_pred_texts.append(pred_text)
#             all_gt_dicts.append(gt_caption)
#             all_pred_dicts.append(pred_dict)

#             sample_result = {
#                 'seq':        record['seq'],
#                 'img_idx':    record['img_idx'],
#                 'gt':         gt_caption,
#                 'pred':       pred_dict,
#                 'raw_output': raw_text,
#             }

#             # Planning results (8 dense waypoints)
#             if use_planning and pred_wp_batch is not None and gt_wp is not None:
#                 pred_wp = pred_wp_batch[idx][:8]  # [8, 3]
#                 gt_wp_8 = gt_wp[:8]               # [8, 3]

#                 all_pred_waypoints.append(pred_wp.tolist())
#                 all_gt_waypoints.append(gt_wp_8.tolist())
#                 sample_result['pred_waypoints'] = pred_wp.tolist()
#                 sample_result['gt_waypoints']   = gt_wp_8.tolist()

#                 # Per-sample displacement error
#                 disp = np.sqrt(
#                     (pred_wp[:, 0] - gt_wp_8[:, 0])**2 +
#                     (pred_wp[:, 1] - gt_wp_8[:, 1])**2
#                 )
#                 sample_result['per_step_error'] = {
#                     k: round(float(disp[i]), 4) for i, k in enumerate(WAYPOINT_KEYS)
#                 }
#                 sample_result['sample_ade'] = round(float(disp.mean()), 4)
#                 sample_result['sample_fde'] = round(float(disp[-1]), 4)

#                 if multi_wp_batch is not None:
#                     all_multi_waypoints.append(multi_wp_batch[idx].tolist())

#             per_sample_results.append(sample_result)

#     # ── Save cache ──
#     if args.predictions_cache:
#         cache_path = Path(args.predictions_cache)
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#         cache_data = {
#             'all_gt_texts':       all_gt_texts,
#             'all_pred_texts':     all_pred_texts,
#             'all_gt_dicts':       all_gt_dicts,
#             'all_pred_dicts':     all_pred_dicts,
#             'all_pred_waypoints': all_pred_waypoints if all_pred_waypoints else None,
#             'all_gt_waypoints':   all_gt_waypoints if all_gt_waypoints else None,
#             'all_multi_waypoints': all_multi_waypoints if all_multi_waypoints else None,
#             'per_sample_results': per_sample_results,
#         }
#         with open(cache_path, 'w', encoding='utf-8') as f:
#             json.dump(cache_data, f, ensure_ascii=False)
#         logger.info(f"Cache saved to {cache_path}")

#     if not all_gt_texts:
#         logger.error("No samples evaluated.")
#         return

#     _run_all_metrics(
#         args, all_gt_texts, all_pred_texts,
#         all_gt_dicts, all_pred_dicts,
#         all_pred_waypoints, all_gt_waypoints,
#         all_multi_waypoints,
#         per_sample_results, sequences, device, use_planning,
#     )


# def _run_all_metrics(args, all_gt_texts, all_pred_texts,
#                      all_gt_dicts, all_pred_dicts,
#                      all_pred_waypoints, all_gt_waypoints,
#                      all_multi_waypoints,
#                      per_sample_results, sequences, device, use_planning):
#     """Compute and print all metrics."""

#     # ══════════════════════════════════════════════════════════
#     # VQA Metrics
#     # ══════════════════════════════════════════════════════════
#     logger.info("Computing BLEU scores ...")
#     bleu = compute_bleu_scores(all_gt_texts, all_pred_texts)

#     bert = {}
#     if not args.no_bert:
#         logger.info("Computing BERTScore ...")
#         bert = compute_bert_scores(all_gt_texts, all_pred_texts,
#                                    lang=args.bert_lang, device=device)

#     logger.info("Computing field accuracy ...")
#     field_acc = compute_field_accuracy(all_gt_dicts, all_pred_dicts)

#     # Per-sequence VQA breakdown
#     seq_vqa_metrics = {}
#     for seq in sequences:
#         idxs = [i for i, r in enumerate(per_sample_results) if r['seq'] == seq]
#         if not idxs:
#             continue
#         seq_gt   = [all_gt_texts[i]   for i in idxs]
#         seq_pred = [all_pred_texts[i] for i in idxs]
#         seq_vqa_metrics[seq] = compute_bleu_scores(seq_gt, seq_pred)
#         seq_vqa_metrics[seq]['n_samples'] = len(idxs)

#     # ══════════════════════════════════════════════════════════
#     # Planning Metrics
#     # ══════════════════════════════════════════════════════════
#     plan_metrics = {}
#     seq_plan_metrics = {}
#     diversity_metrics = {}

#     if use_planning and all_pred_waypoints and all_gt_waypoints:
#         logger.info("Computing planning metrics ...")
#         pred_wp_np = [np.array(w) for w in all_pred_waypoints]
#         gt_wp_np   = [np.array(w) for w in all_gt_waypoints]
#         plan_metrics = compute_planning_metrics(pred_wp_np, gt_wp_np)

#         # Per-sequence planning breakdown
#         for seq in sequences:
#             idxs = [i for i, r in enumerate(per_sample_results)
#                     if r['seq'] == seq and 'pred_waypoints' in r]
#             if not idxs:
#                 continue
#             seq_pred_wp = [np.array(all_pred_waypoints[i]) for i in idxs]
#             seq_gt_wp   = [np.array(all_gt_waypoints[i])   for i in idxs]
#             seq_plan_metrics[seq] = compute_planning_metrics(seq_pred_wp, seq_gt_wp)

#         # Diversity (multi-anchor)
#         if all_multi_waypoints:
#             multi_np = [np.array(w) for w in all_multi_waypoints]
#             diversity_metrics = compute_multimodal_diversity(multi_np)

#     # ══════════════════════════════════════════════════════════
#     # Print summary
#     # ══════════════════════════════════════════════════════════
#     print("\n" + "=" * 70)
#     print("EVALUATION SUMMARY")
#     print("=" * 70)
#     print(f"  Sequences : {sequences}")
#     print(f"  Samples   : {len(all_gt_texts)}")

#     # ── VQA ──
#     print()
#     print("══ SCENE CAPTIONING (VQA) ══════════════════════════════════")
#     print()
#     print("── BLEU Scores (corpus-level) ──")
#     for k, v in bleu.items():
#         print(f"  {k}: {v:.4f}")
#     if bert:
#         print()
#         print("── BERTScore ──")
#         for k, v in bert.items():
#             print(f"  {k}: {v:.4f}")
#     print()
#     print("── Field Accuracy (exact match) ──")
#     for k, v in field_acc.items():
#         print(f"  {k}: {v:.4f}")
#     print()
#     print("── Per-Sequence BLEU-4 ──")
#     for seq, m in seq_vqa_metrics.items():
#         print(f"  {seq:25s}: BLEU-4={m['BLEU-4']:.4f}  n={m['n_samples']}")

#     # ── Planning ──
#     if plan_metrics:
#         print()
#         print("══ TRAJECTORY PLANNING ═════════════════════════════════════")
#         print(f"  Samples: {plan_metrics.get('num_samples', 0)}")

#         print()
#         print("── Loss-Aligned Metrics (compare with training loss) ──")
#         print(f"  smooth_l1 (≈rec_loss):     {plan_metrics['smooth_l1 (≈rec_loss)']:.4f}")
#         print(f"  weighted_ADE (≈ade_loss):   {plan_metrics['weighted_ADE (≈ade_loss)']:.4f}")
#         print(f"  estimated_total (rec+ade):  {plan_metrics['estimated_total_loss']:.4f}")

#         print()
#         print("── Standard Metrics (meters) ──")
#         print(f"  ADE:       {plan_metrics['ADE']:.4f}")
#         print(f"  FDE:       {plan_metrics['FDE']:.4f}")
#         print(f"  medianADE: {plan_metrics['medianADE']:.4f}")
#         print(f"  minADE:    {plan_metrics['minADE']:.4f}")
#         print(f"  maxADE:    {plan_metrics['maxADE']:.4f}")

#         print()
#         print("── Per-Step L2 / Weighted L2 (8 dense waypoints, 0.5s-4.0s) ──")
#         print(f"  {'step':>8s}  {'L2(m)':>8s}  {'wL2':>8s}  {'weight':>6s}  {'long':>8s}  {'lat':>8s}")
#         for i, key in enumerate(WAYPOINT_KEYS):
#             w = WAYPOINT_WEIGHTS[i]
#             print(f"  {key:>8s}  {plan_metrics.get(f'L2_{key}', 0):>8.4f}  "
#                   f"{plan_metrics.get(f'wL2_{key}', 0):>8.4f}  {w:>6.1f}  "
#                   f"{plan_metrics.get(f'long_{key}', 0):>8.4f}  "
#                   f"{plan_metrics.get(f'lat_{key}', 0):>8.4f}")

#         if 'avg_z_err' in plan_metrics:
#             print(f"  z-axis avg: {plan_metrics['avg_z_err']:.4f}")

#         print()
#         print("── Accuracy at Thresholds ──")
#         for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
#             k = f'ADE<{t}m'
#             if k in plan_metrics:
#                 print(f"  {k}: {plan_metrics[k]:.1f}%")

#         if diversity_metrics:
#             print()
#             print("── Multi-Modal Diversity ──")
#             for k, v in diversity_metrics.items():
#                 print(f"  {k}: {v}")

#         if seq_plan_metrics:
#             print()
#             print("── Per-Sequence ──")
#             for seq, m in seq_plan_metrics.items():
#                 print(f"  {seq:25s}: ADE={m['ADE']:.4f}  FDE={m['FDE']:.4f}  "
#                       f"wADE={m['weighted_ADE (≈ade_loss)']:.4f}  n={m['num_samples']}")

#     print("=" * 70)

#     # ── Save results ──
#     output_path = Path(args.output_file)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     results = {
#         'sequences':     sequences,
#         'n_samples':     len(all_gt_texts),
#         'vqa': {
#             'bleu':          bleu,
#             'bert_score':    bert,
#             'field_accuracy': field_acc,
#             'per_sequence':  seq_vqa_metrics,
#         },
#         'planning': {
#             'metrics':       plan_metrics,
#             'diversity':     diversity_metrics,
#             'per_sequence':  seq_plan_metrics,
#         },
#         'per_sample': per_sample_results,
#     }
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     logger.info(f"Results saved to {output_path}")


# if __name__ == '__main__':
#     main()

"""
Evaluation script for Multi-Modal VLM + Trajectory Planning on UNISCP.

Evaluates:
  1. Scene Captioning (VQA):
     - BLEU-1/2/3/4 (corpus-level)
     - BERTScore P/R/F1
     - Field Accuracy (per-field exact match)

  2. Trajectory Planning:
     - ADE (Average Displacement Error) — mean over 4 waypoints
     - FDE (Final Displacement Error) — error at t+10s
     - Per-step ADE at t+1s, t+2s, t+5s, t+10s
     - minADE (best-case ADE)
     - Longitudinal / Lateral error breakdown
     - Anchor selection accuracy

Usage:
    python evaluate_with_planning.py \
        --model_path save/vlm_diffplan/final_model \
        --data_root ./UNISCP \
        --sequences RURAL_A0 RURAL_A1 RURAL_A2 \
        --num_samples 100 \
        --output_file results/eval_planning.json \
        --batch_size 4

    # Skip BERTScore for faster evaluation:
    python evaluate_with_planning.py \
        --model_path save/vlm_diffplan/final_model \
        --data_root ./UNISCP \
        --no_bert \
        --output_file results/eval_fast.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

import math
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Import from the planning-enabled model
from fusionxdrive.model_planning import MultiModalVLM, MultiModalVLMConfig
from fusionxdrive.dataset import (
    UniscpDataset, read_pcd, get_rural_calibration,
    load_timestamps, find_nearest_idx,
    SYSTEM_PROMPT, USER_PROMPT,
)
from fusionxdrive.dataset_planning import load_waypoints as _load_waypoints_orig

from fusionxdrive.metrics import (
    compute_meteor_scores, compute_rouge_scores, compute_cider_scores,
    compute_rouge_f_scores,
    compute_caption_plan_joint, compute_trajectory_safety, print_enhanced_summary,
)

def load_waypoints(path, num_waypoints=None):
    """Load dense 8-point waypoints from JSON."""
    LABELS = ["t+0.5s","t+1.0s","t+1.5s","t+2.0s","t+2.5s","t+3.0s","t+3.5s","t+4.0s"]
    n = num_waypoints or 8
    wp = np.zeros((n, 3), dtype=np.float32)
    with open(path) as f:
        data = json.load(f)
    d = {w['label']: w for w in data.get('waypoints', [])}
    for i, lab in enumerate(LABELS[:n]):
        if lab in d and d[lab].get('available', True):
            wp[i] = [d[lab]['x'], d[lab]['y'], d[lab].get('z', 0.0)]
    return wp

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RURAL_SEQUENCES = ['RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B0', 'RURAL_B1', 'RURAL_B2']
OTHER_SEQUENCES = ['FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO', 'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE']
ALL_SEQUENCES   = RURAL_SEQUENCES + OTHER_SEQUENCES

WAYPOINT_KEYS = ["t+0.5s","t+1.0s","t+1.5s","t+2.0s","t+2.5s","t+3.0s","t+3.5s","t+4.0s"]
WAYPOINT_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_checkpoint_weights(model, model_path: str):
    """Load trained checkpoint weights into the model."""
    ckpt_path = Path(model_path)
    weight_files = sorted(ckpt_path.glob("*.safetensors")) + sorted(ckpt_path.glob("*.bin"))

    if not weight_files:
        logger.warning(f"No weight files found in {ckpt_path}.")
        return

    state_dict = {}
    for wf in weight_files:
        logger.info(f"  Loading weights from {wf.name} ...")
        if wf.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict.update(load_file(str(wf)))
        elif wf.suffix == ".bin":
            loaded = torch.load(str(wf), map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                state_dict.update(loaded)

    if not state_dict:
        logger.warning(f"No valid state_dict loaded from {ckpt_path}.")
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Checkpoint loaded from {ckpt_path}")
    logger.info(f"  Total keys in checkpoint : {len(state_dict)}")
    logger.info(f"  Missing keys : {len(missing)}")
    logger.info(f"  Unexpected keys : {len(unexpected)}")
    if missing:
        logger.info(f"  First missing: {missing[:10]}")
    if unexpected:
        logger.info(f"  First unexpected: {unexpected[:10]}")


# =============================================================================
# Data loading
# =============================================================================

def load_sample_from_record(record: dict, calib, max_lidar=40000, max_radar=16000):
    """Load image, lidar, radar, GT caption, and GT waypoints from a sample record."""
    image = Image.open(record['img_path']).convert('RGB')

    # LiDAR
    data = read_pcd(record['lidar_path'], ['x', 'y', 'z', 'intensity'])
    x, y, z = data.get('x', np.zeros(0)), data.get('y', np.zeros(0)), data.get('z', np.zeros(0))
    intensity = data.get('intensity', np.zeros_like(x))
    lidar = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
    valid = np.all(np.isfinite(lidar), axis=-1) & np.any(lidar[:, :3] != 0, axis=-1)
    lidar = lidar[valid]
    lidar = calib.crop_lidar_to_fov(lidar)
    if len(lidar) > max_lidar:
        idx = np.random.choice(len(lidar), max_lidar, replace=False)
        lidar = lidar[idx]

    # Radar
    data = read_pcd(record['radar_path'], ['x', 'y', 'z', 'doppler', 'power', 'recoveredSpeed'])
    x, y, z = data.get('x', np.zeros(0)), data.get('y', np.zeros(0)), data.get('z', np.zeros(0))
    doppler = data.get('doppler', np.zeros_like(x))
    power   = data.get('power',   np.zeros_like(x))
    speed   = data.get('recoveredSpeed', np.zeros_like(x))
    radar = np.stack([x, y, z, doppler, power, speed], axis=-1).astype(np.float32)
    valid = np.all(np.isfinite(radar), axis=-1)
    radar = radar[valid]
    radar = calib.crop_radar_to_fov(radar)
    if len(radar) > max_radar:
        idx = np.random.choice(len(radar), max_radar, replace=False)
        radar = radar[idx]

    # GT caption
    with open(record['caption_path'], 'r') as f:
        gt_caption = json.load(f)

    # GT waypoints
    gt_waypoints = None
    waypoint_path = record.get('waypoint_path', None)
    if waypoint_path and Path(waypoint_path).exists():
        gt_waypoints = load_waypoints(waypoint_path)

    return image, lidar, radar, gt_caption, gt_waypoints


# =============================================================================
# Inference
# =============================================================================

def build_prompt(tokenizer, num_query_tokens):
    """Build the chat prompt string (same for every sample)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "<image>\n" + USER_PROMPT},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace('<image>', '<|image_pad|>' * num_query_tokens)


def parse_prediction(generated_text: str):
    """Extract JSON dict from generated text."""
    try:
        start = generated_text.find('{')
        end   = generated_text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(generated_text[start:end]), generated_text
    except json.JSONDecodeError:
        pass
    return {}, generated_text


def run_inference_batch(model, tokenizer, processor, images, lidar_list, radar_list,
                        num_query_tokens=64, device='cuda', max_new_tokens=512,
                        use_planning=True):
    """
    Run inference on a batch of samples.
    Returns VQA predictions and (optionally) planning predictions.
    """
    prompt_str = build_prompt(tokenizer, num_query_tokens)

    # Image
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

    # Text
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    B = len(images)
    encoded = tokenizer([prompt_str] * B, return_tensors="pt", padding=True)
    prompt_ids = encoded["input_ids"].to(device)

    # Point clouds
    lidar_tensors = [torch.from_numpy(p).float().to(device) for p in lidar_list]
    radar_tensors = [torch.from_numpy(p).float().to(device) for p in radar_list]

    with torch.no_grad():
        if use_planning and hasattr(model, 'planner'):
            text_outputs, planner_out = model.generate(
                pixel_values=pixel_values,
                lidar_points=lidar_tensors,
                radar_points=radar_tensors,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                return_planning=True,
            )
            pred_waypoints = planner_out['pred_waypoints'].cpu().numpy()  # [B, K, 3] K=8 or 4
            all_waypoints = planner_out.get('all_waypoints', None)
            if all_waypoints is not None:
                all_waypoints = all_waypoints.cpu().numpy()
            anchor_scores = planner_out.get('anchor_scores', None)
            if anchor_scores is not None:
                anchor_scores = anchor_scores.cpu().numpy()
        else:
            text_outputs = model.generate(
                pixel_values=pixel_values,
                lidar_points=lidar_tensors,
                radar_points=radar_tensors,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
            )
            pred_waypoints = None
            all_waypoints = None
            anchor_scores = None

    # Decode text
    predictions, raw_texts = [], []
    for i in range(B):
        text = tokenizer.decode(text_outputs[i], skip_special_tokens=True)
        pred, raw = parse_prediction(text)
        predictions.append(pred)
        raw_texts.append(raw)

    return predictions, raw_texts, pred_waypoints, all_waypoints, anchor_scores


# =============================================================================
# VQA Metrics (same as evaluate_fixed.py)
# =============================================================================

def flatten_json_to_text(obj) -> str:
    if isinstance(obj, dict):
        return ' '.join(flatten_json_to_text(v) for v in obj.values())
    elif isinstance(obj, list):
        return ' '.join(flatten_json_to_text(i) for i in obj)
    else:
        return str(obj)


def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def _modified_precision(references, hypotheses, n):
    clipped_count = 0
    total_count = 0
    for refs, hyp in zip(references, hypotheses):
        hyp_ngrams = Counter(_ngrams(hyp, n))
        max_ref_counts = Counter()
        for ref in refs:
            ref_ngrams = Counter(_ngrams(ref, n))
            for ng, count in ref_ngrams.items():
                max_ref_counts[ng] = max(max_ref_counts[ng], count)
        for ng, count in hyp_ngrams.items():
            clipped_count += min(count, max_ref_counts.get(ng, 0))
            total_count += count
    if total_count == 0:
        return 0.0
    return clipped_count / total_count


def compute_bleu_scores(references, hypotheses):
    refs_tok = [[ref.lower().split()] for ref in references]
    hyps_tok = [hyp.lower().split() for hyp in hypotheses]

    hyp_len = sum(len(h) for h in hyps_tok)
    ref_len = 0
    for refs, hyp in zip(refs_tok, hyps_tok):
        closest = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
        ref_len += closest

    if hyp_len == 0:
        bp = 0.0
    elif hyp_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len / hyp_len)

    weights_map = {
        'BLEU-1': [1.0],
        'BLEU-2': [0.5, 0.5],
        'BLEU-3': [1/3, 1/3, 1/3],
        'BLEU-4': [0.25, 0.25, 0.25, 0.25],
    }
    results = {}
    for name, weights in weights_map.items():
        log_avg = 0.0
        for i, w in enumerate(weights):
            p = _modified_precision(refs_tok, hyps_tok, i + 1) + 1e-5
            log_avg += w * math.log(p)
        results[name] = round(bp * math.exp(log_avg), 4)
    return results


def compute_bert_scores(references, hypotheses, lang='en', device='cuda'):
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, device=device, verbose=False)
    return {
        'BERTScore-P':  round(P.mean().item(), 4),
        'BERTScore-R':  round(R.mean().item(), 4),
        'BERTScore-F1': round(F1.mean().item(), 4),
    }


def compute_field_accuracy(gt_list, pred_list):
    field_paths = [
        ('weather',           'condition'),
        ('weather',           'illumination'),
        ('traffic_light',     'present'),
        ('traffic_light',     'state'),
        ('traffic_sign',      'present'),
        ('traffic_sign',      'category'),
        ('forward_drivability','status'),
        ('lane_keeping',      'status'),
        ('lane_keeping',      'deviation'),
        ('driving_advice',    'action'),
        ('hazard_region',     'present'),
        ('hazard_region',     'type'),
    ]
    counts = {f'{a}/{b}': {'correct': 0, 'total': 0} for a, b in field_paths}
    overall_correct = overall_total = 0

    for gt, pred in zip(gt_list, pred_list):
        if not isinstance(pred, dict):
            continue
        for top, sub in field_paths:
            gt_val   = gt.get(top, {}).get(sub, None)
            pred_val = pred.get(top, {}).get(sub, None)
            if gt_val is not None:
                counts[f'{top}/{sub}']['total'] += 1
                overall_total += 1
                if str(gt_val).lower() == str(pred_val).lower():
                    counts[f'{top}/{sub}']['correct'] += 1
                    overall_correct += 1

    result = {}
    for key, v in counts.items():
        if v['total'] > 0:
            result[key] = round(v['correct'] / v['total'], 4)
    result['overall_field_acc'] = round(overall_correct / overall_total, 4) if overall_total else 0.0
    return result


# =============================================================================
# Planning Metrics
# =============================================================================

def compute_planning_metrics(all_pred, all_gt):
    """
    Compute planning metrics. Auto-adapts to 4-point or 8-point trajectories.
    Handles mixed-length inputs by truncating to the minimum common length.
    """
    if not all_pred or not all_gt:
        return {}

    # Stack — all must be [N, 8, 3]
    all_pred = np.array([np.array(p)[:8] for p in all_pred])  # [N, 8, 3]
    all_gt   = np.array([np.array(g)[:8] for g in all_gt])    # [N, 8, 3]
    N = len(all_pred)
    if N == 0:
        return {}

    wp_keys = WAYPOINT_KEYS
    wp_weights = np.array(WAYPOINT_WEIGHTS)

    # Per-waypoint L2 displacement on (x, y)
    dx = all_pred[:, :, 0] - all_gt[:, :, 0]
    dy = all_pred[:, :, 1] - all_gt[:, :, 1]
    l2_disp = np.sqrt(dx**2 + dy**2 + 1e-6)   # [N, K]

    # Weighted ADE
    weighted_disp = l2_disp * wp_weights[None, :]
    weighted_ade_per_sample = weighted_disp.mean(axis=1)

    # Standard ADE / FDE
    ade_per_sample = l2_disp.mean(axis=1)
    fde_per_sample = l2_disp[:, -1]

    # Smooth L1
    diff_xyz = all_pred - all_gt
    smooth_l1_per_sample = np.where(
        np.abs(diff_xyz) < 1.0,
        0.5 * diff_xyz**2,
        np.abs(diff_xyz) - 0.5
    ).mean(axis=(1, 2))

    results = {}

    # Loss-aligned metrics
    results['smooth_l1 (≈rec_loss)'] = round(float(smooth_l1_per_sample.mean()), 4)
    results['weighted_ADE (≈ade_loss)'] = round(float(weighted_ade_per_sample.mean()), 4)
    results['estimated_total_loss'] = round(
        float(smooth_l1_per_sample.mean()) + float(weighted_ade_per_sample.mean()), 4
    )

    # Standard metrics
    results['ADE'] = round(float(ade_per_sample.mean()), 4)
    results['FDE'] = round(float(fde_per_sample.mean()), 4)
    results['minADE'] = round(float(ade_per_sample.min()), 4)
    results['maxADE'] = round(float(ade_per_sample.max()), 4)
    results['medianADE'] = round(float(np.median(ade_per_sample)), 4)

    # Per-step
    for i, key in enumerate(wp_keys):
        results[f'L2_{key}'] = round(float(l2_disp[:, i].mean()), 4)
        results[f'wL2_{key}'] = round(float(weighted_disp[:, i].mean()), 4)

    # Longitudinal vs Lateral
    long_err = np.abs(dx)
    lat_err  = np.abs(dy)
    results['avg_long_err'] = round(float(long_err.mean()), 4)
    results['avg_lat_err']  = round(float(lat_err.mean()), 4)
    for i, key in enumerate(wp_keys):
        results[f'long_{key}'] = round(float(long_err[:, i].mean()), 4)
        results[f'lat_{key}']  = round(float(lat_err[:, i].mean()), 4)

    # Z-axis
    if all_pred.shape[2] >= 3:
        dz = all_pred[:, :, 2] - all_gt[:, :, 2]
        results['avg_z_err'] = round(float(np.abs(dz).mean()), 4)

    # Thresholds
    for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
        pct = float((ade_per_sample < t).mean() * 100)
        results[f'ADE<{t}m'] = round(pct, 1)

    results['num_samples'] = N
    results['num_waypoints'] = 8
    return results


def compute_multimodal_diversity(all_waypoints_list):
    """
    Compute trajectory diversity metric (from DiffusionDriveV2).

    When the planner generates multiple trajectories per sample
    (one per anchor), this measures how diverse they are.

    Args:
        all_waypoints_list: list of [N_anchor, 4, 3] arrays

    Returns:
        dict with diversity score
    """
    if not all_waypoints_list:
        return {}

    all_divs = []
    for trajs in all_waypoints_list:
        # trajs: [N_anchor, 4, 3]
        M = trajs.shape[0]
        if M < 2:
            continue

        # Pairwise diversity per waypoint
        div_per_wp = []
        for n in range(trajs.shape[1]):
            pts = trajs[:, n, :2]  # [M, 2] — only x, y
            dists = []
            for i in range(M):
                for j in range(i + 1, M):
                    d = np.linalg.norm(pts[i] - pts[j])
                    dists.append(d)
            raw_div = np.mean(dists) if dists else 0.0

            # Normalize by average trajectory scale
            avg_scale = np.mean(np.linalg.norm(pts, axis=1)) + 1e-6
            div_normalized = min(1.0, raw_div / avg_scale)
            div_per_wp.append(div_normalized)

        all_divs.append(np.mean(div_per_wp))

    if not all_divs:
        return {}

    return {
        'trajectory_diversity': round(float(np.mean(all_divs) * 100), 2),
        'diversity_std':        round(float(np.std(all_divs) * 100), 2),
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VLM + Planning')
    parser.add_argument('--model_path',   type=str, required=True)
    parser.add_argument('--data_root',    type=str, required=True)
    parser.add_argument('--sequences',    nargs='+', default=None)
    parser.add_argument('--num_samples',  type=int, default=None)
    parser.add_argument('--start',        type=int, default=None,
                        help='Start frame index (e.g. 0)')
    parser.add_argument('--end',          type=int, default=None,
                        help='End frame index exclusive (e.g. 1000)')
    parser.add_argument('--step',         type=int, default=1,
                        help='Frame step (1=every frame)')
    parser.add_argument('--output_file',  type=str, default='eval_planning_results.json')
    parser.add_argument('--device',       type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--bert_lang',    type=str, default='en')
    parser.add_argument('--no_bert',      action='store_true')
    parser.add_argument('--no_planning',  action='store_true',
                        help='Skip planning evaluation (VQA only)')
    parser.add_argument('--predictions_cache', type=str, default=None)
    parser.add_argument('--batch_size',   type=int, default=1)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    use_planning = not args.no_planning

    sequences = args.sequences if args.sequences else ALL_SEQUENCES
    logger.info(f"Evaluating sequences: {sequences}")
    logger.info(f"Planning evaluation: {'ON' if use_planning else 'OFF'}")

    # ── Model ──
    # IMPORTANT: Load tokenizer FIRST, resize embeddings to match checkpoint,
    # THEN load checkpoint weights.  This avoids size mismatch errors when
    # the checkpoint was saved with a different vocab size (e.g. after adding
    # <|image_pad|> during VQA training).
    logger.info(f"Loading model from {args.model_path} ...")
    config = MultiModalVLMConfig.from_pretrained(args.model_path)
    model  = MultiModalVLM(config)

    # Step 1: Load the saved tokenizer (which already has <|image_pad|>)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except (ValueError, OSError):
        logger.info(f"  Tokenizer not in checkpoint, loading from {config.llm_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

    # Step 2: Add any missing special tokens (<|plan_pad|> if not in saved tokenizer)
    special_tokens_to_add = []
    if '<|image_pad|>' not in tokenizer.get_vocab():
        special_tokens_to_add.append('<|image_pad|>')
    if '<|plan_pad|>' not in tokenizer.get_vocab():
        special_tokens_to_add.append('<|plan_pad|>')
    if special_tokens_to_add:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})

    # Step 3: Resize embeddings to match tokenizer vocab size BEFORE loading weights
    model.llm_model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

    # Step 4: NOW load checkpoint weights (embedding sizes will match)
    load_checkpoint_weights(model, args.model_path)

    model = model.to(device)
    model.eval()

    # ── Load from cache if available ──
    if args.predictions_cache and Path(args.predictions_cache).exists():
        logger.info(f"Loading cache from {args.predictions_cache} ...")
        with open(args.predictions_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        _run_all_metrics(
            args, cache['all_gt_texts'], cache['all_pred_texts'],
            cache['all_gt_dicts'], cache['all_pred_dicts'],
            cache.get('all_pred_waypoints'), cache.get('all_gt_waypoints'),
            cache.get('all_multi_waypoints'),
            cache['per_sample_results'], sequences, device, use_planning,
        )
        return

    # ── Build sample list ──
    calib   = get_rural_calibration()
    dataset = UniscpDataset(
        data_root=args.data_root,
        sequences=sequences,
        tokenizer=tokenizer,
        processor=processor,
    )
    samples = dataset.samples

    # Attach waypoint paths to samples (original dataset doesn't index them)
    data_root = Path(args.data_root)
    n_wp_found = 0
    for s in samples:
        seq_dir = data_root / s['seq']
        wp_path = seq_dir / '7_PLANNING' / 'WAYPOINTS' / f"{s['img_idx']}.json"
        if wp_path.exists():
            s['waypoint_path'] = str(wp_path)
            n_wp_found += 1
        else:
            s['waypoint_path'] = None
    logger.info(f"Waypoint files found: {n_wp_found}/{len(samples)}")

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end or len(samples)
        samples = samples[start:end:args.step]
        logger.info(f"Frame range: [{start}:{end}:{args.step}] → {len(samples)} samples")
    elif args.num_samples:
        per_seq   = args.num_samples
        seq_groups = {}
        for s in samples:
            seq_groups.setdefault(s['seq'], []).append(s)
        samples = []
        for seq in sequences:
            group = seq_groups.get(seq, [])
            step  = max(1, len(group) // per_seq)
            samples += group[::step][:per_seq]

    logger.info(f"Total samples to evaluate: {len(samples)}")

    # ── Inference loop ──
    all_gt_texts, all_pred_texts     = [], []
    all_gt_dicts, all_pred_dicts     = [], []
    all_gt_waypoints, all_pred_waypoints = [], []
    all_multi_waypoints              = []
    per_sample_results               = []

    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(samples), batch_size), desc='Inference'):
        batch_records = samples[batch_start: batch_start + batch_size]

        batch_images, batch_lidar, batch_radar = [], [], []
        batch_gt_captions, batch_gt_wp = [], []
        valid_records = []

        for record in batch_records:
            try:
                image, lidar_pts, radar_pts, gt_caption, gt_wp = \
                    load_sample_from_record(record, calib)
                batch_images.append(image)
                batch_lidar.append(lidar_pts)
                batch_radar.append(radar_pts)
                batch_gt_captions.append(gt_caption)
                batch_gt_wp.append(gt_wp)
                valid_records.append(record)
            except Exception as e:
                logger.warning(f"Load failed [{record['seq']} {record['img_idx']}]: {e}")

        if not batch_images:
            continue

        pred_dicts, raw_texts, pred_wp_batch, multi_wp_batch, anchor_scores = \
            run_inference_batch(
                model, tokenizer, processor,
                batch_images, batch_lidar, batch_radar,
                num_query_tokens=config.num_query_tokens,
                device=device,
                max_new_tokens=args.max_new_tokens,
                use_planning=use_planning,
            )

        for idx, (record, gt_caption, pred_dict, raw_text, gt_wp) in enumerate(
                zip(valid_records, batch_gt_captions, pred_dicts, raw_texts, batch_gt_wp)):

            gt_text   = flatten_json_to_text(gt_caption)
            pred_text = flatten_json_to_text(pred_dict) if pred_dict else raw_text

            all_gt_texts.append(gt_text)
            all_pred_texts.append(pred_text)
            all_gt_dicts.append(gt_caption)
            all_pred_dicts.append(pred_dict)

            sample_result = {
                'seq':        record['seq'],
                'img_idx':    record['img_idx'],
                'gt':         gt_caption,
                'pred':       pred_dict,
                'raw_output': raw_text,
            }

            # Planning results (8 dense waypoints)
            if use_planning and pred_wp_batch is not None and gt_wp is not None:
                pred_wp = pred_wp_batch[idx][:8]  # [8, 3]
                gt_wp_8 = gt_wp[:8]               # [8, 3]

                all_pred_waypoints.append(pred_wp.tolist())
                all_gt_waypoints.append(gt_wp_8.tolist())
                sample_result['pred_waypoints'] = pred_wp.tolist()
                sample_result['gt_waypoints']   = gt_wp_8.tolist()

                # Per-sample displacement error
                disp = np.sqrt(
                    (pred_wp[:, 0] - gt_wp_8[:, 0])**2 +
                    (pred_wp[:, 1] - gt_wp_8[:, 1])**2
                )
                sample_result['per_step_error'] = {
                    k: round(float(disp[i]), 4) for i, k in enumerate(WAYPOINT_KEYS)
                }
                sample_result['sample_ade'] = round(float(disp.mean()), 4)
                sample_result['sample_fde'] = round(float(disp[-1]), 4)

                if multi_wp_batch is not None:
                    all_multi_waypoints.append(multi_wp_batch[idx].tolist())

            per_sample_results.append(sample_result)

    # ── Save cache ──
    if args.predictions_cache:
        cache_path = Path(args.predictions_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'all_gt_texts':       all_gt_texts,
            'all_pred_texts':     all_pred_texts,
            'all_gt_dicts':       all_gt_dicts,
            'all_pred_dicts':     all_pred_dicts,
            'all_pred_waypoints': all_pred_waypoints if all_pred_waypoints else None,
            'all_gt_waypoints':   all_gt_waypoints if all_gt_waypoints else None,
            'all_multi_waypoints': all_multi_waypoints if all_multi_waypoints else None,
            'per_sample_results': per_sample_results,
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        logger.info(f"Cache saved to {cache_path}")

    if not all_gt_texts:
        logger.error("No samples evaluated.")
        return

    _run_all_metrics(
        args, all_gt_texts, all_pred_texts,
        all_gt_dicts, all_pred_dicts,
        all_pred_waypoints, all_gt_waypoints,
        all_multi_waypoints,
        per_sample_results, sequences, device, use_planning,
    )


def _run_all_metrics(args, all_gt_texts, all_pred_texts,
                     all_gt_dicts, all_pred_dicts,
                     all_pred_waypoints, all_gt_waypoints,
                     all_multi_waypoints,
                     per_sample_results, sequences, device, use_planning):
    """Compute and print all metrics."""

    # ══════════════════════════════════════════════════════════
    # VQA Metrics
    # ══════════════════════════════════════════════════════════
    logger.info("Computing BLEU scores ...")
    bleu = compute_bleu_scores(all_gt_texts, all_pred_texts)

    bert = {}
    if not args.no_bert:
        logger.info("Computing BERTScore ...")
        bert = compute_bert_scores(all_gt_texts, all_pred_texts,
                                   lang=args.bert_lang, device=device)

    logger.info("Computing field accuracy ...")
    field_acc = compute_field_accuracy(all_gt_dicts, all_pred_dicts)

    # Per-sequence VQA breakdown
    seq_vqa_metrics = {}
    for seq in sequences:
        idxs = [i for i, r in enumerate(per_sample_results) if r['seq'] == seq]
        if not idxs:
            continue
        seq_gt   = [all_gt_texts[i]   for i in idxs]
        seq_pred = [all_pred_texts[i] for i in idxs]
        seq_vqa_metrics[seq] = compute_bleu_scores(seq_gt, seq_pred)
        seq_vqa_metrics[seq]['n_samples'] = len(idxs)

    # ══════════════════════════════════════════════════════════
    # Planning Metrics
    # ══════════════════════════════════════════════════════════
    plan_metrics = {}
    seq_plan_metrics = {}
    diversity_metrics = {}

    if use_planning and all_pred_waypoints and all_gt_waypoints:
        logger.info("Computing planning metrics ...")
        pred_wp_np = [np.array(w) for w in all_pred_waypoints]
        gt_wp_np   = [np.array(w) for w in all_gt_waypoints]
        plan_metrics = compute_planning_metrics(pred_wp_np, gt_wp_np)

        # Per-sequence planning breakdown
        for seq in sequences:
            idxs = [i for i, r in enumerate(per_sample_results)
                    if r['seq'] == seq and 'pred_waypoints' in r]
            if not idxs:
                continue
            seq_pred_wp = [np.array(all_pred_waypoints[i]) for i in idxs]
            seq_gt_wp   = [np.array(all_gt_waypoints[i])   for i in idxs]
            seq_plan_metrics[seq] = compute_planning_metrics(seq_pred_wp, seq_gt_wp)

        # Diversity (multi-anchor)
        if all_multi_waypoints:
            multi_np = [np.array(w) for w in all_multi_waypoints]
            diversity_metrics = compute_multimodal_diversity(multi_np)

    # meteor = compute_meteor_scores(all_gt_texts, all_pred_texts)
    # rouge  = compute_rouge_scores(all_gt_texts, all_pred_texts)
    # cider  = compute_cider_scores(all_gt_texts, all_pred_texts)

    # # 预计算 per-sample ROUGE-L F 分数，传给 joint analysis 避免重复算 LCS
    # from enhanced_metrics import compute_rouge_f_scores
    # rouge_f_scores = compute_rouge_f_scores(all_gt_texts, all_pred_texts)

    # joint  = compute_caption_plan_joint(
    #     per_sample_results, all_gt_texts, all_pred_texts,
    #     all_gt_dicts, all_pred_dicts,
    #     precomputed_rouge_f=rouge_f_scores)

    print(">>> Starting METEOR ...")
    meteor = compute_meteor_scores(all_gt_texts, all_pred_texts)
    print(">>> Starting ROUGE ...")
    rouge  = compute_rouge_scores(all_gt_texts, all_pred_texts)
    print(">>> Starting CIDEr ...")
    cider  = compute_cider_scores(all_gt_texts, all_pred_texts)
    print(">>> Starting ROUGE-F precompute ...")
    from enhanced_metrics import compute_rouge_f_scores
    rouge_f_scores = compute_rouge_f_scores(all_gt_texts, all_pred_texts)
    print(">>> Starting Joint ...")
    joint  = compute_caption_plan_joint(
        per_sample_results, all_gt_texts, all_pred_texts,
        all_gt_dicts, all_pred_dicts,
        precomputed_rouge_f=rouge_f_scores)
    print(">>> Starting Safety ...")
    safety = {}
    if use_planning and all_pred_waypoints:
        safety = compute_trajectory_safety([np.array(w) for w in all_pred_waypoints])
    print_enhanced_summary(meteor, rouge, cider, joint, safety)

    # ══════════════════════════════════════════════════════════
    # Print summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Sequences : {sequences}")
    print(f"  Samples   : {len(all_gt_texts)}")

    # ── VQA ──
    print()
    print("══ SCENE CAPTIONING (VQA) ══════════════════════════════════")
    print()
    print("── BLEU Scores (corpus-level) ──")
    for k, v in bleu.items():
        print(f"  {k}: {v:.4f}")
    if bert:
        print()
        print("── BERTScore ──")
        for k, v in bert.items():
            print(f"  {k}: {v:.4f}")
    print()
    print("── Field Accuracy (exact match) ──")
    for k, v in field_acc.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("── Per-Sequence BLEU-4 ──")
    for seq, m in seq_vqa_metrics.items():
        print(f"  {seq:25s}: BLEU-4={m['BLEU-4']:.4f}  n={m['n_samples']}")

    # ── Planning ──
    if plan_metrics:
        print()
        print("══ TRAJECTORY PLANNING ═════════════════════════════════════")
        print(f"  Samples: {plan_metrics.get('num_samples', 0)}")

        print()
        print("── Loss-Aligned Metrics (compare with training loss) ──")
        print(f"  smooth_l1 (≈rec_loss):     {plan_metrics['smooth_l1 (≈rec_loss)']:.4f}")
        print(f"  weighted_ADE (≈ade_loss):   {plan_metrics['weighted_ADE (≈ade_loss)']:.4f}")
        print(f"  estimated_total (rec+ade):  {plan_metrics['estimated_total_loss']:.4f}")

        print()
        print("── Standard Metrics (meters) ──")
        print(f"  ADE:       {plan_metrics['ADE']:.4f}")
        print(f"  FDE:       {plan_metrics['FDE']:.4f}")
        print(f"  medianADE: {plan_metrics['medianADE']:.4f}")
        print(f"  minADE:    {plan_metrics['minADE']:.4f}")
        print(f"  maxADE:    {plan_metrics['maxADE']:.4f}")

        print()
        print("── Per-Step L2 / Weighted L2 (8 dense waypoints, 0.5s-4.0s) ──")
        print(f"  {'step':>8s}  {'L2(m)':>8s}  {'wL2':>8s}  {'weight':>6s}  {'long':>8s}  {'lat':>8s}")
        for i, key in enumerate(WAYPOINT_KEYS):
            w = WAYPOINT_WEIGHTS[i]
            print(f"  {key:>8s}  {plan_metrics.get(f'L2_{key}', 0):>8.4f}  "
                  f"{plan_metrics.get(f'wL2_{key}', 0):>8.4f}  {w:>6.1f}  "
                  f"{plan_metrics.get(f'long_{key}', 0):>8.4f}  "
                  f"{plan_metrics.get(f'lat_{key}', 0):>8.4f}")

        if 'avg_z_err' in plan_metrics:
            print(f"  z-axis avg: {plan_metrics['avg_z_err']:.4f}")

        print()
        print("── Accuracy at Thresholds ──")
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            k = f'ADE<{t}m'
            if k in plan_metrics:
                print(f"  {k}: {plan_metrics[k]:.1f}%")

        if diversity_metrics:
            print()
            print("── Multi-Modal Diversity ──")
            for k, v in diversity_metrics.items():
                print(f"  {k}: {v}")

        if seq_plan_metrics:
            print()
            print("── Per-Sequence ──")
            for seq, m in seq_plan_metrics.items():
                print(f"  {seq:25s}: ADE={m['ADE']:.4f}  FDE={m['FDE']:.4f}  "
                      f"wADE={m['weighted_ADE (≈ade_loss)']:.4f}  n={m['num_samples']}")

    print("=" * 70)

    # ── Save results ──
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'sequences':     sequences,
        'n_samples':     len(all_gt_texts),
        'vqa': {
            'bleu':          bleu,
            'bert_score':    bert,
            'field_accuracy': field_acc,
            'per_sequence':  seq_vqa_metrics,
        },
        'planning': {
            'metrics':       plan_metrics,
            'diversity':     diversity_metrics,
            'per_sequence':  seq_plan_metrics,
        },
        'per_sample': per_sample_results,
        'enhanced': {'meteor': meteor, 'rouge': rouge, 'cider': cider,
             'caption_plan_joint': joint, 'trajectory_safety': safety},
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()