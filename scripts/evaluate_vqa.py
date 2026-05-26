# # """
# # Evaluation script for Multi-Modal VLM on UNISCP dataset.

# # Metrics:
# #   - BLEU-1/2/3/4  (nltk)
# #   - BERTScore     (bert-score)

# # Usage:
# #     python evaluate_fixed.py \
# #         --model_path save/multimodal_vlm/final_model \
# #         --data_root ./UNISCP \
# #         --sequences RURAL_A0 RURAL_A1 \
# #         --num_samples 200 \
# #         --output_file results/eval_results.json

# # Install dependencies if needed:
# #     pip install nltk bert-score --break-system-packages
# #     python -c "import nltk; nltk.download('punkt_tab')"
# # """

# # import os
# # import sys
# # import json
# # import argparse
# # import logging
# # from pathlib import Path
# # from tqdm import tqdm

# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # import torch
# # import numpy as np
# # from PIL import Image
# # from transformers import AutoTokenizer, AutoImageProcessor

# # import math
# # from collections import Counter
# # from bert_score import score as bert_score

# # sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# # # from model import MultiModalVLM, MultiModalVLMConfig
# # from model_with_planning import MultiModalVLM, MultiModalVLMConfig
# # from dataset import (
# #     UniscpDataset, read_pcd, get_rural_calibration,
# #     load_timestamps, find_nearest_idx,
# #     SYSTEM_PROMPT, USER_PROMPT,
# # )

# # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# # logger = logging.getLogger(__name__)

# # RURAL_SEQUENCES = ['RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B0', 'RURAL_B1', 'RURAL_B2']
# # OTHER_SEQUENCES = ['FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO', 'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE']
# # ALL_SEQUENCES   = RURAL_SEQUENCES + OTHER_SEQUENCES


# # # =============================================================================
# # # Data loading (reuse inference.py logic, but for whole sequences)
# # # =============================================================================

# # def load_sample_from_record(record: dict, calib, max_lidar=40000, max_radar=16000):
# #     """Load image, lidar, radar and ground truth caption from a sample record."""
# #     image = Image.open(record['img_path']).convert('RGB')

# #     # LiDAR
# #     data = read_pcd(record['lidar_path'], ['x', 'y', 'z', 'intensity'])
# #     x, y, z = data.get('x', np.zeros(0)), data.get('y', np.zeros(0)), data.get('z', np.zeros(0))
# #     intensity = data.get('intensity', np.zeros_like(x))
# #     lidar = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
# #     valid = np.all(np.isfinite(lidar), axis=-1) & np.any(lidar[:, :3] != 0, axis=-1)
# #     lidar = lidar[valid]
# #     lidar = calib.crop_lidar_to_fov(lidar)
# #     if len(lidar) > max_lidar:
# #         idx = np.random.choice(len(lidar), max_lidar, replace=False)
# #         lidar = lidar[idx]

# #     # Radar
# #     data = read_pcd(record['radar_path'], ['x', 'y', 'z', 'doppler', 'power', 'recoveredSpeed'])
# #     x, y, z = data.get('x', np.zeros(0)), data.get('y', np.zeros(0)), data.get('z', np.zeros(0))
# #     doppler = data.get('doppler', np.zeros_like(x))
# #     power   = data.get('power',   np.zeros_like(x))
# #     speed   = data.get('recoveredSpeed', np.zeros_like(x))
# #     radar = np.stack([x, y, z, doppler, power, speed], axis=-1).astype(np.float32)
# #     valid = np.all(np.isfinite(radar), axis=-1)
# #     radar = radar[valid]
# #     radar = calib.crop_radar_to_fov(radar)
# #     if len(radar) > max_radar:
# #         idx = np.random.choice(len(radar), max_radar, replace=False)
# #         radar = radar[idx]

# #     # Ground truth caption
# #     with open(record['caption_path'], 'r') as f:
# #         gt_caption = json.load(f)

# #     return image, lidar, radar, gt_caption


# # # =============================================================================
# # # Checkpoint weight loading  (THE KEY FIX)
# # # =============================================================================

# # def load_checkpoint_weights(model, model_path: str):
# #     """
# #     Load trained checkpoint weights into the model.

# #     This is the critical fix: the original evaluate.py only loaded the config
# #     from model_path and re-initialized the model from scratch.  The Q-Former,
# #     LiDAR/Radar encoders, and any fine-tuned LLM weights were never loaded,
# #     so evaluation was running on a randomly-initialized bridge.

# #     This function mirrors wild-drive/validate.py lines 392-412.
# #     """
# #     ckpt_path = Path(model_path)

# #     # Collect all weight files (.safetensors and .bin)
# #     weight_files = sorted(ckpt_path.glob("*.safetensors")) + sorted(ckpt_path.glob("*.bin"))

# #     if not weight_files:
# #         logger.warning(
# #             f"No weight files (.safetensors / .bin) found in {ckpt_path}. "
# #             f"The model will use random weights for the bridge layers!"
# #         )
# #         return

# #     state_dict = {}
# #     for wf in weight_files:
# #         logger.info(f"  Loading weights from {wf.name} ...")
# #         if wf.suffix == ".safetensors":
# #             from safetensors.torch import load_file
# #             state_dict.update(load_file(str(wf)))
# #         elif wf.suffix == ".bin":
# #             loaded = torch.load(str(wf), map_location="cpu", weights_only=False)
# #             if isinstance(loaded, dict):
# #                 state_dict.update(loaded)
# #             # Skip non-dict files (TrainingArguments, optimizer states, etc.)

# #     if not state_dict:
# #         logger.warning(
# #             f"Weight files were found but no valid state_dict could be loaded from {ckpt_path}."
# #         )
# #         return

# #     missing, unexpected = model.load_state_dict(state_dict, strict=False)
# #     logger.info(f"Checkpoint loaded from {ckpt_path}")
# #     logger.info(f"  Total keys in checkpoint : {len(state_dict)}")
# #     logger.info(f"  Missing keys  (in model but not in ckpt): {len(missing)}")
# #     logger.info(f"  Unexpected keys (in ckpt but not in model): {len(unexpected)}")

# #     # Print details for debugging (only first 10)
# #     if missing:
# #         logger.info(f"  First missing keys: {missing[:10]}")
# #     if unexpected:
# #         logger.info(f"  First unexpected keys: {unexpected[:10]}")


# # # =============================================================================
# # # Inference
# # # =============================================================================

# # def build_prompt(tokenizer, num_query_tokens):
# #     """Build the chat prompt string (same for every sample)."""
# #     messages = [
# #         {"role": "system", "content": SYSTEM_PROMPT},
# #         {"role": "user",   "content": "<image>\n" + USER_PROMPT},
# #     ]
# #     return tokenizer.apply_chat_template(
# #         messages, tokenize=False, add_generation_prompt=True
# #     ).replace('<image>', '<|image_pad|>' * num_query_tokens)


# # def parse_prediction(generated_text: str):
# #     """Extract JSON dict from generated text."""
# #     try:
# #         start = generated_text.find('{')
# #         end   = generated_text.rfind('}') + 1
# #         if start >= 0 and end > start:
# #             return json.loads(generated_text[start:end]), generated_text
# #     except json.JSONDecodeError:
# #         pass
# #     return {}, generated_text


# # def run_inference_batch(model, tokenizer, processor, images, lidar_list, radar_list,
# #                         num_query_tokens=64, device='cuda', max_new_tokens=512):
# #     """
# #     Run inference on a batch of samples.

# #     Args:
# #         images     : list of PIL.Image  (length B)
# #         lidar_list : list of np.ndarray (length B)
# #         radar_list : list of np.ndarray (length B)
# #     Returns:
# #         predictions : list of dicts  (length B)
# #         raw_texts   : list of str    (length B)
# #     """
# #     prompt_str = build_prompt(tokenizer, num_query_tokens)

# #     # image: stack into [B, 3, H, W]
# #     pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

# #     # text: pad to same length (left-pad for decoder-only generation)
# #     tokenizer.padding_side = 'left'
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #     B = len(images)
# #     encoded = tokenizer([prompt_str] * B, return_tensors="pt", padding=True)
# #     prompt_ids = encoded["input_ids"].to(device)

# #     # point clouds: keep as list (model expects List[Tensor])
# #     lidar_tensors = [torch.from_numpy(p).float().to(device) for p in lidar_list]
# #     radar_tensors = [torch.from_numpy(p).float().to(device) for p in radar_list]

# #     with torch.no_grad():
# #         output_ids = model.generate(
# #             pixel_values=pixel_values,
# #             lidar_points=lidar_tensors,
# #             radar_points=radar_tensors,
# #             prompt_ids=prompt_ids,
# #             max_new_tokens=max_new_tokens,
# #             temperature=0.1,
# #         )

# #     predictions, raw_texts = [], []
# #     for i in range(B):
# #         text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
# #         pred, raw = parse_prediction(text)
# #         predictions.append(pred)
# #         raw_texts.append(raw)

# #     return predictions, raw_texts


# # # =============================================================================
# # # Metrics
# # # =============================================================================

# # def flatten_json_to_text(obj) -> str:
# #     """Recursively flatten JSON values to a string for BLEU scoring.
    
# #     Only extracts VALUES (not keys) to match the original validate.py behavior
# #     where GT and pred are raw answer strings without structural key names.
# #     """
# #     if isinstance(obj, dict):
# #         return ' '.join(flatten_json_to_text(v) for v in obj.values())
# #     elif isinstance(obj, list):
# #         return ' '.join(flatten_json_to_text(i) for i in obj)
# #     else:
# #         return str(obj)


# # def _ngrams(tokens, n):
# #     return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# # def _modified_precision(references, hypotheses, n):
# #     """Corpus-level modified n-gram precision."""
# #     clipped_count = 0
# #     total_count = 0
# #     for refs, hyp in zip(references, hypotheses):
# #         hyp_ngrams = Counter(_ngrams(hyp, n))
# #         max_ref_counts = Counter()
# #         for ref in refs:
# #             ref_ngrams = Counter(_ngrams(ref, n))
# #             for ng, count in ref_ngrams.items():
# #                 max_ref_counts[ng] = max(max_ref_counts[ng], count)
# #         for ng, count in hyp_ngrams.items():
# #             clipped_count += min(count, max_ref_counts.get(ng, 0))
# #             total_count += count
# #     if total_count == 0:
# #         return 0.0
# #     return clipped_count / total_count


# # def compute_bleu_scores(references: list[str], hypotheses: list[str]) -> dict:
# #     """
# #     Compute corpus-level BLEU-1/2/3/4 using the same method as validate.py:
# #     - Simple .split() tokenization (no nltk word_tokenize)
# #     - +1e-5 Laplace smoothing
# #     - Closest-reference brevity penalty
# #     """
# #     refs_tok = [[ref.lower().split()] for ref in references]
# #     hyps_tok = [hyp.lower().split() for hyp in hypotheses]

# #     hyp_len = sum(len(h) for h in hyps_tok)
# #     ref_len = 0
# #     for refs, hyp in zip(refs_tok, hyps_tok):
# #         closest = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
# #         ref_len += closest

# #     if hyp_len == 0:
# #         bp = 0.0
# #     elif hyp_len >= ref_len:
# #         bp = 1.0
# #     else:
# #         bp = math.exp(1.0 - ref_len / hyp_len)

# #     weights_map = {
# #         'BLEU-1': [1.0],
# #         'BLEU-2': [0.5, 0.5],
# #         'BLEU-3': [1/3, 1/3, 1/3],
# #         'BLEU-4': [0.25, 0.25, 0.25, 0.25],
# #     }
# #     results = {}
# #     for name, weights in weights_map.items():
# #         log_avg = 0.0
# #         for i, w in enumerate(weights):
# #             p = _modified_precision(refs_tok, hyps_tok, i + 1) + 1e-5
# #             log_avg += w * math.log(p)
# #         results[name] = round(bp * math.exp(log_avg), 4)
# #     return results


# # def compute_bert_scores(references: list[str], hypotheses: list[str],
# #                         lang='en', device='cuda') -> dict:
# #     """Compute BERTScore P/R/F1."""
# #     P, R, F1 = bert_score(
# #         hypotheses, references,
# #         lang=lang,
# #         device=device,
# #         verbose=False,
# #     )
# #     return {
# #         'BERTScore-P':  round(P.mean().item(),  4),
# #         'BERTScore-R':  round(R.mean().item(),  4),
# #         'BERTScore-F1': round(F1.mean().item(), 4),
# #     }


# # def compute_field_accuracy(gt_list: list[dict], pred_list: list[dict]) -> dict:
# #     """
# #     Per-field exact-match accuracy for closed-set fields.
# #     Gives a quick sense of structured prediction quality.
# #     """
# #     field_paths = [
# #         ('weather',           'condition'),
# #         ('weather',           'illumination'),
# #         ('traffic_light',     'present'),
# #         ('traffic_light',     'state'),
# #         ('traffic_sign',      'present'),
# #         ('traffic_sign',      'category'),
# #         ('forward_drivability','status'),
# #         ('lane_keeping',      'status'),
# #         ('lane_keeping',      'deviation'),
# #         ('driving_advice',    'action'),
# #         ('hazard_region',     'present'),
# #         ('hazard_region',     'type'),
# #     ]
# #     counts  = {f'{a}/{b}': {'correct': 0, 'total': 0} for a, b in field_paths}
# #     overall_correct = overall_total = 0

# #     for gt, pred in zip(gt_list, pred_list):
# #         for top, sub in field_paths:
# #             gt_val   = gt.get(top, {}).get(sub, None)
# #             pred_val = pred.get(top, {}).get(sub, None)
# #             if gt_val is not None:
# #                 counts[f'{top}/{sub}']['total'] += 1
# #                 overall_total += 1
# #                 if str(gt_val).lower() == str(pred_val).lower():
# #                     counts[f'{top}/{sub}']['correct'] += 1
# #                     overall_correct += 1

# #     result = {}
# #     for key, v in counts.items():
# #         if v['total'] > 0:
# #             result[key] = round(v['correct'] / v['total'], 4)

# #     result['overall_field_acc'] = round(overall_correct / overall_total, 4) if overall_total else 0.0
# #     return result


# # # =============================================================================
# # # Main
# # # =============================================================================

# # def parse_args():
# #     parser = argparse.ArgumentParser(description='Evaluate Multi-Modal VLM')
# #     parser.add_argument('--model_path',   type=str, required=True,
# #                         help='Path to saved model (e.g. save/multimodal_vlm/final_model)')
# #     parser.add_argument('--data_root',    type=str, required=True,
# #                         help='Path to UNISCP dataset root')
# #     parser.add_argument('--sequences',    nargs='+', default=None,
# #                         help='Sequences to evaluate (default: all). '
# #                              f'Available: {ALL_SEQUENCES}')
# #     parser.add_argument('--num_samples',  type=int, default=None,
# #                         help='Max samples per sequence (default: all)')
# #     parser.add_argument('--output_file',  type=str, default='eval_results.json',
# #                         help='Where to save detailed results JSON')
# #     parser.add_argument('--device',       type=str, default='cuda')
# #     parser.add_argument('--max_new_tokens', type=int, default=512)
# #     parser.add_argument('--bert_lang',    type=str, default='en',
# #                         help='Language for BERTScore (en / zh / ...)')
# #     parser.add_argument('--no_bert',      action='store_true',
# #                         help='Skip BERTScore (faster, saves VRAM)')
# #     parser.add_argument('--predictions_cache', type=str, default=None,
# #                         help='Path to save/load inference results (skip re-running inference if exists)')
# #     parser.add_argument('--batch_size',   type=int, default=1,
# #                         help='Inference batch size (default: 1). Increase for faster evaluation if VRAM allows.')
# #     return parser.parse_args()


# # def main():
# #     args   = parse_args()
# #     device = args.device if torch.cuda.is_available() else 'cpu'

# #     # ── sequences ────────────────────────────────────────────────────────────
# #     sequences = args.sequences if args.sequences else ALL_SEQUENCES
# #     logger.info(f"Evaluating sequences: {sequences}")

# #     # ── model ────────────────────────────────────────────────────────────────
# #     logger.info(f"Loading model from {args.model_path} ...")
# #     config    = MultiModalVLMConfig.from_pretrained(args.model_path)
# #     model     = MultiModalVLM(config)

# #     # =====================================================================
# #     # FIX: Load trained checkpoint weights into the model
# #     # The original code stopped here — it only loaded the config and created
# #     # a fresh model.  Q-Former, LiDAR/Radar encoders, and fine-tuned LLM
# #     # weights were all random.  This is why BLEU-1 was ~10.
# #     # =====================================================================
# #     load_checkpoint_weights(model, args.model_path)

# #     model = model.to(device)
# #     model.eval()

# #     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# #     processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

# #     if '<|image_pad|>' not in tokenizer.get_vocab():
# #         tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
# #         model.llm_model.resize_token_embeddings(len(tokenizer))
# #         model.tokenizer = tokenizer

# #     # ── load from cache if available ────────────────────────────────────────
# #     if args.predictions_cache and Path(args.predictions_cache).exists():
# #         logger.info(f"Loading inference cache from {args.predictions_cache} (skipping inference) ...")
# #         with open(args.predictions_cache, 'r', encoding='utf-8') as f:
# #             cache = json.load(f)
# #         all_gt_texts        = cache['all_gt_texts']
# #         all_pred_texts      = cache['all_pred_texts']
# #         all_gt_dicts        = cache['all_gt_dicts']
# #         all_pred_dicts      = cache['all_pred_dicts']
# #         per_sample_results  = cache['per_sample_results']
# #         logger.info(f"Loaded {len(all_gt_texts)} cached samples, skipping to metrics ...")
# #         # jump straight to metrics
# #         _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
# #                      all_pred_dicts, per_sample_results, sequences, device)
# #         return

# #     # ── build sample list ────────────────────────────────────────────────────
# #     calib   = get_rural_calibration()
# #     dataset = UniscpDataset(
# #         data_root=args.data_root,
# #         sequences=sequences,
# #         tokenizer=tokenizer,
# #         processor=processor,
# #     )
# #     samples = dataset.samples  # raw records (no preprocessing)

# #     if args.num_samples:
# #         # sample evenly across sequences
# #         per_seq   = args.num_samples
# #         seq_groups: dict[str, list] = {}
# #         for s in samples:
# #             seq_groups.setdefault(s['seq'], []).append(s)
# #         samples = []
# #         for seq in sequences:
# #             group = seq_groups.get(seq, [])
# #             step  = max(1, len(group) // per_seq)
# #             samples += group[::step][:per_seq]

# #     logger.info(f"Total samples to evaluate: {len(samples)}")

# #     # ── run inference ────────────────────────────────────────────────────────
# #     all_gt_texts   = []
# #     all_pred_texts = []
# #     all_gt_dicts   = []
# #     all_pred_dicts = []
# #     per_sample_results = []

# #     # ── batch inference loop ────────────────────────────────────────────────
# #     batch_size = args.batch_size
# #     for batch_start in tqdm(range(0, len(samples), batch_size), desc='Inference'):
# #         batch_records = samples[batch_start: batch_start + batch_size]

# #         # load each sample in the mini-batch
# #         batch_images, batch_lidar, batch_radar, batch_gt = [], [], [], []
# #         valid_records = []
# #         for record in batch_records:
# #             try:
# #                 image, lidar_pts, radar_pts, gt_caption = load_sample_from_record(record, calib)
# #                 batch_images.append(image)
# #                 batch_lidar.append(lidar_pts)
# #                 batch_radar.append(radar_pts)
# #                 batch_gt.append(gt_caption)
# #                 valid_records.append(record)
# #             except Exception as e:
# #                 logger.warning(f"Load failed [{record['seq']} {record['img_idx']}]: {e}")

# #         if not batch_images:
# #             continue

# #         pred_dicts, raw_texts = run_inference_batch(
# #             model, tokenizer, processor,
# #             batch_images, batch_lidar, batch_radar,
# #             num_query_tokens=config.num_query_tokens,
# #             device=device,
# #             max_new_tokens=args.max_new_tokens,
# #         )

# #         for record, gt_caption, pred_dict, raw_text in zip(
# #                 valid_records, batch_gt, pred_dicts, raw_texts):
# #             gt_text   = flatten_json_to_text(gt_caption)
# #             pred_text = flatten_json_to_text(pred_dict) if pred_dict else raw_text

# #             all_gt_texts.append(gt_text)
# #             all_pred_texts.append(pred_text)
# #             all_gt_dicts.append(gt_caption)
# #             all_pred_dicts.append(pred_dict)

# #             per_sample_results.append({
# #                 'seq':        record['seq'],
# #                 'img_idx':    record['img_idx'],
# #                 'gt':         gt_caption,
# #                 'pred':       pred_dict,
# #                 'raw_output': raw_text,
# #             })

# #     # ── save inference cache ────────────────────────────────────────────────
# #     if args.predictions_cache:
# #         cache_path = Path(args.predictions_cache)
# #         cache_path.parent.mkdir(parents=True, exist_ok=True)
# #         cache_data = {
# #             'all_gt_texts':   all_gt_texts,
# #             'all_pred_texts': all_pred_texts,
# #             'all_gt_dicts':   all_gt_dicts,
# #             'all_pred_dicts': all_pred_dicts,
# #             'per_sample_results': per_sample_results,
# #         }
# #         with open(cache_path, 'w', encoding='utf-8') as f:
# #             json.dump(cache_data, f, ensure_ascii=False)
# #         logger.info(f"Inference cache saved to {cache_path}")

# #     if not all_gt_texts:
# #         logger.error("No samples were successfully evaluated.")
# #         return

# #     _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
# #                  all_pred_dicts, per_sample_results, sequences, device)


# # def _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
# #                  all_pred_dicts, per_sample_results, sequences, device):
# #     # ── compute metrics ──────────────────────────────────────────────────────
# #     logger.info("Computing BLEU scores ...")
# #     bleu = compute_bleu_scores(all_gt_texts, all_pred_texts)

# #     bert = {}
# #     if not args.no_bert:
# #         logger.info("Computing BERTScore (this may take a while) ...")
# #         bert = compute_bert_scores(all_gt_texts, all_pred_texts,
# #                                    lang=args.bert_lang, device=device)

# #     logger.info("Computing field accuracy ...")
# #     field_acc = compute_field_accuracy(all_gt_dicts, all_pred_dicts)

# #     # ── per-sequence breakdown ────────────────────────────────────────────────
# #     seq_metrics = {}
# #     for seq in sequences:
# #         idxs = [i for i, r in enumerate(per_sample_results) if r['seq'] == seq]
# #         if not idxs:
# #             continue
# #         seq_gt   = [all_gt_texts[i]   for i in idxs]
# #         seq_pred = [all_pred_texts[i] for i in idxs]
# #         seq_metrics[seq] = compute_bleu_scores(seq_gt, seq_pred)
# #         seq_metrics[seq]['n_samples'] = len(idxs)
# #         if not args.no_bert:
# #             seq_metrics[seq].update(
# #                 compute_bert_scores(seq_gt, seq_pred,
# #                                     lang=args.bert_lang, device=device)
# #             )

# #     # ── print summary ────────────────────────────────────────────────────────
# #     print("\n" + "=" * 60)
# #     print("EVALUATION SUMMARY")
# #     print("=" * 60)
# #     print(f"  Sequences : {sequences}")
# #     print(f"  Samples   : {len(all_gt_texts)}")
# #     print()
# #     print("── BLEU Scores (corpus-level) ──────────────────────────────")
# #     for k, v in bleu.items():
# #         print(f"  {k}: {v:.4f}")
# #     if bert:
# #         print()
# #         print("── BERTScore ───────────────────────────────────────────────")
# #         for k, v in bert.items():
# #             print(f"  {k}: {v:.4f}")
# #     print()
# #     print("── Field Accuracy (exact match) ────────────────────────────")
# #     for k, v in field_acc.items():
# #         print(f"  {k}: {v:.4f}")
# #     print()
# #     print("── Per-Sequence BLEU-4 ──────────────────────────────────────")
# #     for seq, m in seq_metrics.items():
# #         print(f"  {seq:25s}: BLEU-4={m['BLEU-4']:.4f}  n={m['n_samples']}")
# #     print("=" * 60)

# #     # ── save results ─────────────────────────────────────────────────────────
# #     output_path = Path(args.output_file)
# #     output_path.parent.mkdir(parents=True, exist_ok=True)

# #     results = {
# #         'sequences': sequences,
# #         'n_samples':  len(all_gt_texts),
# #         'bleu':       bleu,
# #         'bert_score': bert,
# #         'field_accuracy': field_acc,
# #         'per_sequence':   seq_metrics,
# #         'per_sample':     per_sample_results,
# #     }
# #     with open(output_path, 'w', encoding='utf-8') as f:
# #         json.dump(results, f, indent=2, ensure_ascii=False)

# #     logger.info(f"Detailed results saved to {output_path}")


# # if __name__ == '__main__':
# #     main()

# """
# Evaluation script for Multi-Modal VLM on UNISCP dataset.

# Metrics:
#   - BLEU-1/2/3/4  (hand-written, corpus-level)
#   - METEOR        (hand-written, exact-match, sentence-level avg)
#   - ROUGE-L       (hand-written, LCS-based, sentence-level avg)
#   - BERTScore     (bert-score library)

# Usage:
#     python evaluate.py \
#         --model_path save/multimodal_vlm/final_model \
#         --data_root ./UNISCP \
#         --sequences RURAL_A0 RURAL_A1 \
#         --num_samples 200 \
#         --output_file results/eval_results.json

# Install dependencies if needed:
#     pip install bert-score --break-system-packages
# """

# import os
# import sys
# import json
# import argparse
# import logging
# from pathlib import Path
# from tqdm import tqdm

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import torch
# import numpy as np
# from PIL import Image
# from transformers import AutoTokenizer, AutoImageProcessor

# import math
# from collections import Counter
# from bert_score import score as bert_score

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# # from model import MultiModalVLM, MultiModalVLMConfig
# from model_with_planning import MultiModalVLM, MultiModalVLMConfig
# from dataset import (
#     UniscpDataset, read_pcd, get_rural_calibration,
#     load_timestamps, find_nearest_idx,
#     SYSTEM_PROMPT, USER_PROMPT,
# )

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
# logger = logging.getLogger(__name__)

# RURAL_SEQUENCES = ['RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B0', 'RURAL_B1', 'RURAL_B2']
# OTHER_SEQUENCES = ['FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO', 'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE']
# ALL_SEQUENCES   = RURAL_SEQUENCES + OTHER_SEQUENCES


# # =============================================================================
# # Data loading (reuse inference.py logic, but for whole sequences)
# # =============================================================================

# def load_sample_from_record(record: dict, calib, max_lidar=40000, max_radar=16000):
#     """Load image, lidar, radar and ground truth caption from a sample record."""
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

#     # Ground truth caption
#     with open(record['caption_path'], 'r') as f:
#         gt_caption = json.load(f)

#     return image, lidar, radar, gt_caption


# # =============================================================================
# # Checkpoint weight loading  (THE KEY FIX)
# # =============================================================================

# def load_checkpoint_weights(model, model_path: str):
#     """
#     Load trained checkpoint weights into the model.

#     This is the critical fix: the original evaluate.py only loaded the config
#     from model_path and re-initialized the model from scratch.  The Q-Former,
#     LiDAR/Radar encoders, and any fine-tuned LLM weights were never loaded,
#     so evaluation was running on a randomly-initialized bridge.

#     This function mirrors wild-drive/validate.py lines 392-412.
#     """
#     ckpt_path = Path(model_path)

#     # Collect all weight files (.safetensors and .bin)
#     weight_files = sorted(ckpt_path.glob("*.safetensors")) + sorted(ckpt_path.glob("*.bin"))

#     if not weight_files:
#         logger.warning(
#             f"No weight files (.safetensors / .bin) found in {ckpt_path}. "
#             f"The model will use random weights for the bridge layers!"
#         )
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
#             # Skip non-dict files (TrainingArguments, optimizer states, etc.)

#     if not state_dict:
#         logger.warning(
#             f"Weight files were found but no valid state_dict could be loaded from {ckpt_path}."
#         )
#         return

#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     logger.info(f"Checkpoint loaded from {ckpt_path}")
#     logger.info(f"  Total keys in checkpoint : {len(state_dict)}")
#     logger.info(f"  Missing keys  (in model but not in ckpt): {len(missing)}")
#     logger.info(f"  Unexpected keys (in ckpt but not in model): {len(unexpected)}")

#     # Print details for debugging (only first 10)
#     if missing:
#         logger.info(f"  First missing keys: {missing[:10]}")
#     if unexpected:
#         logger.info(f"  First unexpected keys: {unexpected[:10]}")


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
#                         num_query_tokens=64, device='cuda', max_new_tokens=512):
#     """
#     Run inference on a batch of samples.

#     Args:
#         images     : list of PIL.Image  (length B)
#         lidar_list : list of np.ndarray (length B)
#         radar_list : list of np.ndarray (length B)
#     Returns:
#         predictions : list of dicts  (length B)
#         raw_texts   : list of str    (length B)
#     """
#     prompt_str = build_prompt(tokenizer, num_query_tokens)

#     # image: stack into [B, 3, H, W]
#     pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

#     # text: pad to same length (left-pad for decoder-only generation)
#     tokenizer.padding_side = 'left'
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     B = len(images)
#     encoded = tokenizer([prompt_str] * B, return_tensors="pt", padding=True)
#     prompt_ids = encoded["input_ids"].to(device)

#     # point clouds: keep as list (model expects List[Tensor])
#     lidar_tensors = [torch.from_numpy(p).float().to(device) for p in lidar_list]
#     radar_tensors = [torch.from_numpy(p).float().to(device) for p in radar_list]

#     with torch.no_grad():
#         output_ids = model.generate(
#             pixel_values=pixel_values,
#             lidar_points=lidar_tensors,
#             radar_points=radar_tensors,
#             prompt_ids=prompt_ids,
#             max_new_tokens=max_new_tokens,
#             temperature=0.1,
#         )

#     predictions, raw_texts = [], []
#     for i in range(B):
#         text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
#         pred, raw = parse_prediction(text)
#         predictions.append(pred)
#         raw_texts.append(raw)

#     return predictions, raw_texts


# # =============================================================================
# # Metrics
# # =============================================================================

# def flatten_json_to_text(obj) -> str:
#     """Recursively flatten JSON values to a string for BLEU scoring.

#     Only extracts VALUES (not keys) to match the original validate.py behavior
#     where GT and pred are raw answer strings without structural key names.
#     """
#     if isinstance(obj, dict):
#         return ' '.join(flatten_json_to_text(v) for v in obj.values())
#     elif isinstance(obj, list):
#         return ' '.join(flatten_json_to_text(i) for i in obj)
#     else:
#         return str(obj)


# # ── BLEU ─────────────────────────────────────────────────────────────────────

# def _ngrams(tokens, n):
#     return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


# def _modified_precision(references, hypotheses, n):
#     """Corpus-level modified n-gram precision."""
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


# def compute_bleu_scores(references: list[str], hypotheses: list[str]) -> dict:
#     """
#     Compute corpus-level BLEU-1/2/3/4 using the same method as validate.py:
#     - Simple .split() tokenization (no nltk word_tokenize)
#     - +1e-5 Laplace smoothing
#     - Closest-reference brevity penalty
#     """
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


# # ── METEOR (self-contained, no nltk dependency) ─────────────────────────────

# def _meteor_exact_matches(ref_tokens, hyp_tokens):
#     """Find exact unigram matches (greedy, each token matched at most once)."""
#     ref_available = list(range(len(ref_tokens)))
#     matches = []
#     for h_idx, h_tok in enumerate(hyp_tokens):
#         for r_idx in ref_available:
#             if ref_tokens[r_idx] == h_tok:
#                 matches.append((h_idx, r_idx))
#                 ref_available.remove(r_idx)
#                 break
#     return matches


# def _meteor_chunks(matches):
#     """Count the number of contiguous chunks in matched pairs."""
#     if not matches:
#         return 0
#     sorted_matches = sorted(matches, key=lambda x: x[0])
#     chunks = 1
#     for i in range(1, len(sorted_matches)):
#         if (sorted_matches[i][0] != sorted_matches[i-1][0] + 1 or
#                 sorted_matches[i][1] != sorted_matches[i-1][1] + 1):
#             chunks += 1
#     return chunks


# def _meteor_sentence(ref_tokens, hyp_tokens, alpha=0.9, beta=3.0, gamma=0.5):
#     """
#     Compute sentence-level METEOR (exact match only, no stemming/synonyms).

#     Parameters follow Banerjee & Lavie (2005) defaults:
#       alpha = 0.9  (relative weight of precision vs recall in harmonic mean)
#       beta  = 3.0  (shape of fragmentation penalty)
#       gamma = 0.5  (max fragmentation penalty)
#     """
#     if not hyp_tokens and not ref_tokens:
#         return 1.0
#     if not hyp_tokens or not ref_tokens:
#         return 0.0

#     matches = _meteor_exact_matches(ref_tokens, hyp_tokens)
#     m = len(matches)

#     if m == 0:
#         return 0.0

#     p = m / len(hyp_tokens)
#     r = m / len(ref_tokens)

#     f_mean = (p * r) / (alpha * p + (1 - alpha) * r)

#     chunks = _meteor_chunks(matches)
#     frag = gamma * (chunks / m) ** beta
#     score = f_mean * (1 - frag)

#     return score


# def compute_meteor_score(references: list[str], hypotheses: list[str]) -> dict:
#     """
#     Compute corpus-level METEOR (average of sentence-level scores).

#     Exact-match only variant — no stemming or WordNet synonyms, but
#     avoids all nltk/sqlite3 dependencies.
#     Tokenization: simple .split() to stay consistent with BLEU above.
#     """
#     scores = []
#     for ref, hyp in zip(references, hypotheses):
#         ref_tokens = ref.lower().split()
#         hyp_tokens = hyp.lower().split()
#         scores.append(_meteor_sentence(ref_tokens, hyp_tokens))

#     avg = sum(scores) / len(scores) if scores else 0.0
#     return {'METEOR': round(avg, 4)}


# # ── ROUGE-L (self-contained, LCS-based) ─────────────────────────────────────

# def _lcs_length(x: list, y: list) -> int:
#     """Compute length of Longest Common Subsequence via DP (space-optimized)."""
#     m, n = len(x), len(y)
#     prev = [0] * (n + 1)
#     curr = [0] * (n + 1)
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if x[i - 1] == y[j - 1]:
#                 curr[j] = prev[j - 1] + 1
#             else:
#                 curr[j] = max(curr[j - 1], prev[j])
#         prev, curr = curr, [0] * (n + 1)
#     return prev[n]


# def compute_rouge_l(references: list[str], hypotheses: list[str],
#                     beta: float = 1.2) -> dict:
#     """
#     Compute corpus-level ROUGE-L (sentence-level, then averaged).

#     ROUGE-L uses the Longest Common Subsequence (LCS) between reference and
#     hypothesis to compute precision, recall, and F-measure.

#     Args:
#         beta: Controls the relative importance of recall vs precision in F-measure.
#               beta > 1 favors recall (default 1.2 follows the original Lin 2004 paper).

#     Returns:
#         dict with ROUGE-L-P, ROUGE-L-R, ROUGE-L-F keys.
#     """
#     precisions, recalls, f_scores = [], [], []

#     for ref, hyp in zip(references, hypotheses):
#         ref_tokens = ref.lower().split()
#         hyp_tokens = hyp.lower().split()

#         if len(hyp_tokens) == 0 and len(ref_tokens) == 0:
#             precisions.append(1.0)
#             recalls.append(1.0)
#             f_scores.append(1.0)
#             continue

#         if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
#             precisions.append(0.0)
#             recalls.append(0.0)
#             f_scores.append(0.0)
#             continue

#         lcs = _lcs_length(ref_tokens, hyp_tokens)
#         p = lcs / len(hyp_tokens)
#         r = lcs / len(ref_tokens)

#         if p == 0 and r == 0:
#             f = 0.0
#         else:
#             beta_sq = beta ** 2
#             f = (1 + beta_sq) * p * r / (beta_sq * p + r)

#         precisions.append(p)
#         recalls.append(r)
#         f_scores.append(f)

#     n = len(f_scores) if f_scores else 1
#     return {
#         'ROUGE-L-P': round(sum(precisions) / n, 4),
#         'ROUGE-L-R': round(sum(recalls) / n, 4),
#         'ROUGE-L-F': round(sum(f_scores) / n, 4),
#     }


# # ── BERTScore ────────────────────────────────────────────────────────────────

# def compute_bert_scores(references: list[str], hypotheses: list[str],
#                         lang='en', device='cuda') -> dict:
#     """Compute BERTScore P/R/F1."""
#     P, R, F1 = bert_score(
#         hypotheses, references,
#         lang=lang,
#         device=device,
#         verbose=False,
#     )
#     return {
#         'BERTScore-P':  round(P.mean().item(),  4),
#         'BERTScore-R':  round(R.mean().item(),  4),
#         'BERTScore-F1': round(F1.mean().item(), 4),
#     }


# # ── Field Accuracy ───────────────────────────────────────────────────────────

# def compute_field_accuracy(gt_list: list[dict], pred_list: list[dict]) -> dict:
#     """
#     Per-field exact-match accuracy for closed-set fields.
#     Gives a quick sense of structured prediction quality.
#     """
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
#     counts  = {f'{a}/{b}': {'correct': 0, 'total': 0} for a, b in field_paths}
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
# # Main
# # =============================================================================

# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate Multi-Modal VLM')
#     parser.add_argument('--model_path',   type=str, required=True,
#                         help='Path to saved model (e.g. save/multimodal_vlm/final_model)')
#     parser.add_argument('--data_root',    type=str, required=True,
#                         help='Path to UNISCP dataset root')
#     parser.add_argument('--sequences',    nargs='+', default=None,
#                         help='Sequences to evaluate (default: all). '
#                              f'Available: {ALL_SEQUENCES}')
#     parser.add_argument('--num_samples',  type=int, default=None,
#                         help='Max samples per sequence (default: all)')
#     parser.add_argument('--output_file',  type=str, default='eval_results.json',
#                         help='Where to save detailed results JSON')
#     parser.add_argument('--device',       type=str, default='cuda')
#     parser.add_argument('--max_new_tokens', type=int, default=512)
#     parser.add_argument('--bert_lang',    type=str, default='en',
#                         help='Language for BERTScore (en / zh / ...)')
#     parser.add_argument('--no_bert',      action='store_true',
#                         help='Skip BERTScore (faster, saves VRAM)')
#     parser.add_argument('--predictions_cache', type=str, default=None,
#                         help='Path to save/load inference results (skip re-running inference if exists)')
#     parser.add_argument('--batch_size',   type=int, default=1,
#                         help='Inference batch size (default: 1). Increase for faster evaluation if VRAM allows.')
#     return parser.parse_args()


# def main():
#     args   = parse_args()
#     device = args.device if torch.cuda.is_available() else 'cpu'

#     # ── sequences ────────────────────────────────────────────────────────────
#     sequences = args.sequences if args.sequences else ALL_SEQUENCES
#     logger.info(f"Evaluating sequences: {sequences}")

#     # ── model ────────────────────────────────────────────────────────────────
#     logger.info(f"Loading model from {args.model_path} ...")
#     config    = MultiModalVLMConfig.from_pretrained(args.model_path)
#     model     = MultiModalVLM(config)

#     # =====================================================================
#     # FIX: Load trained checkpoint weights into the model
#     # The original code stopped here — it only loaded the config and created
#     # a fresh model.  Q-Former, LiDAR/Radar encoders, and fine-tuned LLM
#     # weights were all random.  This is why BLEU-1 was ~10.
#     # =====================================================================
#     load_checkpoint_weights(model, args.model_path)

#     model = model.to(device)
#     model.eval()

#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

#     if '<|image_pad|>' not in tokenizer.get_vocab():
#         tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
#         model.llm_model.resize_token_embeddings(len(tokenizer))
#         model.tokenizer = tokenizer

#     # ── load from cache if available ────────────────────────────────────────
#     if args.predictions_cache and Path(args.predictions_cache).exists():
#         logger.info(f"Loading inference cache from {args.predictions_cache} (skipping inference) ...")
#         with open(args.predictions_cache, 'r', encoding='utf-8') as f:
#             cache = json.load(f)
#         all_gt_texts        = cache['all_gt_texts']
#         all_pred_texts      = cache['all_pred_texts']
#         all_gt_dicts        = cache['all_gt_dicts']
#         all_pred_dicts      = cache['all_pred_dicts']
#         per_sample_results  = cache['per_sample_results']
#         logger.info(f"Loaded {len(all_gt_texts)} cached samples, skipping to metrics ...")
#         _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
#                      all_pred_dicts, per_sample_results, sequences, device)
#         return

#     # ── build sample list ────────────────────────────────────────────────────
#     calib   = get_rural_calibration()
#     dataset = UniscpDataset(
#         data_root=args.data_root,
#         sequences=sequences,
#         tokenizer=tokenizer,
#         processor=processor,
#     )
#     samples = dataset.samples  # raw records (no preprocessing)

#     if args.num_samples:
#         # sample evenly across sequences
#         per_seq   = args.num_samples
#         seq_groups: dict[str, list] = {}
#         for s in samples:
#             seq_groups.setdefault(s['seq'], []).append(s)
#         samples = []
#         for seq in sequences:
#             group = seq_groups.get(seq, [])
#             step  = max(1, len(group) // per_seq)
#             samples += group[::step][:per_seq]

#     logger.info(f"Total samples to evaluate: {len(samples)}")

#     # ── run inference ────────────────────────────────────────────────────────
#     all_gt_texts   = []
#     all_pred_texts = []
#     all_gt_dicts   = []
#     all_pred_dicts = []
#     per_sample_results = []

#     # ── batch inference loop ────────────────────────────────────────────────
#     batch_size = args.batch_size
#     for batch_start in tqdm(range(0, len(samples), batch_size), desc='Inference'):
#         batch_records = samples[batch_start: batch_start + batch_size]

#         # load each sample in the mini-batch
#         batch_images, batch_lidar, batch_radar, batch_gt = [], [], [], []
#         valid_records = []
#         for record in batch_records:
#             try:
#                 image, lidar_pts, radar_pts, gt_caption = load_sample_from_record(record, calib)
#                 batch_images.append(image)
#                 batch_lidar.append(lidar_pts)
#                 batch_radar.append(radar_pts)
#                 batch_gt.append(gt_caption)
#                 valid_records.append(record)
#             except Exception as e:
#                 logger.warning(f"Load failed [{record['seq']} {record['img_idx']}]: {e}")

#         if not batch_images:
#             continue

#         pred_dicts, raw_texts = run_inference_batch(
#             model, tokenizer, processor,
#             batch_images, batch_lidar, batch_radar,
#             num_query_tokens=config.num_query_tokens,
#             device=device,
#             max_new_tokens=args.max_new_tokens,
#         )

#         for record, gt_caption, pred_dict, raw_text in zip(
#                 valid_records, batch_gt, pred_dicts, raw_texts):
#             gt_text   = flatten_json_to_text(gt_caption)
#             pred_text = flatten_json_to_text(pred_dict) if pred_dict else raw_text

#             all_gt_texts.append(gt_text)
#             all_pred_texts.append(pred_text)
#             all_gt_dicts.append(gt_caption)
#             all_pred_dicts.append(pred_dict)

#             per_sample_results.append({
#                 'seq':        record['seq'],
#                 'img_idx':    record['img_idx'],
#                 'gt':         gt_caption,
#                 'pred':       pred_dict,
#                 'raw_output': raw_text,
#             })

#     # ── save inference cache ────────────────────────────────────────────────
#     if args.predictions_cache:
#         cache_path = Path(args.predictions_cache)
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#         cache_data = {
#             'all_gt_texts':   all_gt_texts,
#             'all_pred_texts': all_pred_texts,
#             'all_gt_dicts':   all_gt_dicts,
#             'all_pred_dicts': all_pred_dicts,
#             'per_sample_results': per_sample_results,
#         }
#         with open(cache_path, 'w', encoding='utf-8') as f:
#             json.dump(cache_data, f, ensure_ascii=False)
#         logger.info(f"Inference cache saved to {cache_path}")

#     if not all_gt_texts:
#         logger.error("No samples were successfully evaluated.")
#         return

#     _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
#                  all_pred_dicts, per_sample_results, sequences, device)


# def _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
#                  all_pred_dicts, per_sample_results, sequences, device):
#     # ── compute metrics ──────────────────────────────────────────────────────
#     logger.info("Computing BLEU scores ...")
#     bleu = compute_bleu_scores(all_gt_texts, all_pred_texts)

#     logger.info("Computing METEOR score ...")
#     meteor = compute_meteor_score(all_gt_texts, all_pred_texts)

#     logger.info("Computing ROUGE-L score ...")
#     rouge_l = compute_rouge_l(all_gt_texts, all_pred_texts)

#     bert = {}
#     if not args.no_bert:
#         logger.info("Computing BERTScore (this may take a while) ...")
#         bert = compute_bert_scores(all_gt_texts, all_pred_texts,
#                                    lang=args.bert_lang, device=device)

#     logger.info("Computing field accuracy ...")
#     field_acc = compute_field_accuracy(all_gt_dicts, all_pred_dicts)

#     # ── per-sequence breakdown ────────────────────────────────────────────────
#     seq_metrics = {}
#     for seq in sequences:
#         idxs = [i for i, r in enumerate(per_sample_results) if r['seq'] == seq]
#         if not idxs:
#             continue
#         seq_gt   = [all_gt_texts[i]   for i in idxs]
#         seq_pred = [all_pred_texts[i] for i in idxs]
#         seq_metrics[seq] = compute_bleu_scores(seq_gt, seq_pred)
#         seq_metrics[seq].update(compute_meteor_score(seq_gt, seq_pred))
#         seq_metrics[seq].update(compute_rouge_l(seq_gt, seq_pred))
#         seq_metrics[seq]['n_samples'] = len(idxs)
#         if not args.no_bert:
#             seq_metrics[seq].update(
#                 compute_bert_scores(seq_gt, seq_pred,
#                                     lang=args.bert_lang, device=device)
#             )

#     # ── print summary ────────────────────────────────────────────────────────
#     print("\n" + "=" * 60)
#     print("EVALUATION SUMMARY")
#     print("=" * 60)
#     print(f"  Sequences : {sequences}")
#     print(f"  Samples   : {len(all_gt_texts)}")
#     print()
#     print("── BLEU Scores (corpus-level) ──────────────────────────────")
#     for k, v in bleu.items():
#         print(f"  {k}: {v:.4f}")
#     print()
#     print("── METEOR Score (avg sentence-level) ───────────────────────")
#     for k, v in meteor.items():
#         print(f"  {k}: {v:.4f}")
#     print()
#     print("── ROUGE-L Score (avg sentence-level) ──────────────────────")
#     for k, v in rouge_l.items():
#         print(f"  {k}: {v:.4f}")
#     if bert:
#         print()
#         print("── BERTScore ───────────────────────────────────────────────")
#         for k, v in bert.items():
#             print(f"  {k}: {v:.4f}")
#     print()
#     print("── Field Accuracy (exact match) ────────────────────────────")
#     for k, v in field_acc.items():
#         print(f"  {k}: {v:.4f}")
#     print()
#     print("── Per-Sequence BLEU-4 / METEOR / ROUGE-L-F ─────────────────")
#     for seq, m in seq_metrics.items():
#         print(f"  {seq:25s}: BLEU-4={m['BLEU-4']:.4f}  "
#               f"METEOR={m['METEOR']:.4f}  "
#               f"ROUGE-L-F={m['ROUGE-L-F']:.4f}  "
#               f"n={m['n_samples']}")
#     print("=" * 60)

#     # ── save results ─────────────────────────────────────────────────────────
#     output_path = Path(args.output_file)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     results = {
#         'sequences': sequences,
#         'n_samples':  len(all_gt_texts),
#         'bleu':       bleu,
#         'meteor':     meteor,
#         'rouge_l':    rouge_l,
#         'bert_score': bert,
#         'field_accuracy': field_acc,
#         'per_sequence':   seq_metrics,
#         'per_sample':     per_sample_results,
#     }
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     logger.info(f"Detailed results saved to {output_path}")


# if __name__ == '__main__':
#     main()

"""
Evaluation script for Multi-Modal VLM on UNISCP dataset.

Metrics:
  - BLEU-1/2/3/4  (hand-written, corpus-level)
  - METEOR        (hand-written, exact-match, sentence-level avg)
  - ROUGE-L       (hand-written, LCS-based, sentence-level avg)
  - CIDEr-D       (hand-written, TF-IDF cosine, corpus-level)
  - BERTScore     (bert-score library)

Usage:
    python evaluate.py \
        --model_path save/multimodal_vlm/final_model \
        --data_root ./UNISCP \
        --sequences RURAL_A0 RURAL_A1 \
        --num_samples 200 \
        --output_file results/eval_results.json

Install dependencies if needed:
    pip install bert-score --break-system-packages
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
from collections import Counter, defaultdict
from bert_score import score as bert_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusionxdrive.model_planning import MultiModalVLM, MultiModalVLMConfig
from fusionxdrive.dataset import (
    UniscpDataset, read_pcd, get_rural_calibration,
    load_timestamps, find_nearest_idx,
    SYSTEM_PROMPT, USER_PROMPT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

RURAL_SEQUENCES = ['RURAL_A0', 'RURAL_A1', 'RURAL_A2', 'RURAL_B0', 'RURAL_B1', 'RURAL_B2']
OTHER_SEQUENCES = ['FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO', 'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE']
ALL_SEQUENCES   = RURAL_SEQUENCES + OTHER_SEQUENCES


# =============================================================================
# Data loading (reuse inference.py logic, but for whole sequences)
# =============================================================================

def load_sample_from_record(record: dict, calib, max_lidar=40000, max_radar=16000):
    """Load image, lidar, radar and ground truth caption from a sample record."""
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

    # Ground truth caption
    with open(record['caption_path'], 'r') as f:
        gt_caption = json.load(f)

    return image, lidar, radar, gt_caption


# =============================================================================
# Checkpoint weight loading  (THE KEY FIX)
# =============================================================================

def load_checkpoint_weights(model, model_path: str):
    """
    Load trained checkpoint weights into the model.

    This is the critical fix: the original evaluate.py only loaded the config
    from model_path and re-initialized the model from scratch.  The Q-Former,
    LiDAR/Radar encoders, and any fine-tuned LLM weights were never loaded,
    so evaluation was running on a randomly-initialized bridge.

    This function mirrors wild-drive/validate.py lines 392-412.
    """
    ckpt_path = Path(model_path)

    # Collect all weight files (.safetensors and .bin)
    weight_files = sorted(ckpt_path.glob("*.safetensors")) + sorted(ckpt_path.glob("*.bin"))

    if not weight_files:
        logger.warning(
            f"No weight files (.safetensors / .bin) found in {ckpt_path}. "
            f"The model will use random weights for the bridge layers!"
        )
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
            # Skip non-dict files (TrainingArguments, optimizer states, etc.)

    if not state_dict:
        logger.warning(
            f"Weight files were found but no valid state_dict could be loaded from {ckpt_path}."
        )
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Checkpoint loaded from {ckpt_path}")
    logger.info(f"  Total keys in checkpoint : {len(state_dict)}")
    logger.info(f"  Missing keys  (in model but not in ckpt): {len(missing)}")
    logger.info(f"  Unexpected keys (in ckpt but not in model): {len(unexpected)}")

    # Print details for debugging (only first 10)
    if missing:
        logger.info(f"  First missing keys: {missing[:10]}")
    if unexpected:
        logger.info(f"  First unexpected keys: {unexpected[:10]}")


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
                        num_query_tokens=64, device='cuda', max_new_tokens=512):
    """
    Run inference on a batch of samples.

    Args:
        images     : list of PIL.Image  (length B)
        lidar_list : list of np.ndarray (length B)
        radar_list : list of np.ndarray (length B)
    Returns:
        predictions : list of dicts  (length B)
        raw_texts   : list of str    (length B)
    """
    prompt_str = build_prompt(tokenizer, num_query_tokens)

    # image: stack into [B, 3, H, W]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

    # text: pad to same length (left-pad for decoder-only generation)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    B = len(images)
    encoded = tokenizer([prompt_str] * B, return_tensors="pt", padding=True)
    prompt_ids = encoded["input_ids"].to(device)

    # point clouds: keep as list (model expects List[Tensor])
    lidar_tensors = [torch.from_numpy(p).float().to(device) for p in lidar_list]
    radar_tensors = [torch.from_numpy(p).float().to(device) for p in radar_list]

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            lidar_points=lidar_tensors,
            radar_points=radar_tensors,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )

    predictions, raw_texts = [], []
    for i in range(B):
        text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        pred, raw = parse_prediction(text)
        predictions.append(pred)
        raw_texts.append(raw)

    return predictions, raw_texts


# =============================================================================
# Metrics
# =============================================================================

def flatten_json_to_text(obj) -> str:
    """Recursively flatten JSON values to a string for BLEU scoring.

    Only extracts VALUES (not keys) to match the original validate.py behavior
    where GT and pred are raw answer strings without structural key names.
    """
    if isinstance(obj, dict):
        return ' '.join(flatten_json_to_text(v) for v in obj.values())
    elif isinstance(obj, list):
        return ' '.join(flatten_json_to_text(i) for i in obj)
    else:
        return str(obj)


# ── BLEU ─────────────────────────────────────────────────────────────────────

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def _modified_precision(references, hypotheses, n):
    """Corpus-level modified n-gram precision."""
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


def compute_bleu_scores(references: list[str], hypotheses: list[str]) -> dict:
    """
    Compute corpus-level BLEU-1/2/3/4 using the same method as validate.py:
    - Simple .split() tokenization (no nltk word_tokenize)
    - +1e-5 Laplace smoothing
    - Closest-reference brevity penalty
    """
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


# ── METEOR (self-contained, no nltk dependency) ─────────────────────────────

def _meteor_exact_matches(ref_tokens, hyp_tokens):
    """Find exact unigram matches (greedy, each token matched at most once)."""
    ref_available = list(range(len(ref_tokens)))
    matches = []
    for h_idx, h_tok in enumerate(hyp_tokens):
        for r_idx in ref_available:
            if ref_tokens[r_idx] == h_tok:
                matches.append((h_idx, r_idx))
                ref_available.remove(r_idx)
                break
    return matches


def _meteor_chunks(matches):
    """Count the number of contiguous chunks in matched pairs."""
    if not matches:
        return 0
    sorted_matches = sorted(matches, key=lambda x: x[0])
    chunks = 1
    for i in range(1, len(sorted_matches)):
        if (sorted_matches[i][0] != sorted_matches[i-1][0] + 1 or
                sorted_matches[i][1] != sorted_matches[i-1][1] + 1):
            chunks += 1
    return chunks


def _meteor_sentence(ref_tokens, hyp_tokens, alpha=0.9, beta=3.0, gamma=0.5):
    """
    Compute sentence-level METEOR (exact match only, no stemming/synonyms).

    Parameters follow Banerjee & Lavie (2005) defaults:
      alpha = 0.9  (relative weight of precision vs recall in harmonic mean)
      beta  = 3.0  (shape of fragmentation penalty)
      gamma = 0.5  (max fragmentation penalty)
    """
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0

    matches = _meteor_exact_matches(ref_tokens, hyp_tokens)
    m = len(matches)

    if m == 0:
        return 0.0

    p = m / len(hyp_tokens)
    r = m / len(ref_tokens)

    f_mean = (p * r) / (alpha * p + (1 - alpha) * r)

    chunks = _meteor_chunks(matches)
    frag = gamma * (chunks / m) ** beta
    score = f_mean * (1 - frag)

    return score


def compute_meteor_score(references: list[str], hypotheses: list[str]) -> dict:
    """
    Compute corpus-level METEOR (average of sentence-level scores).

    Exact-match only variant — no stemming or WordNet synonyms, but
    avoids all nltk/sqlite3 dependencies.
    Tokenization: simple .split() to stay consistent with BLEU above.
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        scores.append(_meteor_sentence(ref_tokens, hyp_tokens))

    avg = sum(scores) / len(scores) if scores else 0.0
    return {'METEOR': round(avg, 4)}


# ── ROUGE-L (self-contained, LCS-based) ─────────────────────────────────────

def _lcs_length(x: list, y: list) -> int:
    """Compute length of Longest Common Subsequence via DP (space-optimized)."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(references: list[str], hypotheses: list[str],
                    beta: float = 1.2) -> dict:
    """
    Compute corpus-level ROUGE-L (sentence-level, then averaged).

    ROUGE-L uses the Longest Common Subsequence (LCS) between reference and
    hypothesis to compute precision, recall, and F-measure.

    Args:
        beta: Controls the relative importance of recall vs precision in F-measure.
              beta > 1 favors recall (default 1.2 follows the original Lin 2004 paper).

    Returns:
        dict with ROUGE-L-P, ROUGE-L-R, ROUGE-L-F keys.
    """
    precisions, recalls, f_scores = [], [], []

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()

        if len(hyp_tokens) == 0 and len(ref_tokens) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f_scores.append(1.0)
            continue

        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f_scores.append(0.0)
            continue

        lcs = _lcs_length(ref_tokens, hyp_tokens)
        p = lcs / len(hyp_tokens)
        r = lcs / len(ref_tokens)

        if p == 0 and r == 0:
            f = 0.0
        else:
            beta_sq = beta ** 2
            f = (1 + beta_sq) * p * r / (beta_sq * p + r)

        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

    n = len(f_scores) if f_scores else 1
    return {
        'ROUGE-L-P': round(sum(precisions) / n, 4),
        'ROUGE-L-R': round(sum(recalls) / n, 4),
        'ROUGE-L-F': round(sum(f_scores) / n, 4),
    }


# ── CIDEr-D (self-contained, TF-IDF cosine) ─────────────────────────────────

def _cider_cook_ngrams(tokens, n):
    """Extract n-grams from token list and return Counter."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _cider_compute_doc_freq(references_list):
    """
    Compute document frequency: how many reference *images* contain each n-gram.

    Args:
        references_list: list of list[str], outer = per image, inner = multiple refs.
    Returns:
        doc_freq: Counter mapping n-gram -> number of images containing it.
        num_docs: total number of images.
    """
    doc_freq = Counter()
    for refs in references_list:
        # union of n-grams across all references for this image
        seen = set()
        for ref in refs:
            tokens = ref.lower().split()
            for n in range(1, 5):  # 1-gram to 4-gram
                for ng in _cider_cook_ngrams(tokens, n):
                    seen.add(ng)
        for ng in seen:
            doc_freq[ng] += 1
    return doc_freq, len(references_list)


def _cider_tfidf_vec(tokens, doc_freq, num_docs, n):
    """
    Build a TF-IDF vector (as dict) for n-grams of order `n`.

    TF  = count(ngram) / total_ngrams_in_sentence
    IDF = log( (num_docs + 1) / (df(ngram) + 1) )     # +1 smoothing
    """
    ngram_counts = _cider_cook_ngrams(tokens, n)
    total = sum(ngram_counts.values())
    if total == 0:
        return {}
    vec = {}
    for ng, cnt in ngram_counts.items():
        tf = cnt / total
        idf = math.log((num_docs + 1.0) / (doc_freq.get(ng, 0) + 1.0))
        vec[ng] = tf * idf
    return vec


def _cider_cosine_sim(vec_a, vec_b):
    """Cosine similarity between two sparse vectors (dicts)."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_cider_score(references: list[str], hypotheses: list[str],
                        sigma: float = 6.0) -> dict:
    """
    Compute corpus-level CIDEr-D score.

    Follows Vedantam et al. (2015) "CIDEr: Consensus-based Image Description
    Evaluation":
      1. Compute document frequency (IDF) from all references.
      2. For each image, build TF-IDF vectors for n=1..4.
      3. CIDEr_n = cosine_sim(hyp_vec, ref_vec) with Gaussian length penalty.
      4. CIDEr-D = (1/4) * sum(CIDEr_n for n in 1..4) * 10.

    Each image has a single reference in this dataset. The score is scaled by 10
    following the standard CIDEr convention.

    Args:
        references:  list of reference strings (one per image).
        hypotheses:  list of hypothesis strings (one per image).
        sigma:       Gaussian length penalty parameter (default 6.0).

    Returns:
        dict with 'CIDEr-D' key.
    """
    # Wrap each reference as a list (CIDEr supports multiple refs per image)
    refs_list = [[ref] for ref in references]

    # Step 1: document frequency from all references
    doc_freq, num_docs = _cider_compute_doc_freq(refs_list)

    scores = []
    for refs, hyp in zip(refs_list, hypotheses):
        hyp_tokens = hyp.lower().split()

        # Accumulate CIDEr_n for n = 1..4
        cider_n_sum = 0.0
        for n in range(1, 5):
            hyp_vec = _cider_tfidf_vec(hyp_tokens, doc_freq, num_docs, n)

            # Average cosine similarity across references for this image
            sim_sum = 0.0
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_vec = _cider_tfidf_vec(ref_tokens, doc_freq, num_docs, n)

                cos_sim = _cider_cosine_sim(hyp_vec, ref_vec)

                # Gaussian length penalty: exp( -(len_hyp - len_ref)^2 / (2*sigma^2) )
                len_diff = len(hyp_tokens) - len(ref_tokens)
                length_penalty = math.exp(-(len_diff ** 2) / (2.0 * sigma ** 2))

                sim_sum += cos_sim * length_penalty

            cider_n_sum += sim_sum / len(refs)

        # Average over n=1..4, scale by 10
        scores.append((cider_n_sum / 4.0) * 10.0)

    avg_cider = sum(scores) / len(scores) if scores else 0.0
    return {'CIDEr-D': round(avg_cider, 4)}


# ── BERTScore ────────────────────────────────────────────────────────────────

def compute_bert_scores(references: list[str], hypotheses: list[str],
                        lang='en', device='cuda') -> dict:
    """Compute BERTScore P/R/F1."""
    P, R, F1 = bert_score(
        hypotheses, references,
        lang=lang,
        device=device,
        verbose=False,
    )
    return {
        'BERTScore-P':  round(P.mean().item(),  4),
        'BERTScore-R':  round(R.mean().item(),  4),
        'BERTScore-F1': round(F1.mean().item(), 4),
    }


# ── Field Accuracy ───────────────────────────────────────────────────────────

def compute_field_accuracy(gt_list: list[dict], pred_list: list[dict]) -> dict:
    """
    Per-field exact-match accuracy for closed-set fields.
    Gives a quick sense of structured prediction quality.
    """
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
    counts  = {f'{a}/{b}': {'correct': 0, 'total': 0} for a, b in field_paths}
    overall_correct = overall_total = 0

    for gt, pred in zip(gt_list, pred_list):
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
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Modal VLM')
    parser.add_argument('--model_path',   type=str, required=True,
                        help='Path to saved model (e.g. save/multimodal_vlm/final_model)')
    parser.add_argument('--data_root',    type=str, required=True,
                        help='Path to UNISCP dataset root')
    parser.add_argument('--sequences',    nargs='+', default=None,
                        help='Sequences to evaluate (default: all). '
                             f'Available: {ALL_SEQUENCES}')
    parser.add_argument('--num_samples',  type=int, default=None,
                        help='Max samples per sequence (default: all)')
    parser.add_argument('--output_file',  type=str, default='eval_results.json',
                        help='Where to save detailed results JSON')
    parser.add_argument('--device',       type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--bert_lang',    type=str, default='en',
                        help='Language for BERTScore (en / zh / ...)')
    parser.add_argument('--no_bert',      action='store_true',
                        help='Skip BERTScore (faster, saves VRAM)')
    parser.add_argument('--predictions_cache', type=str, default=None,
                        help='Path to save/load inference results (skip re-running inference if exists)')
    parser.add_argument('--batch_size',   type=int, default=1,
                        help='Inference batch size (default: 1). Increase for faster evaluation if VRAM allows.')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    # ── sequences ────────────────────────────────────────────────────────────
    sequences = args.sequences if args.sequences else ALL_SEQUENCES
    logger.info(f"Evaluating sequences: {sequences}")

    # ── model ────────────────────────────────────────────────────────────────
    logger.info(f"Loading model from {args.model_path} ...")
    config    = MultiModalVLMConfig.from_pretrained(args.model_path)
    model     = MultiModalVLM(config)

    # =====================================================================
    # FIX: Load trained checkpoint weights into the model
    # The original code stopped here — it only loaded the config and created
    # a fresh model.  Q-Former, LiDAR/Radar encoders, and fine-tuned LLM
    # weights were all random.  This is why BLEU-1 was ~10.
    # =====================================================================
    load_checkpoint_weights(model, args.model_path)

    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoImageProcessor.from_pretrained(config.vision_model_path)

    if '<|image_pad|>' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_pad|>']})
        model.llm_model.resize_token_embeddings(len(tokenizer))
        model.tokenizer = tokenizer

    # ── load from cache if available ────────────────────────────────────────
    if args.predictions_cache and Path(args.predictions_cache).exists():
        logger.info(f"Loading inference cache from {args.predictions_cache} (skipping inference) ...")
        with open(args.predictions_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        all_gt_texts        = cache['all_gt_texts']
        all_pred_texts      = cache['all_pred_texts']
        all_gt_dicts        = cache['all_gt_dicts']
        all_pred_dicts      = cache['all_pred_dicts']
        per_sample_results  = cache['per_sample_results']
        logger.info(f"Loaded {len(all_gt_texts)} cached samples, skipping to metrics ...")
        _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
                     all_pred_dicts, per_sample_results, sequences, device)
        return

    # ── build sample list ────────────────────────────────────────────────────
    calib   = get_rural_calibration()
    dataset = UniscpDataset(
        data_root=args.data_root,
        sequences=sequences,
        tokenizer=tokenizer,
        processor=processor,
    )
    samples = dataset.samples  # raw records (no preprocessing)

    if args.num_samples:
        # sample evenly across sequences
        per_seq   = args.num_samples
        seq_groups: dict[str, list] = {}
        for s in samples:
            seq_groups.setdefault(s['seq'], []).append(s)
        samples = []
        for seq in sequences:
            group = seq_groups.get(seq, [])
            step  = max(1, len(group) // per_seq)
            samples += group[::step][:per_seq]

    logger.info(f"Total samples to evaluate: {len(samples)}")

    # ── run inference ────────────────────────────────────────────────────────
    all_gt_texts   = []
    all_pred_texts = []
    all_gt_dicts   = []
    all_pred_dicts = []
    per_sample_results = []

    # ── batch inference loop ────────────────────────────────────────────────
    batch_size = args.batch_size
    for batch_start in tqdm(range(0, len(samples), batch_size), desc='Inference'):
        batch_records = samples[batch_start: batch_start + batch_size]

        # load each sample in the mini-batch
        batch_images, batch_lidar, batch_radar, batch_gt = [], [], [], []
        valid_records = []
        for record in batch_records:
            try:
                image, lidar_pts, radar_pts, gt_caption = load_sample_from_record(record, calib)
                batch_images.append(image)
                batch_lidar.append(lidar_pts)
                batch_radar.append(radar_pts)
                batch_gt.append(gt_caption)
                valid_records.append(record)
            except Exception as e:
                logger.warning(f"Load failed [{record['seq']} {record['img_idx']}]: {e}")

        if not batch_images:
            continue

        pred_dicts, raw_texts = run_inference_batch(
            model, tokenizer, processor,
            batch_images, batch_lidar, batch_radar,
            num_query_tokens=config.num_query_tokens,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        for record, gt_caption, pred_dict, raw_text in zip(
                valid_records, batch_gt, pred_dicts, raw_texts):
            gt_text   = flatten_json_to_text(gt_caption)
            pred_text = flatten_json_to_text(pred_dict) if pred_dict else raw_text

            all_gt_texts.append(gt_text)
            all_pred_texts.append(pred_text)
            all_gt_dicts.append(gt_caption)
            all_pred_dicts.append(pred_dict)

            per_sample_results.append({
                'seq':        record['seq'],
                'img_idx':    record['img_idx'],
                'gt':         gt_caption,
                'pred':       pred_dict,
                'raw_output': raw_text,
            })

    # ── save inference cache ────────────────────────────────────────────────
    if args.predictions_cache:
        cache_path = Path(args.predictions_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'all_gt_texts':   all_gt_texts,
            'all_pred_texts': all_pred_texts,
            'all_gt_dicts':   all_gt_dicts,
            'all_pred_dicts': all_pred_dicts,
            'per_sample_results': per_sample_results,
        }
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        logger.info(f"Inference cache saved to {cache_path}")

    if not all_gt_texts:
        logger.error("No samples were successfully evaluated.")
        return

    _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
                 all_pred_dicts, per_sample_results, sequences, device)


def _run_metrics(args, all_gt_texts, all_pred_texts, all_gt_dicts,
                 all_pred_dicts, per_sample_results, sequences, device):
    # ── compute metrics ──────────────────────────────────────────────────────
    logger.info("Computing BLEU scores ...")
    bleu = compute_bleu_scores(all_gt_texts, all_pred_texts)

    logger.info("Computing METEOR score ...")
    meteor = compute_meteor_score(all_gt_texts, all_pred_texts)

    logger.info("Computing ROUGE-L score ...")
    rouge_l = compute_rouge_l(all_gt_texts, all_pred_texts)

    logger.info("Computing CIDEr-D score ...")
    cider = compute_cider_score(all_gt_texts, all_pred_texts)

    bert = {}
    if not args.no_bert:
        logger.info("Computing BERTScore (this may take a while) ...")
        bert = compute_bert_scores(all_gt_texts, all_pred_texts,
                                   lang=args.bert_lang, device=device)

    logger.info("Computing field accuracy ...")
    field_acc = compute_field_accuracy(all_gt_dicts, all_pred_dicts)

    # ── per-sequence breakdown ────────────────────────────────────────────────
    seq_metrics = {}
    for seq in sequences:
        idxs = [i for i, r in enumerate(per_sample_results) if r['seq'] == seq]
        if not idxs:
            continue
        seq_gt   = [all_gt_texts[i]   for i in idxs]
        seq_pred = [all_pred_texts[i] for i in idxs]
        seq_metrics[seq] = compute_bleu_scores(seq_gt, seq_pred)
        seq_metrics[seq].update(compute_meteor_score(seq_gt, seq_pred))
        seq_metrics[seq].update(compute_rouge_l(seq_gt, seq_pred))
        seq_metrics[seq].update(compute_cider_score(seq_gt, seq_pred))
        seq_metrics[seq]['n_samples'] = len(idxs)
        if not args.no_bert:
            seq_metrics[seq].update(
                compute_bert_scores(seq_gt, seq_pred,
                                    lang=args.bert_lang, device=device)
            )

    # ── print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Sequences : {sequences}")
    print(f"  Samples   : {len(all_gt_texts)}")
    print()
    print("── BLEU Scores (corpus-level) ──────────────────────────────────────")
    for k, v in bleu.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("── METEOR Score (avg sentence-level) ───────────────────────────────")
    for k, v in meteor.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("── ROUGE-L Score (avg sentence-level) ──────────────────────────────")
    for k, v in rouge_l.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("── CIDEr-D Score (corpus-level) ────────────────────────────────────")
    for k, v in cider.items():
        print(f"  {k}: {v:.4f}")
    if bert:
        print()
        print("── BERTScore ───────────────────────────────────────────────────────")
        for k, v in bert.items():
            print(f"  {k}: {v:.4f}")
    print()
    print("── Field Accuracy (exact match) ────────────────────────────────────")
    for k, v in field_acc.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("── Per-Sequence Breakdown ──────────────────────────────────────────")
    for seq, m in seq_metrics.items():
        print(f"  {seq:25s}: BLEU-4={m['BLEU-4']:.4f}  "
              f"METEOR={m['METEOR']:.4f}  "
              f"ROUGE-L-F={m['ROUGE-L-F']:.4f}  "
              f"CIDEr-D={m['CIDEr-D']:.4f}  "
              f"n={m['n_samples']}")
    print("=" * 70)

    # ── save results ─────────────────────────────────────────────────────────
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'sequences': sequences,
        'n_samples':  len(all_gt_texts),
        'bleu':       bleu,
        'meteor':     meteor,
        'rouge_l':    rouge_l,
        'cider':      cider,
        'bert_score': bert,
        'field_accuracy': field_acc,
        'per_sequence':   seq_metrics,
        'per_sample':     per_sample_results,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed results saved to {output_path}")


if __name__ == '__main__':
    main()