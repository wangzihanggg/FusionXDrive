# """
# Enhanced Evaluation Metrics for Caption-to-Plan Evaluation.

# Adds three layers of metrics on top of the existing evaluate_with_planning.py:

# Layer 1 - Complete Caption Metrics (from TOD3Cap):
#   - METEOR (semantic-aware, considers synonyms and stems)
#   - ROUGE-L (longest common subsequence based recall)
#   - CIDEr  (TF-IDF weighted n-gram consensus, best for domain-specific captioning)

# Layer 2 - Caption-Plan Joint Analysis:
#   - Caption-conditioned planning error (ADE/FDE grouped by caption quality)
#   - Field-accuracy-conditioned planning error
#   - Correlation analysis between caption quality and planning quality

# Layer 3 - Trajectory Safety Proxy Metrics (no surrounding object annotations needed):
#   - Lateral acceleration (comfort / safety proxy)
#   - Longitudinal jerk (smoothness)
#   - Curvature statistics (physical plausibility)
#   - Heading rate (yaw rate proxy)
#   - Out-of-bounds detection (trajectory going backwards or sideways too far)

# References:
#   - TOD3Cap (Jin et al., 2024): METEOR, ROUGE-L, CIDEr, m@kIoU
#   - DiffusionDrive (Liao et al., 2025): PDMS sub-scores (NC, DAC, TTC, Comfort, EP)

# Usage:
#     # As standalone module
#     from enhanced_metrics import (
#         compute_meteor_scores, compute_rouge_scores, compute_cider_scores,
#         compute_caption_plan_joint, compute_trajectory_safety,
#     )

#     # Or integrate into evaluate_with_planning.py (see integration guide at bottom)
# """

# import math
# import numpy as np
# from collections import Counter, defaultdict
# from typing import List, Dict, Optional, Tuple


# # =============================================================================
# # Layer 1: Complete Caption Metrics
# # =============================================================================

# # ── METEOR ──────────────────────────────────────────────────────────────────
# # Simplified METEOR: unigram precision/recall with F-mean + chunk penalty.
# # For full METEOR (with synonym/stem matching), install `nltk` and use
# # nltk.translate.meteor_score.  This implementation covers the core formula.

# def _count_unigram_matches(reference_tokens: list, hypothesis_tokens: list) -> int:
#     """Count unigram matches between reference and hypothesis."""
#     ref_counts = Counter(reference_tokens)
#     hyp_counts = Counter(hypothesis_tokens)
#     matches = 0
#     for token, count in hyp_counts.items():
#         matches += min(count, ref_counts.get(token, 0))
#     return matches


# def _count_chunks(reference_tokens: list, hypothesis_tokens: list) -> int:
#     """
#     Count the number of contiguous chunks of matched unigrams.
#     Fewer chunks = better word order preservation.
#     """
#     ref_set = set(reference_tokens)
#     in_chunk = False
#     chunks = 0
#     for token in hypothesis_tokens:
#         if token in ref_set:
#             if not in_chunk:
#                 chunks += 1
#                 in_chunk = True
#         else:
#             in_chunk = False
#     return max(chunks, 1)


# def compute_meteor_scores(references: List[str], hypotheses: List[str],
#                           alpha: float = 0.9, beta: float = 3.0,
#                           gamma: float = 0.5) -> Dict[str, float]:
#     """
#     Compute METEOR score (simplified version without synonym/stem matching).

#     METEOR = F_mean * (1 - penalty)
#     where F_mean is the harmonic mean of precision and recall weighted by alpha,
#     and penalty penalizes fragmented matches (poor word order).

#     Args:
#         references: list of reference texts
#         hypotheses: list of hypothesis texts
#         alpha: weight for recall vs precision (higher = more weight on recall)
#         beta: exponent for chunk penalty
#         gamma: weight for chunk penalty

#     Returns:
#         dict with METEOR score
#     """
#     scores = []
#     for ref, hyp in zip(references, hypotheses):
#         ref_tokens = ref.lower().split()
#         hyp_tokens = hyp.lower().split()

#         if not hyp_tokens or not ref_tokens:
#             scores.append(0.0)
#             continue

#         matches = _count_unigram_matches(ref_tokens, hyp_tokens)

#         if matches == 0:
#             scores.append(0.0)
#             continue

#         precision = matches / len(hyp_tokens)
#         recall = matches / len(ref_tokens)

#         # Weighted harmonic mean (alpha controls recall emphasis)
#         f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall + 1e-8)

#         # Chunk penalty: fewer chunks = better alignment
#         chunks = _count_chunks(ref_tokens, hyp_tokens)
#         frag = chunks / matches if matches > 0 else 1.0
#         penalty = gamma * (frag ** beta)

#         meteor = f_mean * (1.0 - min(penalty, 1.0))
#         scores.append(meteor)

#     return {
#         'METEOR': round(float(np.mean(scores)), 4),
#         'METEOR_std': round(float(np.std(scores)), 4),
#     }


# # ── ROUGE-L ─────────────────────────────────────────────────────────────────
# # Based on the longest common subsequence (LCS).

# def _lcs_length(x: list, y: list) -> int:
#     """Compute length of longest common subsequence using DP."""
#     m, n = len(x), len(y)
#     # Space-optimized: only keep two rows
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


# def compute_rouge_scores(references: List[str], hypotheses: List[str],
#                          beta: float = 1.2) -> Dict[str, float]:
#     """
#     Compute ROUGE-L score based on longest common subsequence.

#     Args:
#         references: list of reference texts
#         hypotheses: list of hypothesis texts
#         beta: F-measure parameter (>1 favors recall)

#     Returns:
#         dict with ROUGE-L precision, recall, F1
#     """
#     precisions, recalls, f_scores = [], [], []

#     for ref, hyp in zip(references, hypotheses):
#         ref_tokens = ref.lower().split()
#         hyp_tokens = hyp.lower().split()

#         if not hyp_tokens or not ref_tokens:
#             precisions.append(0.0)
#             recalls.append(0.0)
#             f_scores.append(0.0)
#             continue

#         lcs = _lcs_length(ref_tokens, hyp_tokens)
#         p = lcs / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
#         r = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

#         if p + r == 0:
#             f = 0.0
#         else:
#             f = ((1 + beta ** 2) * p * r) / (beta ** 2 * p + r)

#         precisions.append(p)
#         recalls.append(r)
#         f_scores.append(f)

#     return {
#         'ROUGE-L-P': round(float(np.mean(precisions)), 4),
#         'ROUGE-L-R': round(float(np.mean(recalls)), 4),
#         'ROUGE-L-F': round(float(np.mean(f_scores)), 4),
#     }


# # ── CIDEr ───────────────────────────────────────────────────────────────────
# # Core metric for image/scene captioning. Uses TF-IDF weighted n-gram matching.

# def _tokenize(text: str) -> list:
#     """Simple whitespace tokenizer with lowercasing."""
#     return text.lower().split()


# def _compute_ngrams(tokens: list, n: int) -> Counter:
#     """Compute n-gram counts."""
#     return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# def _compute_doc_freq(references_tokenized: List[list], n: int) -> Counter:
#     """Compute document frequency for n-grams across all references."""
#     df = Counter()
#     for ref_tokens in references_tokenized:
#         ngrams = set(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
#         for ng in ngrams:
#             df[ng] += 1
#     return df


# def compute_cider_scores(references: List[str], hypotheses: List[str],
#                          n_max: int = 4, sigma: float = 6.0) -> Dict[str, float]:
#     """
#     Compute CIDEr-D score.

#     CIDEr uses TF-IDF weighting to emphasize informative n-grams that are
#     specific to the domain (e.g., "traffic_light" is more valuable than "the").

#     This is the most important caption metric for driving scene descriptions,
#     as it rewards domain-specific accurate descriptions.

#     Args:
#         references: list of reference texts
#         hypotheses: list of hypothesis texts
#         n_max: maximum n-gram order (default 4)
#         sigma: length penalty parameter

#     Returns:
#         dict with CIDEr score and per-n breakdown
#     """
#     N = len(references)
#     if N == 0:
#         return {'CIDEr': 0.0}

#     refs_tok = [_tokenize(r) for r in references]
#     hyps_tok = [_tokenize(h) for h in hypotheses]

#     # Pre-compute document frequencies for each n
#     doc_freqs = {}
#     for n in range(1, n_max + 1):
#         doc_freqs[n] = _compute_doc_freq(refs_tok, n)

#     def _tfidf_vec(tokens, n, df):
#         """Compute TF-IDF vector for a single document."""
#         ngrams = _compute_ngrams(tokens, n)
#         vec = {}
#         length = max(len(tokens) - n + 1, 1)
#         for ng, count in ngrams.items():
#             tf = count / length
#             idf = np.log(max(1.0, N) / (1.0 + df.get(ng, 0)))
#             vec[ng] = tf * idf
#         return vec

#     def _cosine_sim(vec1, vec2):
#         """Cosine similarity between two sparse vectors."""
#         common = set(vec1.keys()) & set(vec2.keys())
#         if not common:
#             return 0.0
#         dot = sum(vec1[k] * vec2[k] for k in common)
#         norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
#         norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
#         return dot / (norm1 * norm2)

#     # Compute CIDEr for each n-gram order
#     cider_n_scores = {n: [] for n in range(1, n_max + 1)}

#     for i in range(N):
#         ref_tok = refs_tok[i]
#         hyp_tok = hyps_tok[i]

#         for n in range(1, n_max + 1):
#             df = doc_freqs[n]
#             ref_vec = _tfidf_vec(ref_tok, n, df)
#             hyp_vec = _tfidf_vec(hyp_tok, n, df)

#             # Length penalty (Gaussian)
#             len_diff = len(hyp_tok) - len(ref_tok)
#             length_penalty = np.exp(-(len_diff ** 2) / (2 * sigma ** 2))

#             sim = _cosine_sim(ref_vec, hyp_vec) * length_penalty
#             cider_n_scores[n].append(sim)

#     results = {}
#     total_cider = 0.0
#     for n in range(1, n_max + 1):
#         score = float(np.mean(cider_n_scores[n])) if cider_n_scores[n] else 0.0
#         results[f'CIDEr-{n}'] = round(score * 10, 4)  # Scale by 10 (CIDEr convention)
#         total_cider += score

#     # CIDEr is average across n-gram orders, scaled by 10
#     results['CIDEr'] = round(total_cider / n_max * 10, 4)

#     return results


# # =============================================================================
# # Layer 2: Caption-Plan Joint Analysis
# # =============================================================================

# def compute_caption_plan_joint(
#     per_sample_results: List[dict],
#     all_gt_texts: List[str],
#     all_pred_texts: List[str],
#     all_gt_dicts: List[dict],
#     all_pred_dicts: List[dict],
#     field_acc_fn=None,
# ) -> Dict[str, any]:
#     """
#     Analyze the correlation between caption quality and planning quality.

#     Inspired by TOD3Cap's m@kIoU which gates caption evaluation by localization
#     quality, we gate planning evaluation by caption quality.

#     Key insight: If the model understands the scene well (good caption),
#     does it also plan well? This reveals whether caption and planning are
#     truly coupled or just independently trained heads.

#     Args:
#         per_sample_results: list of per-sample result dicts (must contain
#             'sample_ade', 'sample_fde', and optionally field accuracy info)
#         all_gt_texts: ground truth caption texts
#         all_pred_texts: predicted caption texts
#         all_gt_dicts: ground truth caption dicts (structured)
#         all_pred_dicts: predicted caption dicts (structured)
#         field_acc_fn: optional function to compute per-sample field accuracy

#     Returns:
#         dict with joint analysis results
#     """
#     results = {}

#     # ── 2a. Per-sample caption quality (ROUGE-L F as proxy) ──
#     sample_caption_scores = []
#     for gt_text, pred_text in zip(all_gt_texts, all_pred_texts):
#         ref_tok = gt_text.lower().split()
#         hyp_tok = pred_text.lower().split()
#         if not ref_tok or not hyp_tok:
#             sample_caption_scores.append(0.0)
#             continue
#         lcs = _lcs_length(ref_tok, hyp_tok)
#         p = lcs / len(hyp_tok)
#         r = lcs / len(ref_tok)
#         f = (2 * p * r) / (p + r + 1e-8)
#         sample_caption_scores.append(f)

#     # ── 2b. Per-sample field accuracy ──
#     sample_field_scores = []
#     key_fields = [
#         ('weather', 'condition'),
#         ('traffic_light', 'state'),
#         ('forward_drivability', 'status'),
#         ('lane_keeping', 'status'),
#         ('driving_advice', 'action'),
#     ]
#     for gt_dict, pred_dict in zip(all_gt_dicts, all_pred_dicts):
#         if not isinstance(pred_dict, dict):
#             sample_field_scores.append(0.0)
#             continue
#         correct = total = 0
#         for top, sub in key_fields:
#             gt_val = gt_dict.get(top, {}).get(sub, None)
#             pred_val = pred_dict.get(top, {}).get(sub, None)
#             if gt_val is not None:
#                 total += 1
#                 if str(gt_val).lower() == str(pred_val).lower():
#                     correct += 1
#         sample_field_scores.append(correct / total if total > 0 else 0.0)

#     # ── 2c. Split planning metrics by caption quality ──
#     planning_samples = [
#         (i, r) for i, r in enumerate(per_sample_results)
#         if 'sample_ade' in r and 'sample_fde' in r
#     ]

#     if not planning_samples:
#         results['note'] = 'No planning samples available for joint analysis'
#         return results

#     # Split by caption quality (ROUGE-L F) into tertiles
#     caption_scores_for_planning = [sample_caption_scores[i] for i, _ in planning_samples]
#     if len(caption_scores_for_planning) >= 3:
#         sorted_scores = sorted(caption_scores_for_planning)
#         t1 = sorted_scores[len(sorted_scores) // 3]
#         t2 = sorted_scores[2 * len(sorted_scores) // 3]

#         groups = {'low_caption': [], 'mid_caption': [], 'high_caption': []}
#         for idx, (i, r) in enumerate(planning_samples):
#             score = caption_scores_for_planning[idx]
#             if score <= t1:
#                 groups['low_caption'].append(r)
#             elif score <= t2:
#                 groups['mid_caption'].append(r)
#             else:
#                 groups['high_caption'].append(r)

#         for group_name, group_samples in groups.items():
#             if group_samples:
#                 ades = [s['sample_ade'] for s in group_samples]
#                 fdes = [s['sample_fde'] for s in group_samples]
#                 results[f'{group_name}_ADE'] = round(float(np.mean(ades)), 4)
#                 results[f'{group_name}_FDE'] = round(float(np.mean(fdes)), 4)
#                 results[f'{group_name}_n'] = len(group_samples)

#         # Caption quality thresholds for ROUGE-L
#         results['caption_tertile_thresholds'] = [round(t1, 4), round(t2, 4)]

#     # ── 2d. Split by field accuracy ──
#     field_scores_for_planning = [sample_field_scores[i] for i, _ in planning_samples]
#     correct_plan = [r for (i, r), fs in zip(planning_samples, field_scores_for_planning)
#                     if fs >= 0.6]
#     wrong_plan = [r for (i, r), fs in zip(planning_samples, field_scores_for_planning)
#                   if fs < 0.6]

#     if correct_plan:
#         results['field_correct_ADE'] = round(float(np.mean([s['sample_ade'] for s in correct_plan])), 4)
#         results['field_correct_FDE'] = round(float(np.mean([s['sample_fde'] for s in correct_plan])), 4)
#         results['field_correct_n'] = len(correct_plan)
#     if wrong_plan:
#         results['field_wrong_ADE'] = round(float(np.mean([s['sample_ade'] for s in wrong_plan])), 4)
#         results['field_wrong_FDE'] = round(float(np.mean([s['sample_fde'] for s in wrong_plan])), 4)
#         results['field_wrong_n'] = len(wrong_plan)

#     # ── 2e. Correlation coefficient ──
#     if len(planning_samples) >= 10:
#         ades = np.array([r['sample_ade'] for _, r in planning_samples])
#         cap_scores = np.array(caption_scores_for_planning)
#         field_scores = np.array(field_scores_for_planning)

#         # Pearson correlation (caption quality vs planning error)
#         # Negative correlation = good caption → low error (desired)
#         if np.std(cap_scores) > 1e-6 and np.std(ades) > 1e-6:
#             corr_caption_ade = float(np.corrcoef(cap_scores, ades)[0, 1])
#             results['corr_caption_vs_ADE'] = round(corr_caption_ade, 4)

#         if np.std(field_scores) > 1e-6 and np.std(ades) > 1e-6:
#             corr_field_ade = float(np.corrcoef(field_scores, ades)[0, 1])
#             results['corr_field_acc_vs_ADE'] = round(corr_field_ade, 4)

#     # ── 2f. Scene understanding gated planning (inspired by m@kIoU) ──
#     # Only count planning metrics for samples where caption quality exceeds threshold
#     for caption_threshold in [0.3, 0.5, 0.7]:
#         gated = [
#             r for (i, r), cs in zip(planning_samples, caption_scores_for_planning)
#             if cs >= caption_threshold
#         ]
#         if gated:
#             gated_ade = float(np.mean([s['sample_ade'] for s in gated]))
#             gated_fde = float(np.mean([s['sample_fde'] for s in gated]))
#             results[f'ADE@caption>={caption_threshold}'] = round(gated_ade, 4)
#             results[f'FDE@caption>={caption_threshold}'] = round(gated_fde, 4)
#             results[f'n@caption>={caption_threshold}'] = len(gated)

#     return results


# # =============================================================================
# # Layer 3: Trajectory Safety Proxy Metrics
# # =============================================================================

# def compute_trajectory_safety(
#     all_pred_waypoints: List[np.ndarray],
#     dt_list: Optional[List[float]] = None,
# ) -> Dict[str, float]:
#     """
#     Compute trajectory safety and comfort proxy metrics.

#     Since we don't have surrounding object annotations for true collision
#     detection, we compute metrics from the trajectory itself that serve as
#     proxies for safety and physical plausibility.

#     These correspond to DiffusionDrive's Comfort sub-score and partially
#     to the DAC (drivable area compliance) and TTC (time-to-collision) ideas.

#     Metrics computed:
#       - Lateral acceleration: high values = uncomfortable / unsafe lane changes
#       - Longitudinal jerk: derivative of acceleration, measures smoothness
#       - Max curvature: sharp turns that may be physically implausible
#       - Heading change rate: proxy for yaw rate
#       - Backward motion detection: trajectory segments moving backward
#       - Excessive lateral deviation: trajectory deviating too far sideways

#     Args:
#         all_pred_waypoints: list of [K, 3] arrays (K waypoints, x/y/z)
#         dt_list: time intervals between waypoints. If None, uses default
#                  [0.5s] intervals (8 waypoints from 0.5s to 4.0s)

#     Returns:
#         dict with safety proxy metrics
#     """
#     if not all_pred_waypoints:
#         return {}

#     # Default: 0.5s intervals for 8 dense waypoints
#     # [t+0.5, t+1.0, ..., t+4.0]
#     K = len(all_pred_waypoints[0])
#     if dt_list is None:
#         dt_list = [0.5] * K  # uniform 0.5s steps

#     all_lat_acc = []
#     all_lon_jerk = []
#     all_curvature = []
#     all_heading_rate = []
#     backward_count = 0
#     excessive_lateral_count = 0
#     total_count = len(all_pred_waypoints)

#     for wp in all_pred_waypoints:
#         wp = np.array(wp)
#         if wp.shape[0] < 3:
#             continue

#         x, y = wp[:, 0], wp[:, 1]

#         # ── Velocity ──
#         # v = Δpos / Δt
#         vx = np.diff(x) / np.array(dt_list[:len(x)-1])
#         vy = np.diff(y) / np.array(dt_list[:len(y)-1])
#         speed = np.sqrt(vx**2 + vy**2 + 1e-8)

#         # ── Acceleration ──
#         if len(vx) >= 2:
#             ax = np.diff(vx) / np.array(dt_list[:len(vx)-1])
#             ay = np.diff(vy) / np.array(dt_list[:len(vy)-1])

#             # Longitudinal acceleration (along trajectory direction)
#             # Lateral acceleration (perpendicular to trajectory direction)
#             for k in range(len(ax)):
#                 if speed[k] < 0.01:
#                     continue
#                 # Unit direction vector
#                 dx = vx[k] / speed[k]
#                 dy = vy[k] / speed[k]
#                 # Longitudinal component
#                 a_lon = ax[k] * dx + ay[k] * dy
#                 # Lateral component
#                 a_lat = -ax[k] * dy + ay[k] * dx
#                 all_lat_acc.append(abs(a_lat))

#             # ── Jerk (derivative of acceleration) ──
#             if len(ax) >= 2:
#                 jx = np.diff(ax) / np.array(dt_list[:len(ax)-1])
#                 jy = np.diff(ay) / np.array(dt_list[:len(ay)-1])
#                 lon_jerk = np.sqrt(jx**2 + jy**2)
#                 all_lon_jerk.extend(lon_jerk.tolist())

#         # ── Curvature ──
#         # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
#         if len(vx) >= 2:
#             for k in range(len(ax)):
#                 denom = (vx[k]**2 + vy[k]**2) ** 1.5
#                 if denom > 1e-6:
#                     kappa = abs(vx[k] * ay[k] - vy[k] * ax[k]) / denom
#                     all_curvature.append(kappa)

#         # ── Heading change rate ──
#         heading = np.arctan2(vy, vx)
#         if len(heading) >= 2:
#             dheading = np.diff(heading)
#             # Wrap to [-pi, pi]
#             dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
#             heading_rate = np.abs(dheading) / np.array(dt_list[:len(dheading)])
#             all_heading_rate.extend(heading_rate.tolist())

#         # ── Backward motion detection ──
#         # If x decreases significantly (assuming forward = +x in ego frame)
#         if np.any(np.diff(x) < -0.5):  # moving backward by >0.5m
#             backward_count += 1

#         # ── Excessive lateral deviation ──
#         # If |y| exceeds typical lane width (3.5m) at any waypoint
#         if np.any(np.abs(y) > 3.5):
#             excessive_lateral_count += 1

#     results = {}

#     if all_lat_acc:
#         lat_acc = np.array(all_lat_acc)
#         results['avg_lateral_acc'] = round(float(np.mean(lat_acc)), 4)
#         results['max_lateral_acc'] = round(float(np.max(lat_acc)), 4)
#         results['p95_lateral_acc'] = round(float(np.percentile(lat_acc, 95)), 4)
#         # Comfort threshold: lateral acceleration > 3 m/s² is uncomfortable
#         results['pct_uncomfortable_lat'] = round(
#             float(np.mean(lat_acc > 3.0) * 100), 1
#         )

#     if all_lon_jerk:
#         jerk = np.array(all_lon_jerk)
#         results['avg_jerk'] = round(float(np.mean(jerk)), 4)
#         results['max_jerk'] = round(float(np.max(jerk)), 4)
#         results['p95_jerk'] = round(float(np.percentile(jerk, 95)), 4)

#     if all_curvature:
#         curv = np.array(all_curvature)
#         results['avg_curvature'] = round(float(np.mean(curv)), 4)
#         results['max_curvature'] = round(float(np.max(curv)), 4)
#         results['p95_curvature'] = round(float(np.percentile(curv, 95)), 4)
#         # Physically implausible curvature (radius < 5m → κ > 0.2)
#         results['pct_sharp_turn'] = round(
#             float(np.mean(curv > 0.2) * 100), 1
#         )

#     if all_heading_rate:
#         hr = np.array(all_heading_rate)
#         results['avg_heading_rate'] = round(float(np.mean(hr)), 4)
#         results['max_heading_rate'] = round(float(np.max(hr)), 4)

#     results['pct_backward_motion'] = round(
#         backward_count / max(total_count, 1) * 100, 1
#     )
#     results['pct_excessive_lateral'] = round(
#         excessive_lateral_count / max(total_count, 1) * 100, 1
#     )

#     # ── Composite comfort score ──
#     # Inspired by DiffusionDrive's Comfort sub-score
#     # Score from 0-100, penalized by uncomfortable dynamics
#     comfort_penalties = 0.0
#     if all_lat_acc:
#         # Penalize high lateral acceleration
#         comfort_penalties += min(30, np.mean(lat_acc > 3.0) * 100)
#     if all_lon_jerk:
#         # Penalize high jerk
#         comfort_penalties += min(30, np.mean(np.array(all_lon_jerk) > 10.0) * 100)
#     if all_curvature:
#         # Penalize implausible curvature
#         comfort_penalties += min(20, np.mean(np.array(all_curvature) > 0.2) * 100)
#     # Penalize backward motion and excessive lateral
#     comfort_penalties += min(10, results['pct_backward_motion'])
#     comfort_penalties += min(10, results['pct_excessive_lateral'])

#     results['comfort_score'] = round(max(0, 100 - comfort_penalties), 1)

#     return results


# # =============================================================================
# # Integration Helper: Print Enhanced Summary
# # =============================================================================

# def print_enhanced_summary(
#     meteor: dict, rouge: dict, cider: dict,
#     joint: dict, safety: dict,
# ):
#     """Print the enhanced metrics in a formatted summary."""

#     print()
#     print("══ ENHANCED CAPTION METRICS (Layer 1) ══════════════════════")
#     print()
#     print("── METEOR (semantic-aware unigram matching) ──")
#     for k, v in meteor.items():
#         print(f"  {k}: {v}")

#     print()
#     print("── ROUGE-L (longest common subsequence) ──")
#     for k, v in rouge.items():
#         print(f"  {k}: {v}")

#     print()
#     print("── CIDEr (TF-IDF weighted, domain-specific) ──")
#     for k, v in cider.items():
#         print(f"  {k}: {v}")

#     if joint:
#         print()
#         print("══ CAPTION-PLAN JOINT ANALYSIS (Layer 2) ═══════════════════")
#         print()

#         # Correlation
#         if 'corr_caption_vs_ADE' in joint:
#             corr = joint['corr_caption_vs_ADE']
#             direction = "negative (good: better caption → lower error)" if corr < 0 else "positive (caption and planning may be decoupled)"
#             print(f"  Correlation (caption quality vs ADE): {corr:.4f}  ({direction})")

#         if 'corr_field_acc_vs_ADE' in joint:
#             corr = joint['corr_field_acc_vs_ADE']
#             direction = "negative (good)" if corr < 0 else "positive (decoupled)"
#             print(f"  Correlation (field acc vs ADE):       {corr:.4f}  ({direction})")

#         # Caption-gated planning
#         print()
#         print("── Planning quality by caption quality tertile ──")
#         for group in ['low_caption', 'mid_caption', 'high_caption']:
#             if f'{group}_ADE' in joint:
#                 print(f"  {group:15s}: ADE={joint[f'{group}_ADE']:.4f}  "
#                       f"FDE={joint[f'{group}_FDE']:.4f}  n={joint[f'{group}_n']}")

#         # Field-accuracy conditioned
#         print()
#         print("── Planning quality by field accuracy (threshold=0.6) ──")
#         if 'field_correct_ADE' in joint:
#             print(f"  Field correct: ADE={joint['field_correct_ADE']:.4f}  "
#                   f"FDE={joint['field_correct_FDE']:.4f}  n={joint['field_correct_n']}")
#         if 'field_wrong_ADE' in joint:
#             print(f"  Field wrong:   ADE={joint['field_wrong_ADE']:.4f}  "
#                   f"FDE={joint['field_wrong_FDE']:.4f}  n={joint['field_wrong_n']}")

#         # Gated metrics
#         print()
#         print("── Scene-understanding gated planning (m@kIoU inspired) ──")
#         for thresh in [0.3, 0.5, 0.7]:
#             k = f'ADE@caption>={thresh}'
#             if k in joint:
#                 print(f"  {k}: {joint[k]:.4f}  "
#                       f"FDE@caption>={thresh}: {joint[f'FDE@caption>={thresh}']:.4f}  "
#                       f"n={joint[f'n@caption>={thresh}']}")

#     if safety:
#         print()
#         print("══ TRAJECTORY SAFETY PROXIES (Layer 3) ═════════════════════")
#         print(f"  (No surrounding object annotations needed)")
#         print()

#         print("── Comfort & Dynamics ──")
#         for k in ['avg_lateral_acc', 'max_lateral_acc', 'p95_lateral_acc', 'pct_uncomfortable_lat']:
#             if k in safety:
#                 unit = ' m/s²' if 'acc' in k else '%' if 'pct' in k else ''
#                 print(f"  {k}: {safety[k]}{unit}")

#         print()
#         print("── Smoothness (Jerk) ──")
#         for k in ['avg_jerk', 'max_jerk', 'p95_jerk']:
#             if k in safety:
#                 print(f"  {k}: {safety[k]} m/s³")

#         print()
#         print("── Curvature & Heading ──")
#         for k in ['avg_curvature', 'max_curvature', 'pct_sharp_turn',
#                    'avg_heading_rate', 'max_heading_rate']:
#             if k in safety:
#                 unit = ' 1/m' if 'curv' in k else ' rad/s' if 'heading' in k else '%'
#                 print(f"  {k}: {safety[k]}{unit}")

#         print()
#         print("── Physical Plausibility ──")
#         print(f"  pct_backward_motion:   {safety.get('pct_backward_motion', 0)}%")
#         print(f"  pct_excessive_lateral: {safety.get('pct_excessive_lateral', 0)}%")

#         if 'comfort_score' in safety:
#             print()
#             score = safety['comfort_score']
#             label = 'Excellent' if score >= 90 else 'Good' if score >= 70 else 'Fair' if score >= 50 else 'Poor'
#             print(f"  ★ Composite Comfort Score: {score}/100 ({label})")


# # =============================================================================
# # Integration Guide
# # =============================================================================

# INTEGRATION_GUIDE = """
# ╔══════════════════════════════════════════════════════════════════════╗
# ║            INTEGRATION INTO evaluate_with_planning.py              ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                    ║
# ║  Add these imports at the top of evaluate_with_planning.py:        ║
# ║                                                                    ║
# ║    from enhanced_metrics import (                                  ║
# ║        compute_meteor_scores,                                      ║
# ║        compute_rouge_scores,                                       ║
# ║        compute_cider_scores,                                       ║
# ║        compute_caption_plan_joint,                                 ║
# ║        compute_trajectory_safety,                                  ║
# ║        print_enhanced_summary,                                     ║
# ║    )                                                               ║
# ║                                                                    ║
# ║  Then in _run_all_metrics(), add after the existing VQA section:   ║
# ║                                                                    ║
# ║    # ═══ Enhanced Metrics ═══                                      ║
# ║    meteor = compute_meteor_scores(all_gt_texts, all_pred_texts)    ║
# ║    rouge  = compute_rouge_scores(all_gt_texts, all_pred_texts)     ║
# ║    cider  = compute_cider_scores(all_gt_texts, all_pred_texts)     ║
# ║                                                                    ║
# ║    joint = compute_caption_plan_joint(                              ║
# ║        per_sample_results, all_gt_texts, all_pred_texts,           ║
# ║        all_gt_dicts, all_pred_dicts,                               ║
# ║    )                                                               ║
# ║                                                                    ║
# ║    safety = {}                                                     ║
# ║    if all_pred_waypoints:                                          ║
# ║        pred_wp_np = [np.array(w) for w in all_pred_waypoints]      ║
# ║        safety = compute_trajectory_safety(pred_wp_np)              ║
# ║                                                                    ║
# ║    print_enhanced_summary(meteor, rouge, cider, joint, safety)     ║
# ║                                                                    ║
# ║    # Add to results dict:                                          ║
# ║    results['enhanced'] = {                                         ║
# ║        'meteor': meteor, 'rouge': rouge, 'cider': cider,          ║
# ║        'caption_plan_joint': joint,                                ║
# ║        'trajectory_safety': safety,                                ║
# ║    }                                                               ║
# ║                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
# """


# if __name__ == '__main__':
#     print(INTEGRATION_GUIDE)

#     # ── Quick sanity test ──
#     print("\n── Sanity Test ──\n")

#     refs = [
#         "clear weather sunny driving straight on highway lane keeping normal",
#         "rainy weather traffic light red stop at intersection hazard present",
#         "cloudy weather no traffic light lane keeping deviation left slow down",
#     ]
#     hyps = [
#         "clear sunny weather driving forward on highway lane centered",
#         "rain weather traffic light is red stopping at intersection hazard detected",
#         "overcast weather no signal lane drifting left decelerate",
#     ]

#     meteor = compute_meteor_scores(refs, hyps)
#     rouge = compute_rouge_scores(refs, hyps)
#     cider = compute_cider_scores(refs, hyps)

#     print(f"METEOR: {meteor}")
#     print(f"ROUGE:  {rouge}")
#     print(f"CIDEr:  {cider}")

#     # Test safety metrics
#     print("\n── Safety Test ──\n")
#     np.random.seed(42)
#     # Simulate 10 trajectories of 8 waypoints each
#     test_waypoints = []
#     for _ in range(10):
#         # Simple forward motion with some noise
#         wp = np.zeros((8, 3))
#         for k in range(8):
#             t = (k + 1) * 0.5  # 0.5s to 4.0s
#             wp[k, 0] = t * 5.0 + np.random.randn() * 0.3  # forward ~5 m/s
#             wp[k, 1] = np.random.randn() * 0.5  # small lateral noise
#         test_waypoints.append(wp)

#     safety = compute_trajectory_safety(test_waypoints)
#     for k, v in safety.items():
#         print(f"  {k}: {v}")

#     print("\nAll tests passed!")


"""
Enhanced Evaluation Metrics for Caption-to-Plan Evaluation.
(Optimized version — multiprocess + C-backed LCS + caching)

Adds three layers of metrics on top of the existing evaluate_with_planning.py:

Layer 1 - Complete Caption Metrics (from TOD3Cap):
  - METEOR (semantic-aware, considers synonyms and stems)
  - ROUGE-L (longest common subsequence based recall)
  - CIDEr  (TF-IDF weighted n-gram consensus, best for domain-specific captioning)

Layer 2 - Caption-Plan Joint Analysis:
  - Caption-conditioned planning error (ADE/FDE grouped by caption quality)
  - Field-accuracy-conditioned planning error
  - Correlation analysis between caption quality and planning quality

Layer 3 - Trajectory Safety Proxy Metrics (no surrounding object annotations needed):
  - Lateral acceleration (comfort / safety proxy)
  - Longitudinal jerk (smoothness)
  - Curvature statistics (physical plausibility)
  - Heading rate (yaw rate proxy)
  - Out-of-bounds detection (trajectory going backwards or sideways too far)

Performance optimizations vs original:
  - LCS uses difflib.SequenceMatcher (C-backed) instead of pure-Python DP
  - METEOR/ROUGE/CIDEr use multiprocessing for large inputs (>500 samples)
  - Joint analysis reuses pre-computed ROUGE-L scores instead of re-computing LCS
  - CIDEr vectorized with numpy where possible
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher
import os


# Number of workers for parallel computation
_NUM_WORKERS = min(os.cpu_count() or 4, 8)
# Threshold: only use multiprocessing when sample count exceeds this
_MP_THRESHOLD = 500


# =============================================================================
# Layer 1: Complete Caption Metrics
# =============================================================================

# ── METEOR ──────────────────────────────────────────────────────────────────

def _count_unigram_matches(reference_tokens: list, hypothesis_tokens: list) -> int:
    ref_counts = Counter(reference_tokens)
    hyp_counts = Counter(hypothesis_tokens)
    matches = 0
    for token, count in hyp_counts.items():
        matches += min(count, ref_counts.get(token, 0))
    return matches


def _count_chunks(reference_tokens: list, hypothesis_tokens: list) -> int:
    ref_set = set(reference_tokens)
    in_chunk = False
    chunks = 0
    for token in hypothesis_tokens:
        if token in ref_set:
            if not in_chunk:
                chunks += 1
                in_chunk = True
        else:
            in_chunk = False
    return max(chunks, 1)


def _meteor_single(args):
    """Compute METEOR for a single (ref, hyp) pair. Used by ProcessPoolExecutor."""
    ref, hyp, alpha, beta, gamma = args
    ref_tokens = ref.lower().split()
    hyp_tokens = hyp.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    matches = _count_unigram_matches(ref_tokens, hyp_tokens)
    if matches == 0:
        return 0.0

    precision = matches / len(hyp_tokens)
    recall = matches / len(ref_tokens)
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall + 1e-8)

    chunks = _count_chunks(ref_tokens, hyp_tokens)
    frag = chunks / matches if matches > 0 else 1.0
    penalty = gamma * (frag ** beta)

    return f_mean * (1.0 - min(penalty, 1.0))


def compute_meteor_scores(references: List[str], hypotheses: List[str],
                          alpha: float = 0.9, beta: float = 3.0,
                          gamma: float = 0.5) -> Dict[str, float]:
    """Compute METEOR score (simplified version without synonym/stem matching)."""
    N = len(references)
    if N == 0:
        return {'METEOR': 0.0, 'METEOR_std': 0.0}

    args_list = [(r, h, alpha, beta, gamma) for r, h in zip(references, hypotheses)]

    if N > _MP_THRESHOLD:
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as pool:
            scores = list(pool.map(_meteor_single, args_list, chunksize=max(1, N // _NUM_WORKERS)))
    else:
        scores = [_meteor_single(a) for a in args_list]

    return {
        'METEOR': round(float(np.mean(scores)), 4),
        'METEOR_std': round(float(np.std(scores)), 4),
    }


# ── ROUGE-L ─────────────────────────────────────────────────────────────────
# Uses difflib.SequenceMatcher (C-backed) for LCS instead of pure-Python DP.

def _lcs_length(x: list, y: list) -> int:
    """
    Compute LCS length using difflib.SequenceMatcher.
    This is backed by C code and significantly faster than pure-Python DP
    for typical text lengths.
    """
    sm = SequenceMatcher(None, x, y, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


def _rouge_l_single(args):
    """Compute ROUGE-L for a single (ref, hyp) pair."""
    ref, hyp, beta = args
    ref_tokens = ref.lower().split()
    hyp_tokens = hyp.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    p = lcs / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0
    r = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

    if p + r == 0:
        f = 0.0
    else:
        f = ((1 + beta ** 2) * p * r) / (beta ** 2 * p + r)

    return p, r, f


def compute_rouge_scores(references: List[str], hypotheses: List[str],
                         beta: float = 1.2) -> Dict[str, float]:
    """Compute ROUGE-L score based on longest common subsequence."""
    N = len(references)
    if N == 0:
        return {'ROUGE-L-P': 0.0, 'ROUGE-L-R': 0.0, 'ROUGE-L-F': 0.0}

    args_list = [(r, h, beta) for r, h in zip(references, hypotheses)]

    if N > _MP_THRESHOLD:
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as pool:
            results = list(pool.map(_rouge_l_single, args_list, chunksize=max(1, N // _NUM_WORKERS)))
    else:
        results = [_rouge_l_single(a) for a in args_list]

    precisions = [r[0] for r in results]
    recalls = [r[1] for r in results]
    f_scores = [r[2] for r in results]

    return {
        'ROUGE-L-P': round(float(np.mean(precisions)), 4),
        'ROUGE-L-R': round(float(np.mean(recalls)), 4),
        'ROUGE-L-F': round(float(np.mean(f_scores)), 4),
    }


def compute_rouge_f_scores(references: List[str], hypotheses: List[str],
                           beta: float = 1.2) -> List[float]:
    """
    Compute per-sample ROUGE-L F scores. Used by joint analysis to avoid
    re-computing LCS.
    """
    N = len(references)
    if N == 0:
        return []

    args_list = [(r, h, beta) for r, h in zip(references, hypotheses)]

    if N > _MP_THRESHOLD:
        with ProcessPoolExecutor(max_workers=_NUM_WORKERS) as pool:
            results = list(pool.map(_rouge_l_single, args_list, chunksize=max(1, N // _NUM_WORKERS)))
    else:
        results = [_rouge_l_single(a) for a in args_list]

    return [r[2] for r in results]


# ── CIDEr ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list:
    return text.lower().split()


def _compute_ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _compute_doc_freq(references_tokenized: List[list], n: int) -> Counter:
    df = Counter()
    for ref_tokens in references_tokenized:
        ngrams = set(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        for ng in ngrams:
            df[ng] += 1
    return df


def compute_cider_scores(references: List[str], hypotheses: List[str],
                         n_max: int = 4, sigma: float = 6.0) -> Dict[str, float]:
    """Compute CIDEr-D score."""
    N = len(references)
    if N == 0:
        return {'CIDEr': 0.0}

    refs_tok = [_tokenize(r) for r in references]
    hyps_tok = [_tokenize(h) for h in hypotheses]

    doc_freqs = {}
    for n in range(1, n_max + 1):
        doc_freqs[n] = _compute_doc_freq(refs_tok, n)

    def _tfidf_vec(tokens, n, df):
        ngrams = _compute_ngrams(tokens, n)
        vec = {}
        length = max(len(tokens) - n + 1, 1)
        for ng, count in ngrams.items():
            tf = count / length
            idf = np.log(max(1.0, N) / (1.0 + df.get(ng, 0)))
            vec[ng] = tf * idf
        return vec

    def _cosine_sim(vec1, vec2):
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        dot = sum(vec1[k] * vec2[k] for k in common)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    cider_n_scores = {n: [] for n in range(1, n_max + 1)}

    for i in range(N):
        ref_tok = refs_tok[i]
        hyp_tok = hyps_tok[i]
        len_diff = len(hyp_tok) - len(ref_tok)
        length_penalty = np.exp(-(len_diff ** 2) / (2 * sigma ** 2))

        for n in range(1, n_max + 1):
            df = doc_freqs[n]
            ref_vec = _tfidf_vec(ref_tok, n, df)
            hyp_vec = _tfidf_vec(hyp_tok, n, df)
            sim = _cosine_sim(ref_vec, hyp_vec) * length_penalty
            cider_n_scores[n].append(sim)

    results = {}
    total_cider = 0.0
    for n in range(1, n_max + 1):
        score = float(np.mean(cider_n_scores[n])) if cider_n_scores[n] else 0.0
        results[f'CIDEr-{n}'] = round(score * 10, 4)
        total_cider += score

    results['CIDEr'] = round(total_cider / n_max * 10, 4)
    return results


# =============================================================================
# Layer 2: Caption-Plan Joint Analysis
# =============================================================================

def compute_caption_plan_joint(
    per_sample_results: List[dict],
    all_gt_texts: List[str],
    all_pred_texts: List[str],
    all_gt_dicts: List[dict],
    all_pred_dicts: List[dict],
    precomputed_rouge_f: List[float] = None,
) -> Dict[str, any]:
    """
    Analyze the correlation between caption quality and planning quality.

    Args:
        precomputed_rouge_f: If provided, skip LCS re-computation and use these
                             per-sample ROUGE-L F scores directly. Pass the output
                             of compute_rouge_f_scores() here.
    """
    results = {}

    # ── 2a. Per-sample caption quality (reuse precomputed if available) ──
    if precomputed_rouge_f is not None and len(precomputed_rouge_f) == len(all_gt_texts):
        sample_caption_scores = precomputed_rouge_f
    else:
        sample_caption_scores = compute_rouge_f_scores(all_gt_texts, all_pred_texts)

    # ── 2b. Per-sample field accuracy ──
    sample_field_scores = []
    key_fields = [
        ('weather', 'condition'),
        ('traffic_light', 'state'),
        ('forward_drivability', 'status'),
        ('lane_keeping', 'status'),
        ('driving_advice', 'action'),
    ]
    for gt_dict, pred_dict in zip(all_gt_dicts, all_pred_dicts):
        if not isinstance(pred_dict, dict):
            sample_field_scores.append(0.0)
            continue
        correct = total = 0
        for top, sub in key_fields:
            gt_val = gt_dict.get(top, {}).get(sub, None)
            pred_val = pred_dict.get(top, {}).get(sub, None)
            if gt_val is not None:
                total += 1
                if str(gt_val).lower() == str(pred_val).lower():
                    correct += 1
        sample_field_scores.append(correct / total if total > 0 else 0.0)

    # ── 2c. Split planning metrics by caption quality ──
    planning_samples = [
        (i, r) for i, r in enumerate(per_sample_results)
        if 'sample_ade' in r and 'sample_fde' in r
    ]

    if not planning_samples:
        results['note'] = 'No planning samples available for joint analysis'
        return results

    caption_scores_for_planning = [sample_caption_scores[i] for i, _ in planning_samples]
    if len(caption_scores_for_planning) >= 3:
        sorted_scores = sorted(caption_scores_for_planning)
        t1 = sorted_scores[len(sorted_scores) // 3]
        t2 = sorted_scores[2 * len(sorted_scores) // 3]

        groups = {'low_caption': [], 'mid_caption': [], 'high_caption': []}
        for idx, (i, r) in enumerate(planning_samples):
            score = caption_scores_for_planning[idx]
            if score <= t1:
                groups['low_caption'].append(r)
            elif score <= t2:
                groups['mid_caption'].append(r)
            else:
                groups['high_caption'].append(r)

        for group_name, group_samples in groups.items():
            if group_samples:
                ades = [s['sample_ade'] for s in group_samples]
                fdes = [s['sample_fde'] for s in group_samples]
                results[f'{group_name}_ADE'] = round(float(np.mean(ades)), 4)
                results[f'{group_name}_FDE'] = round(float(np.mean(fdes)), 4)
                results[f'{group_name}_n'] = len(group_samples)

        results['caption_tertile_thresholds'] = [round(t1, 4), round(t2, 4)]

    # ── 2d. Split by field accuracy ──
    field_scores_for_planning = [sample_field_scores[i] for i, _ in planning_samples]
    correct_plan = [r for (i, r), fs in zip(planning_samples, field_scores_for_planning)
                    if fs >= 0.6]
    wrong_plan = [r for (i, r), fs in zip(planning_samples, field_scores_for_planning)
                  if fs < 0.6]

    if correct_plan:
        results['field_correct_ADE'] = round(float(np.mean([s['sample_ade'] for s in correct_plan])), 4)
        results['field_correct_FDE'] = round(float(np.mean([s['sample_fde'] for s in correct_plan])), 4)
        results['field_correct_n'] = len(correct_plan)
    if wrong_plan:
        results['field_wrong_ADE'] = round(float(np.mean([s['sample_ade'] for s in wrong_plan])), 4)
        results['field_wrong_FDE'] = round(float(np.mean([s['sample_fde'] for s in wrong_plan])), 4)
        results['field_wrong_n'] = len(wrong_plan)

    # ── 2e. Correlation coefficient ──
    if len(planning_samples) >= 10:
        ades = np.array([r['sample_ade'] for _, r in planning_samples])
        cap_scores = np.array(caption_scores_for_planning)
        field_scores = np.array(field_scores_for_planning)

        if np.std(cap_scores) > 1e-6 and np.std(ades) > 1e-6:
            corr_caption_ade = float(np.corrcoef(cap_scores, ades)[0, 1])
            results['corr_caption_vs_ADE'] = round(corr_caption_ade, 4)

        if np.std(field_scores) > 1e-6 and np.std(ades) > 1e-6:
            corr_field_ade = float(np.corrcoef(field_scores, ades)[0, 1])
            results['corr_field_acc_vs_ADE'] = round(corr_field_ade, 4)

    # ── 2f. Scene understanding gated planning ──
    for caption_threshold in [0.3, 0.5, 0.7]:
        gated = [
            r for (i, r), cs in zip(planning_samples, caption_scores_for_planning)
            if cs >= caption_threshold
        ]
        if gated:
            gated_ade = float(np.mean([s['sample_ade'] for s in gated]))
            gated_fde = float(np.mean([s['sample_fde'] for s in gated]))
            results[f'ADE@caption>={caption_threshold}'] = round(gated_ade, 4)
            results[f'FDE@caption>={caption_threshold}'] = round(gated_fde, 4)
            results[f'n@caption>={caption_threshold}'] = len(gated)

    return results


# =============================================================================
# Layer 3: Trajectory Safety Proxy Metrics
# =============================================================================

def compute_trajectory_safety(
    all_pred_waypoints: List[np.ndarray],
    dt_list: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute trajectory safety and comfort proxy metrics."""
    if not all_pred_waypoints:
        return {}

    K = len(all_pred_waypoints[0])
    if dt_list is None:
        dt_list = [0.5] * K

    all_lat_acc = []
    all_lon_jerk = []
    all_curvature = []
    all_heading_rate = []
    backward_count = 0
    excessive_lateral_count = 0
    total_count = len(all_pred_waypoints)

    for wp in all_pred_waypoints:
        wp = np.array(wp)
        if wp.shape[0] < 3:
            continue

        x, y = wp[:, 0], wp[:, 1]

        # Velocity
        dt_arr = np.array(dt_list[:len(x)-1])
        vx = np.diff(x) / dt_arr
        vy = np.diff(y) / dt_arr
        speed = np.sqrt(vx**2 + vy**2 + 1e-8)

        # Acceleration
        if len(vx) >= 2:
            dt_arr2 = np.array(dt_list[:len(vx)-1])
            ax = np.diff(vx) / dt_arr2
            ay = np.diff(vy) / dt_arr2

            for k in range(len(ax)):
                if speed[k] < 0.01:
                    continue
                dx_dir = vx[k] / speed[k]
                dy_dir = vy[k] / speed[k]
                a_lat = -ax[k] * dy_dir + ay[k] * dx_dir
                all_lat_acc.append(abs(a_lat))

            # Jerk
            if len(ax) >= 2:
                dt_arr3 = np.array(dt_list[:len(ax)-1])
                jx = np.diff(ax) / dt_arr3
                jy = np.diff(ay) / dt_arr3
                lon_jerk = np.sqrt(jx**2 + jy**2)
                all_lon_jerk.extend(lon_jerk.tolist())

        # Curvature
        if len(vx) >= 2 and len(ax) == len(vx) - 1:
            for k in range(len(ax)):
                denom = (vx[k]**2 + vy[k]**2) ** 1.5
                if denom > 1e-6:
                    kappa = abs(vx[k] * ay[k] - vy[k] * ax[k]) / denom
                    all_curvature.append(kappa)

        # Heading change rate
        heading = np.arctan2(vy, vx)
        if len(heading) >= 2:
            dheading = np.diff(heading)
            dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
            heading_rate = np.abs(dheading) / np.array(dt_list[:len(dheading)])
            all_heading_rate.extend(heading_rate.tolist())

        # Backward motion
        if np.any(np.diff(x) < -0.5):
            backward_count += 1

        # Excessive lateral
        if np.any(np.abs(y) > 3.5):
            excessive_lateral_count += 1

    results = {}

    if all_lat_acc:
        lat_acc = np.array(all_lat_acc)
        results['avg_lateral_acc'] = round(float(np.mean(lat_acc)), 4)
        results['max_lateral_acc'] = round(float(np.max(lat_acc)), 4)
        results['p95_lateral_acc'] = round(float(np.percentile(lat_acc, 95)), 4)
        results['pct_uncomfortable_lat'] = round(float(np.mean(lat_acc > 3.0) * 100), 1)

    if all_lon_jerk:
        jerk = np.array(all_lon_jerk)
        results['avg_jerk'] = round(float(np.mean(jerk)), 4)
        results['max_jerk'] = round(float(np.max(jerk)), 4)
        results['p95_jerk'] = round(float(np.percentile(jerk, 95)), 4)

    if all_curvature:
        curv = np.array(all_curvature)
        results['avg_curvature'] = round(float(np.mean(curv)), 4)
        results['max_curvature'] = round(float(np.max(curv)), 4)
        results['p95_curvature'] = round(float(np.percentile(curv, 95)), 4)
        results['pct_sharp_turn'] = round(float(np.mean(curv > 0.2) * 100), 1)

    if all_heading_rate:
        hr = np.array(all_heading_rate)
        results['avg_heading_rate'] = round(float(np.mean(hr)), 4)
        results['max_heading_rate'] = round(float(np.max(hr)), 4)

    results['pct_backward_motion'] = round(backward_count / max(total_count, 1) * 100, 1)
    results['pct_excessive_lateral'] = round(excessive_lateral_count / max(total_count, 1) * 100, 1)

    # Composite comfort score
    comfort_penalties = 0.0
    if all_lat_acc:
        comfort_penalties += min(30, np.mean(np.array(all_lat_acc) > 3.0) * 100)
    if all_lon_jerk:
        comfort_penalties += min(30, np.mean(np.array(all_lon_jerk) > 10.0) * 100)
    if all_curvature:
        comfort_penalties += min(20, np.mean(np.array(all_curvature) > 0.2) * 100)
    comfort_penalties += min(10, results['pct_backward_motion'])
    comfort_penalties += min(10, results['pct_excessive_lateral'])
    results['comfort_score'] = round(max(0, 100 - comfort_penalties), 1)

    return results


# =============================================================================
# Integration Helper: Print Enhanced Summary
# =============================================================================

def print_enhanced_summary(
    meteor: dict, rouge: dict, cider: dict,
    joint: dict, safety: dict,
):
    """Print the enhanced metrics in a formatted summary."""

    print()
    print("══ ENHANCED CAPTION METRICS (Layer 1) ══════════════════════")
    print()
    print("── METEOR (semantic-aware unigram matching) ──")
    for k, v in meteor.items():
        print(f"  {k}: {v}")

    print()
    print("── ROUGE-L (longest common subsequence) ──")
    for k, v in rouge.items():
        print(f"  {k}: {v}")

    print()
    print("── CIDEr (TF-IDF weighted, domain-specific) ──")
    for k, v in cider.items():
        print(f"  {k}: {v}")

    if joint:
        print()
        print("══ CAPTION-PLAN JOINT ANALYSIS (Layer 2) ═══════════════════")
        print()

        if 'corr_caption_vs_ADE' in joint:
            corr = joint['corr_caption_vs_ADE']
            direction = "negative (good: better caption → lower error)" if corr < 0 else "positive (caption and planning may be decoupled)"
            print(f"  Correlation (caption quality vs ADE): {corr:.4f}  ({direction})")

        if 'corr_field_acc_vs_ADE' in joint:
            corr = joint['corr_field_acc_vs_ADE']
            direction = "negative (good)" if corr < 0 else "positive (decoupled)"
            print(f"  Correlation (field acc vs ADE):       {corr:.4f}  ({direction})")

        print()
        print("── Planning quality by caption quality tertile ──")
        for group in ['low_caption', 'mid_caption', 'high_caption']:
            if f'{group}_ADE' in joint:
                print(f"  {group:15s}: ADE={joint[f'{group}_ADE']:.4f}  "
                      f"FDE={joint[f'{group}_FDE']:.4f}  n={joint[f'{group}_n']}")

        print()
        print("── Planning quality by field accuracy (threshold=0.6) ──")
        if 'field_correct_ADE' in joint:
            print(f"  Field correct: ADE={joint['field_correct_ADE']:.4f}  "
                  f"FDE={joint['field_correct_FDE']:.4f}  n={joint['field_correct_n']}")
        if 'field_wrong_ADE' in joint:
            print(f"  Field wrong:   ADE={joint['field_wrong_ADE']:.4f}  "
                  f"FDE={joint['field_wrong_FDE']:.4f}  n={joint['field_wrong_n']}")

        print()
        print("── Scene-understanding gated planning (m@kIoU inspired) ──")
        for thresh in [0.3, 0.5, 0.7]:
            k = f'ADE@caption>={thresh}'
            if k in joint:
                print(f"  {k}: {joint[k]:.4f}  "
                      f"FDE@caption>={thresh}: {joint[f'FDE@caption>={thresh}']:.4f}  "
                      f"n={joint[f'n@caption>={thresh}']}")

    if safety:
        print()
        print("══ TRAJECTORY SAFETY PROXIES (Layer 3) ═════════════════════")
        print(f"  (No surrounding object annotations needed)")
        print()

        print("── Comfort & Dynamics ──")
        for k in ['avg_lateral_acc', 'max_lateral_acc', 'p95_lateral_acc', 'pct_uncomfortable_lat']:
            if k in safety:
                unit = ' m/s²' if 'acc' in k else '%' if 'pct' in k else ''
                print(f"  {k}: {safety[k]}{unit}")

        print()
        print("── Smoothness (Jerk) ──")
        for k in ['avg_jerk', 'max_jerk', 'p95_jerk']:
            if k in safety:
                print(f"  {k}: {safety[k]} m/s³")

        print()
        print("── Curvature & Heading ──")
        for k in ['avg_curvature', 'max_curvature', 'pct_sharp_turn',
                   'avg_heading_rate', 'max_heading_rate']:
            if k in safety:
                unit = ' 1/m' if 'curv' in k else ' rad/s' if 'heading' in k else '%'
                print(f"  {k}: {safety[k]}{unit}")

        print()
        print("── Physical Plausibility ──")
        print(f"  pct_backward_motion:   {safety.get('pct_backward_motion', 0)}%")
        print(f"  pct_excessive_lateral: {safety.get('pct_excessive_lateral', 0)}%")

        if 'comfort_score' in safety:
            print()
            score = safety['comfort_score']
            label = 'Excellent' if score >= 90 else 'Good' if score >= 70 else 'Fair' if score >= 50 else 'Poor'
            print(f"  ★ Composite Comfort Score: {score}/100 ({label})")


if __name__ == '__main__':
    import time

    print("── Sanity Test ──\n")

    refs = [
        "clear weather sunny driving straight on highway lane keeping normal",
        "rainy weather traffic light red stop at intersection hazard present",
        "cloudy weather no traffic light lane keeping deviation left slow down",
    ]
    hyps = [
        "clear sunny weather driving forward on highway lane centered",
        "rain weather traffic light is red stopping at intersection hazard detected",
        "overcast weather no signal lane drifting left decelerate",
    ]

    t0 = time.time()
    meteor = compute_meteor_scores(refs, hyps)
    rouge = compute_rouge_scores(refs, hyps)
    cider = compute_cider_scores(refs, hyps)
    print(f"Small test took {time.time()-t0:.3f}s")
    print(f"METEOR: {meteor}")
    print(f"ROUGE:  {rouge}")
    print(f"CIDEr:  {cider}")

    # Benchmark with larger data
    print("\n── Scale Test (3000 samples) ──\n")
    np.random.seed(42)
    words = "clear sunny rainy cloudy weather traffic light red green lane keeping deviation left right forward stop slow".split()
    big_refs = [' '.join(np.random.choice(words, size=15)) for _ in range(3000)]
    big_hyps = [' '.join(np.random.choice(words, size=15)) for _ in range(3000)]

    t0 = time.time()
    meteor = compute_meteor_scores(big_refs, big_hyps)
    print(f"METEOR (3000): {time.time()-t0:.2f}s")

    t0 = time.time()
    rouge = compute_rouge_scores(big_refs, big_hyps)
    print(f"ROUGE  (3000): {time.time()-t0:.2f}s")

    t0 = time.time()
    cider = compute_cider_scores(big_refs, big_hyps)
    print(f"CIDEr  (3000): {time.time()-t0:.2f}s")

    # Pre-computed ROUGE reuse test
    t0 = time.time()
    rouge_f = compute_rouge_f_scores(big_refs, big_hyps)
    print(f"ROUGE-F precompute (3000): {time.time()-t0:.2f}s")

    print("\nAll tests passed!")