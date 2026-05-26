"""
MoRo-Former: Task-Aware Modality Routing Q-Former Bridge
=========================================================

Replaces the vanilla QFormerBridge with a Mixture-of-Experts design
that routes different task queries to different modality expert branches.

Architecture:
  9 structured tasks, each with K learnable queries + task-group embedding
  4 expert branches: camera(i), camera+LiDAR(il), camera+Radar(ir), trimodal(ilr)
  3 task groups:
    - Image-dominant (weather, traffic_light, traffic_sign): → image expert only
    - Adaptive-routing (participants, hazard, drivability, lane, advice): → router selects expert
    - Global (explanation): → trimodal expert

  For adaptive-routing tasks, a lightweight router predicts per-query expert
  selection probability. Hard routing at inference, Gumbel-softmax at training.

Interface: drop-in replacement for QFormerBridge
  Input:  image_features [B, N_img, D_img], lidar [B, D_lid], radar [B, D_rad]
  Output: [B, num_query_tokens, D_llm]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════
# Task definitions
# ═══════════════════════════════════════════════════════════

TASK_NAMES = [
    'weather_illumination',   # 0 — image-dominant
    'traffic_light',          # 1 — image-dominant
    'traffic_sign',           # 2 — image-dominant
    'participants',           # 3 — adaptive
    'hazard_region',          # 4 — adaptive
    'forward_drivability',    # 5 — adaptive
    'lane_keeping',           # 6 — adaptive
    'driving_advice',         # 7 — adaptive
    'explanation',            # 8 — global
]
NUM_TASKS = len(TASK_NAMES)

# Task group indices
IMAGE_TASKS = [0, 1, 2]       # weather, traffic_light, traffic_sign
ADAPTIVE_TASKS = [3, 4, 5, 6, 7]  # participants, hazard, drivability, lane, advice
GLOBAL_TASKS = [8]             # explanation

# Expert branch names
EXPERT_NAMES = ['i', 'il', 'ir', 'ilr']
NUM_EXPERTS = 4


# ═══════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to key-value pairs."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value, key_padding_mask=None):
        B, Nq, D = query.shape
        Nkv = key_value.shape[1]
        H = self.num_heads

        q = self.q_proj(query).view(B, Nq, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, Nkv, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, Nkv, H, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.out_proj(out)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mhca = MultiHeadCrossAttention(dim, num_heads, dropout)

    def forward(self, x):
        return self.mhca(x, x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ExpertBlock(nn.Module):
    """
    Single expert block: self-attention + cross-attention to expert features + FFN.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.ffn = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, queries, features):
        queries = queries + self.self_attn(self.norm1(queries))
        queries = queries + self.cross_attn(self.norm2(queries), self.norm_kv(features))
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class AttentionPool(nn.Module):
    """Compress N tokens to M tokens via attention pooling."""
    def __init__(self, dim, n_out):
        super().__init__()
        self.pool_queries = nn.Parameter(torch.randn(1, n_out, dim) * 0.02)
        self.attn = MultiHeadCrossAttention(dim, num_heads=4, dropout=0.0)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.shape[0]
        pq = self.pool_queries.expand(B, -1, -1)
        return self.attn(pq, self.norm(x))


# ═══════════════════════════════════════════════════════════
# Expert branch: constructs fused features for each branch
# ═══════════════════════════════════════════════════════════

class ExpertBranch(nn.Module):
    """
    One expert branch that fuses a subset of modalities.
    Processes queries through N layers of self-attn + cross-attn.
    """
    def __init__(self, dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ExpertBlock(dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, queries, features):
        for layer in self.layers:
            queries = layer(queries, features)
        return queries


# ═══════════════════════════════════════════════════════════
# Modality Router
# ═══════════════════════════════════════════════════════════

class ModalityRouter(nn.Module):
    """
    Per-query router that selects among 4 expert branches.
    
    For each query, produces a 4-dim logit over experts.
    Training: Gumbel-softmax for differentiable routing.
    Inference: hard argmax.
    """
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        # Lightweight router: query → probe each expert's response → score
        self.router_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_experts),
        )

    def forward(self, query_features, temperature=1.0):
        """
        Args:
            query_features: [B, N, D] — query representations
            temperature: Gumbel-softmax temperature

        Returns:
            weights: [B, N, num_experts] — routing weights (soft or hard)
            logits:  [B, N, num_experts] — raw logits for aux loss
        """
        logits = self.router_proj(query_features)  # [B, N, 4]

        if self.training:
            # Gumbel-softmax: differentiable + approximately discrete
            weights = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        else:
            # Hard routing at inference
            idx = logits.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(logits).scatter_(-1, idx, 1.0)

        return weights, logits


# ═══════════════════════════════════════════════════════════
# MoRo-Former: main module
# ═══════════════════════════════════════════════════════════

class MoRoFormer(nn.Module):
    """
    MoRo-Former: Task-Aware Modality Routing Q-Former Bridge.

    Drop-in replacement for QFormerBridge with the same interface.

    Args: same as QFormerBridge for compatibility.
    """
    def __init__(self,
                 image_dim: int = 768,
                 lidar_dim: int = 256,
                 radar_dim: int = 256,
                 qformer_dim: int = 512,
                 llm_dim: int = 896,
                 num_query_tokens: int = 64,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 queries_per_task: int = 8,
                 expert_layers: int = 2,
                 router_temperature: float = 1.0):
        super().__init__()

        self.num_query_tokens = num_query_tokens
        self.qformer_dim = qformer_dim
        self.num_tasks = NUM_TASKS
        self.queries_per_task = queries_per_task
        self.router_temperature = router_temperature
        self.total_internal_queries = NUM_TASKS * queries_per_task  # 9 × 8 = 72

        # ── Modality projections ──
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, qformer_dim), nn.LayerNorm(qformer_dim))
        self.lidar_proj = nn.Sequential(
            nn.Linear(lidar_dim, qformer_dim), nn.LayerNorm(qformer_dim))
        self.radar_proj = nn.Sequential(
            nn.Linear(radar_dim, qformer_dim), nn.LayerNorm(qformer_dim))

        # ── Modality type embeddings ──
        self.image_type_embed = nn.Parameter(torch.randn(1, 1, qformer_dim) * 0.02)
        self.lidar_type_embed = nn.Parameter(torch.randn(1, 1, qformer_dim) * 0.02)
        self.radar_type_embed = nn.Parameter(torch.randn(1, 1, qformer_dim) * 0.02)

        # ── Task query tokens: [T, K, D] ──
        self.task_queries = nn.Parameter(
            torch.randn(NUM_TASKS, queries_per_task, qformer_dim) * 0.02
        )

        # ── Task-group embeddings: [T, D] ──
        self.task_group_embed = nn.Parameter(
            torch.randn(NUM_TASKS, qformer_dim) * 0.02
        )

        # ── 4 Expert branches ──
        # Each expert has its own cross-attention layers
        self.expert_branches = nn.ModuleDict({
            'i':   ExpertBranch(qformer_dim, expert_layers, num_heads, dropout),
            'il':  ExpertBranch(qformer_dim, expert_layers, num_heads, dropout),
            'ir':  ExpertBranch(qformer_dim, expert_layers, num_heads, dropout),
            'ilr': ExpertBranch(qformer_dim, expert_layers, num_heads, dropout),
        })

        # ── Modality router (for adaptive tasks) ──
        self.router = ModalityRouter(qformer_dim, NUM_EXPERTS)

        # ── Per-task attention pooling: compress K queries → N_out tokens ──
        # Distribute num_query_tokens across tasks
        tokens_per_task = self._compute_tokens_per_task(num_query_tokens)
        self.tokens_per_task = tokens_per_task

        self.task_pools = nn.ModuleList([
            AttentionPool(qformer_dim, n_out=tokens_per_task[t])
            for t in range(NUM_TASKS)
        ])

        # ── Output projection to LLM space ──
        self.output_proj = nn.Sequential(
            nn.Linear(qformer_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

        # ── Load-balancing aux loss weight ──
        self.router_aux_weight = 0.01

        self._init_weights()

    def _compute_tokens_per_task(self, total):
        """Distribute output tokens across tasks, more to adaptive/global tasks."""
        # Image-dominant tasks: fewer tokens (simpler)
        # Adaptive tasks: more tokens (complex scene understanding)
        # Global task: moderate
        base = total // NUM_TASKS  # 64 // 9 = 7
        remainder = total - base * NUM_TASKS  # 64 - 63 = 1

        tokens = [base] * NUM_TASKS
        # Give extra tokens to adaptive tasks first, then global
        priority = ADAPTIVE_TASKS + GLOBAL_TASKS + IMAGE_TASKS
        for i in range(remainder):
            tokens[priority[i % len(priority)]] += 1

        return tokens

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _build_expert_features(self, img_tokens, lid_tokens, rad_tokens):
        """
        Construct 4 expert feature sets from modality tokens.

        Returns:
            dict of expert_name → [B, N_tokens, D]
        """
        return {
            'i':   img_tokens,                                           # image only
            'il':  torch.cat([img_tokens, lid_tokens], dim=1),           # image + lidar
            'ir':  torch.cat([img_tokens, rad_tokens], dim=1),           # image + radar
            'ilr': torch.cat([img_tokens, lid_tokens, rad_tokens], dim=1),  # trimodal
        }

    def forward(self,
                image_features: torch.Tensor,
                lidar_features: torch.Tensor,
                radar_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [B, N_img, D_img]
            lidar_features: [B, D_lidar]
            radar_features: [B, D_radar]

        Returns:
            output_tokens: [B, num_query_tokens, D_llm]
        """
        B = image_features.shape[0]
        device = image_features.device

        # ── 1. Project modalities ──
        img_tokens = self.image_proj(image_features) + self.image_type_embed
        lid_tokens = self.lidar_proj(lidar_features.unsqueeze(1)) + self.lidar_type_embed
        rad_tokens = self.radar_proj(radar_features.unsqueeze(1)) + self.radar_type_embed

        # ── 2. Build expert feature sets ──
        expert_features = self._build_expert_features(img_tokens, lid_tokens, rad_tokens)

        # ── 3. Prepare task-aware queries ──
        # task_queries: [T, K, D] + task_group_embed: [T, D] → [T, K, D]
        queries = self.task_queries + self.task_group_embed.unsqueeze(1)
        queries = queries.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, K, D]

        # ── 4. Process each task group ──
        all_task_outputs = []
        router_logits_all = []

        # ─── 4a. Image-dominant tasks: only use image expert ───
        for t in IMAGE_TASKS:
            q_t = queries[:, t, :, :]  # [B, K, D]
            out_t = self.expert_branches['i'](q_t, expert_features['i'])
            all_task_outputs.append((t, out_t))

        # ─── 4b. Adaptive-routing tasks: router selects expert ───
        adaptive_queries = []
        adaptive_task_ids = []
        for t in ADAPTIVE_TASKS:
            adaptive_queries.append(queries[:, t, :, :])  # [B, K, D]
            adaptive_task_ids.extend([t] * self.queries_per_task)

        if adaptive_queries:
            # Stack all adaptive queries for batch routing
            aq = torch.cat(adaptive_queries, dim=1)  # [B, N_adaptive*K, D]

            # Router: predict expert selection
            weights, logits = self.router(aq, self.router_temperature)
            # weights: [B, N_adaptive*K, 4], logits: [B, N_adaptive*K, 4]
            router_logits_all.append(logits)

            # Process through each expert and combine via routing weights
            expert_outputs = []
            for ei, ename in enumerate(EXPERT_NAMES):
                out_e = self.expert_branches[ename](aq, expert_features[ename])
                expert_outputs.append(out_e)  # [B, N_adaptive*K, D]

            expert_stack = torch.stack(expert_outputs, dim=2)  # [B, N, 4, D]
            # Apply routing weights
            routed = (expert_stack * weights.unsqueeze(-1)).sum(dim=2)  # [B, N, D]

            # Split back into per-task chunks
            K = self.queries_per_task
            for i, t in enumerate(ADAPTIVE_TASKS):
                out_t = routed[:, i*K:(i+1)*K, :]  # [B, K, D]
                all_task_outputs.append((t, out_t))

        # ─── 4c. Global task: use trimodal expert ───
        for t in GLOBAL_TASKS:
            q_t = queries[:, t, :, :]
            out_t = self.expert_branches['ilr'](q_t, expert_features['ilr'])
            all_task_outputs.append((t, out_t))

        # ── 5. Sort by task index and compress ──
        all_task_outputs.sort(key=lambda x: x[0])

        compressed_tokens = []
        for t, out_t in all_task_outputs:
            pooled = self.task_pools[t](out_t)  # [B, tokens_per_task[t], D]
            compressed_tokens.append(pooled)

        # Concatenate all task tokens
        all_tokens = torch.cat(compressed_tokens, dim=1)  # [B, num_query_tokens, D]

        # ── 6. Project to LLM space ──
        output = self.output_proj(all_tokens)  # [B, num_query_tokens, D_llm]

        # ── 7. Store router aux loss for training ──
        if router_logits_all and self.training:
            self._router_aux_loss = self._compute_load_balance_loss(
                torch.cat(router_logits_all, dim=1)
            )
        else:
            self._router_aux_loss = torch.tensor(0.0, device=device)

        return output

    def _compute_load_balance_loss(self, logits):
        """
        Load-balancing auxiliary loss to prevent router collapse.

        Encourages equal utilization of all expert branches.

        Args:
            logits: [B, N, num_experts]
        """
        probs = F.softmax(logits, dim=-1)  # [B, N, 4]
        # Average probability per expert across all queries and batch
        avg_prob = probs.mean(dim=(0, 1))  # [4]
        # Fraction of queries routed to each expert
        assignments = probs.argmax(dim=-1)  # [B, N]
        freq = torch.zeros(NUM_EXPERTS, device=logits.device)
        for i in range(NUM_EXPERTS):
            freq[i] = (assignments == i).float().mean()

        # Switch loss: dot product of avg_prob and freq
        # Minimized when both are uniform (= 1/num_experts)
        return NUM_EXPERTS * (avg_prob * freq).sum()

    def get_router_loss(self):
        """Get the router auxiliary loss (call after forward)."""
        return self._router_aux_loss * self.router_aux_weight


# ═══════════════════════════════════════════════════════════
# Drop-in alias for backward compatibility
# ═══════════════════════════════════════════════════════════

# Use MoRoFormer as QFormerBridge replacement
QFormerBridge = MoRoFormer


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing MoRo-Former...")
    bridge = MoRoFormer(
        image_dim=768,
        lidar_dim=256,
        radar_dim=256,
        qformer_dim=512,
        llm_dim=896,
        num_query_tokens=64,
        num_layers=4,
        queries_per_task=8,
        expert_layers=2,
    ).to(device)

    # Simulate inputs
    img_feat = torch.randn(2, 256, 768).to(device)
    lidar_feat = torch.randn(2, 256).to(device)
    radar_feat = torch.randn(2, 256).to(device)

    bridge.train()
    out = bridge(img_feat, lidar_feat, radar_feat)
    router_loss = bridge.get_router_loss()
    print(f"  Output shape: {out.shape}")   # [2, 64, 896]
    print(f"  Router aux loss: {router_loss.item():.4f}")
    print(f"  Tokens per task: {bridge.tokens_per_task}")
    print(f"  Total params: {sum(p.numel() for p in bridge.parameters()):,}")

    # Inference mode
    bridge.eval()
    with torch.no_grad():
        out_eval = bridge(img_feat, lidar_feat, radar_feat)
    print(f"  Eval output shape: {out_eval.shape}")
    print("  MoRo-Former test passed!")
