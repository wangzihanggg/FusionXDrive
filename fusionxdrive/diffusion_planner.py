"""
Truncated Diffusion Planner — Dense Trajectory Version
=======================================================
Inspired by DiffusionDrive / DiffusionDriveV2.

KEY CHANGES from sparse 4-point version:
  - Output: 8 waypoints at 0.5s intervals (t=0.5, 1.0, ..., 4.0s)
  - GT: B-spline interpolated from sparse GNSS waypoints (1s, 2s, 5s, 10s)
  - Anchors: initialized via K-Means on B-spline interpolated training data
  - Denoiser: operates on 8×3=24 dim trajectory space
  - Smoothness loss: penalizes jerk (3rd derivative) for trajectory coherence

Pipeline:
    LLM planning tokens [B, num_plan_tokens, D_llm]
        → Attention pooling + MLP → condition z [B, D_cond]
        → Truncated Diffusion Decoder (anchor + 2-step denoise)
        → N_anchor × waypoints [B, N_anchor, 8, 3]
        → Anchor selector → best trajectory [B, 8, 3]

Waypoint format: (x, y, z) at t=0.5s, 1.0s, ..., 4.0s
    ego frame: x=forward, y=left, z=up
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from scipy.interpolate import make_interp_spline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Constants
# =============================================================================

# Dense output: 8 waypoints at 0.5s intervals
DENSE_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
NUM_DENSE_WP = 8

# Sparse GT time points (from UNISCP waypoints JSON)
SPARSE_TIMES = [0.0, 1.0, 2.0, 5.0, 10.0]


# =============================================================================
# B-spline interpolation utilities
# =============================================================================

def bspline_interpolate_trajectory(
    sparse_wp: torch.Tensor,
    sparse_times: list = None,
    dense_times: list = None,
) -> torch.Tensor:
    """
    Interpolate sparse waypoints to dense waypoints using B-spline.

    Args:
        sparse_wp: [B, N_sparse, 3] — waypoints at sparse_times (including t0)
                   Format: [[x0,y0,z0], [x1,y1,z1], ...] where first is t0=(0,0,0)
        sparse_times: time stamps for sparse_wp, default [0, 1, 2, 5, 10]
        dense_times: desired output times, default [0.5, 1.0, ..., 4.0]

    Returns:
        dense_wp: [B, N_dense, 3] — interpolated waypoints
    """
    if sparse_times is None:
        sparse_times = SPARSE_TIMES
    if dense_times is None:
        dense_times = DENSE_TIMES

    B = sparse_wp.shape[0]
    device = sparse_wp.device
    dtype = sparse_wp.dtype
    N_dense = len(dense_times)

    if HAS_SCIPY:
        # Use scipy B-spline (CPU, per-sample)
        import numpy as np
        t_sparse = np.array(sparse_times)
        t_dense = np.array(dense_times)

        result = torch.zeros(B, N_dense, 3, device=device, dtype=dtype)
        wp_np = sparse_wp.detach().cpu().numpy()

        for b in range(B):
            for d in range(3):  # x, y, z
                vals = wp_np[b, :, d]
                # Handle NaN/invalid
                if np.any(np.isnan(vals)):
                    result[b, :, d] = 0.0
                    continue
                k = min(3, len(t_sparse) - 1)
                try:
                    spline = make_interp_spline(t_sparse, vals, k=k)
                    result[b, :, d] = torch.from_numpy(
                        spline(t_dense).astype(np.float32)
                    ).to(device)
                except Exception:
                    # Fallback: linear interpolation
                    interp_vals = np.interp(t_dense, t_sparse, vals)
                    result[b, :, d] = torch.from_numpy(
                        interp_vals.astype(np.float32)
                    ).to(device)
        return result
    else:
        # Fallback: linear interpolation (pure torch)
        return _linear_interpolate(sparse_wp, sparse_times, dense_times)


def _linear_interpolate(
    sparse_wp: torch.Tensor,
    sparse_times: list,
    dense_times: list,
) -> torch.Tensor:
    """Pure-torch linear interpolation fallback."""
    B = sparse_wp.shape[0]
    device = sparse_wp.device
    dtype = sparse_wp.dtype

    t_s = torch.tensor(sparse_times, device=device, dtype=dtype)
    t_d = torch.tensor(dense_times, device=device, dtype=dtype)
    N_dense = len(dense_times)

    result = torch.zeros(B, N_dense, 3, device=device, dtype=dtype)

    for i, td in enumerate(t_d):
        # Find surrounding sparse points
        idx = torch.searchsorted(t_s, td).clamp(1, len(t_s) - 1)
        t0, t1 = t_s[idx - 1], t_s[idx]
        alpha = ((td - t0) / (t1 - t0 + 1e-8)).clamp(0, 1)
        result[:, i, :] = (1 - alpha) * sparse_wp[:, idx - 1, :] + alpha * sparse_wp[:, idx, :]

    return result


def sparse_to_dense_gt(
    sparse_4wp: torch.Tensor,
) -> torch.Tensor:
    """
    Convert sparse 4-point GT to dense 8-point GT.

    Args:
        sparse_4wp: [B, 4, 3] — waypoints at t=1s, 2s, 5s, 10s

    Returns:
        dense_8wp: [B, 8, 3] — waypoints at t=0.5s, 1.0s, ..., 4.0s
    """
    B = sparse_4wp.shape[0]
    device = sparse_4wp.device
    dtype = sparse_4wp.dtype

    # Prepend origin (t=0)
    origin = torch.zeros(B, 1, 3, device=device, dtype=dtype)
    sparse_5wp = torch.cat([origin, sparse_4wp], dim=1)  # [B, 5, 3]

    return bspline_interpolate_trajectory(sparse_5wp)


# =============================================================================
# Anchor generation: speed × curvature grid
# =============================================================================

def _arc_trajectory(speed: float, curvature: float,
                    times: torch.Tensor) -> torch.Tensor:
    """
    Generate a smooth arc trajectory given constant speed and curvature.

    Physics:
      - curvature κ = 0: straight line, x = v*t, y = 0
      - curvature κ ≠ 0: circular arc
          x = (1/κ) * sin(κ * v * t)
          y = (1/κ) * (1 - cos(κ * v * t))

    Args:
        speed: forward speed (m/s)
        curvature: signed curvature (rad/m), positive=left, negative=right
        times: [T] time stamps

    Returns:
        trajectory: [T, 3] waypoints (x, y, z=0)
    """
    T = len(times)
    z = torch.zeros(T)

    if abs(curvature) < 1e-6:
        # Straight line
        x = speed * times
        y = torch.zeros(T)
    else:
        R = 1.0 / curvature  # signed radius
        theta = speed * times * curvature  # angle swept
        x = R * torch.sin(theta)
        y = R * (1.0 - torch.cos(theta))

    return torch.stack([x, y, z], dim=-1)  # [T, 3]


def generate_default_anchors(
    num_anchors: int = 8,
    num_waypoints: int = NUM_DENSE_WP,
) -> torch.Tensor:
    """
    Generate anchors as a speed × curvature grid.

    Layout (default 16 candidates → select num_anchors):
      speeds:     [2, 5, 8, 12] m/s  (slow, medium, fast, very fast)
      curvatures: [−0.04, −0.01, 0, 0.01, 0.04] rad/m
                   (hard right, gentle right, straight, gentle left, hard left)

    This creates a fan-shaped spread from ego, like the reference image.
    """
    times = torch.tensor(DENSE_TIMES[:num_waypoints])

    # Speed × curvature grid
    speeds = [2.0, 5.0, 8.0, 12.0]
    curvatures = [-0.04, -0.015, 0.0, 0.015, 0.04]

    candidates = []
    for v in speeds:
        for k in curvatures:
            traj = _arc_trajectory(v, k, times)
            candidates.append(traj)

    # Select num_anchors with maximum coverage (greedy farthest point sampling)
    candidates = torch.stack(candidates, dim=0)  # [N_cand, T, 3]
    anchors = _farthest_point_sample(candidates, num_anchors)

    return anchors


def _farthest_point_sample(candidates: torch.Tensor, k: int) -> torch.Tensor:
    """
    Greedy farthest-point sampling to select k diverse anchors.
    Ensures coverage of both speed and curvature space.
    """
    N = candidates.shape[0]
    flat = candidates.reshape(N, -1)  # [N, 24]

    # Start with the straight medium-speed one (closest to median)
    norms = flat.norm(dim=1)
    median_norm = norms.median()
    selected = [(norms - median_norm).abs().argmin().item()]

    for _ in range(k - 1):
        # Distance from each candidate to nearest selected anchor
        sel_flat = flat[selected]  # [n_sel, 24]
        dists = torch.cdist(flat.unsqueeze(0), sel_flat.unsqueeze(0)).squeeze(0)  # [N, n_sel]
        min_dists = dists.min(dim=1).values  # [N]

        # Pick the farthest
        next_idx = min_dists.argmax().item()
        selected.append(next_idx)

    return candidates[selected]  # [k, T, 3]


def initialize_anchors_from_data(
    all_dense_trajs: torch.Tensor,
    num_anchors: int = 8,
) -> torch.Tensor:
    """
    Initialize anchors from training data: extract speed/curvature distribution,
    then generate a grid covering the data's actual range.

    Steps:
      1. For each GT trajectory, estimate speed and curvature
      2. Find the data's speed/curvature range
      3. Generate a speed × curvature grid covering that range
      4. Select num_anchors via farthest-point sampling

    Args:
        all_dense_trajs: [N, 8, 3] — all dense GT trajectories
        num_anchors: number of anchors

    Returns:
        anchors: [num_anchors, 8, 3]
    """
    import numpy as np
    trajs = all_dense_trajs.numpy()  # [N, 8, 3]
    N = trajs.shape[0]
    times_np = np.array(DENSE_TIMES)

    # ── Step 1: Estimate speed and curvature per trajectory ──
    speeds = []
    curvatures = []
    for i in range(N):
        x, y = trajs[i, :, 0], trajs[i, :, 1]

        # Average speed from total forward distance / total time
        total_dist = np.sqrt(np.diff(x)**2 + np.diff(y)**2).sum()
        avg_speed = total_dist / (times_np[-1] - times_np[0])
        speeds.append(avg_speed)

        # Curvature: fit y = f(x), estimate average curvature
        # κ ≈ y'' / (1 + y'^2)^1.5, but simpler: use final lateral offset
        if abs(x[-1]) > 1.0:  # avoid division by near-zero
            # Approximate curvature from arc: κ ≈ 2*y_final / (x_final^2 + y_final^2)
            kappa = 2.0 * y[-1] / (x[-1]**2 + y[-1]**2 + 1e-6)
        else:
            kappa = 0.0
        curvatures.append(np.clip(kappa, -0.1, 0.1))

    speeds = np.array(speeds)
    curvatures = np.array(curvatures)

    # ── Step 2: Determine grid range from data ──
    sp_min, sp_max = max(0.5, np.percentile(speeds, 5)), np.percentile(speeds, 95)
    kp_min, kp_max = np.percentile(curvatures, 5), np.percentile(curvatures, 95)

    # Ensure curvature range is symmetric and has minimum spread
    kp_abs = max(abs(kp_min), abs(kp_max), 0.01)
    kp_min, kp_max = -kp_abs, kp_abs

    # ── Step 3: Generate grid ──
    # More speeds than curvatures (speed variation usually larger)
    n_speed = max(3, int(np.sqrt(num_anchors * 2)))
    n_curv = max(3, int(np.sqrt(num_anchors * 1.5)))

    speed_grid = np.linspace(sp_min, sp_max, n_speed)
    curv_grid = np.linspace(kp_min, kp_max, n_curv)

    times_t = torch.tensor(DENSE_TIMES, dtype=torch.float32)
    candidates = []
    for v in speed_grid:
        for k in curv_grid:
            traj = _arc_trajectory(float(v), float(k), times_t)
            candidates.append(traj)

    candidates = torch.stack(candidates, dim=0)  # [n_speed * n_curv, 8, 3]

    # ── Step 4: Select diverse anchors ──
    anchors = _farthest_point_sample(candidates, num_anchors)

    # ── Log ──
    print(f"  Data stats: speed [{sp_min:.1f}, {sp_max:.1f}] m/s, "
          f"curvature [{-kp_abs:.4f}, {kp_abs:.4f}] rad/m")
    print(f"  Grid: {n_speed} speeds × {n_curv} curvatures = "
          f"{len(candidates)} candidates → {num_anchors} anchors")
    for i, a in enumerate(anchors):
        a_np = a.numpy()
        sp = np.sqrt(np.diff(a_np[:,0])**2 + np.diff(a_np[:,1])**2).sum() / 3.5
        max_y = np.abs(a_np[:, 1]).max()
        mean_y = a_np[:, 1].mean()
        tag = "LEFT" if mean_y > 0.1 else ("RIGHT" if mean_y < -0.1 else "STRAIGHT")
        print(f"    Anchor {i}: speed≈{sp:.1f}m/s  max|y|={max_y:.2f}m  → {tag}")

    return anchors


def visualize_anchors(anchors: torch.Tensor, save_path: str,
                      gt_trajs: torch.Tensor = None):
    """
    Visualize anchor trajectories in BEV, fan-shaped from ego origin.
    Optionally overlay sampled GT trajectories for comparison.

    Args:
        anchors: [K, 8, 3] anchor trajectories
        save_path: path to save the figure
        gt_trajs: optional [N, 8, 3] GT trajectories (will sample ~200)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    K = anchors.shape[0]
    anc = anchors.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor='#0a0a18')

    # ── Left: Anchor-only view ──
    ax = axes[0]
    ax.set_facecolor('#0f0f23')

    # Grid
    for d in range(5, 55, 5):
        ax.axhline(y=d, color='#1e1e3a', lw=0.3, alpha=0.5)
        ax.axvline(x=d, color='#1e1e3a', lw=0.3, alpha=0.4)
        ax.axvline(x=-d, color='#1e1e3a', lw=0.3, alpha=0.4)
    ax.axhline(y=0, color='#2a2a4a', lw=0.8)
    ax.axvline(x=0, color='#2a2a4a', lw=0.8)

    # Ego vehicle
    tri = plt.Polygon([(0,1.2),(-.5,-.5),(0,-.2),(.5,-.5)],
                       color='#27ae60', alpha=0.9, zorder=10, ec='#1abc9c', lw=1)
    ax.add_patch(tri)

    # Color map for anchors
    cmap = plt.cm.get_cmap('tab10', K)
    for i in range(K):
        x, y = anc[i, :, 0], anc[i, :, 1]
        px, py = -y, x  # ego→plot: x→up, y→-left

        # Trajectory line
        ax.plot(px, py, '-', color=cmap(i), lw=3, alpha=0.85, zorder=6)
        # Start/end dots
        ax.scatter(px[0], py[0], c=[cmap(i)], s=60, zorder=8, edgecolors='white', lw=1)
        ax.scatter(px[-1], py[-1], c=[cmap(i)], s=100, zorder=8,
                   edgecolors='white', lw=1.5, marker='D')

        # Label
        sp = np.sqrt(np.diff(x)**2 + np.diff(y)**2).sum() / 3.5
        max_y_val = np.abs(y).max()
        tag = f"A{i}: {sp:.1f}m/s"
        if max_y_val > 0.1:
            tag += f" {'L' if y.mean()>0 else 'R'}{max_y_val:.1f}m"
        ax.annotate(tag, (px[-1], py[-1]), textcoords="offset points",
                    xytext=(8, 3), fontsize=7, color=cmap(i), fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#0f0f23', alpha=0.8))

    all_x = anc[:, :, 0]
    all_y = anc[:, :, 1]
    fwd_max = max(all_x.max(), 15) * 1.15
    lat_max = max(abs(all_y).max(), 3) * 1.5

    ax.set_xlim(-lat_max, lat_max)
    ax.set_ylim(-2, fwd_max)
    ax.set_title(f'{K} Anchors (speed × curvature)', fontsize=13,
                 color='white', fontweight='bold')
    ax.set_xlabel('← Right    Lateral (m)    Left →', fontsize=9, color='#888899')
    ax.set_ylabel('Forward (m) →', fontsize=9, color='#888899')
    ax.tick_params(colors='#666688', labelsize=7)
    for sp in ax.spines.values(): sp.set_color('#1e1e3a')

    # ── Right: Anchors + GT overlay ──
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f23')

    for d in range(5, 55, 5):
        ax2.axhline(y=d, color='#1e1e3a', lw=0.3, alpha=0.5)
        ax2.axvline(x=d, color='#1e1e3a', lw=0.3, alpha=0.4)
        ax2.axvline(x=-d, color='#1e1e3a', lw=0.3, alpha=0.4)
    ax2.axhline(y=0, color='#2a2a4a', lw=0.8)
    ax2.axvline(x=0, color='#2a2a4a', lw=0.8)

    tri2 = plt.Polygon([(0,1.2),(-.5,-.5),(0,-.2),(.5,-.5)],
                        color='#27ae60', alpha=0.9, zorder=10, ec='#1abc9c', lw=1)
    ax2.add_patch(tri2)

    # Plot sampled GT trajectories
    if gt_trajs is not None:
        gt = gt_trajs.numpy()
        n_show = min(300, len(gt))
        idx = np.random.choice(len(gt), n_show, replace=False)
        for j in idx:
            x, y = gt[j, :, 0], gt[j, :, 1]
            ax2.plot(-y, x, '-', color='#555577', lw=0.5, alpha=0.3, zorder=3)

    # Overlay anchors (thicker)
    for i in range(K):
        x, y = anc[i, :, 0], anc[i, :, 1]
        ax2.plot(-y, x, '-', color=cmap(i), lw=3.5, alpha=0.9, zorder=7)
        ax2.scatter(-y[-1], x[-1], c=[cmap(i)], s=80, zorder=8,
                    edgecolors='white', lw=1.5, marker='D')

    ax2.set_xlim(-lat_max, lat_max)
    ax2.set_ylim(-2, fwd_max)
    ax2.set_title(f'Anchors + GT trajectories (n={len(gt_trajs) if gt_trajs is not None else 0})',
                  fontsize=13, color='white', fontweight='bold')
    ax2.set_xlabel('← Right    Lateral (m)    Left →', fontsize=9, color='#888899')
    ax2.set_ylabel('Forward (m) →', fontsize=9, color='#888899')
    ax2.tick_params(colors='#666688', labelsize=7)
    for sp in ax2.spines.values(): sp.set_color('#1e1e3a')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Anchor visualization saved to {save_path}")


# =============================================================================
# Smoothness loss
# =============================================================================

def trajectory_smoothness_loss(traj: torch.Tensor, dt: float = 0.5) -> torch.Tensor:
    """
    Penalize jerk (3rd derivative) for trajectory smoothness.

    Args:
        traj: [B, 8, 3] — dense waypoints
        dt: time step between consecutive points (0.5s)

    Returns:
        scalar loss
    """
    # Velocity: 1st difference
    vel = (traj[:, 1:, :] - traj[:, :-1, :]) / dt      # [B, 7, 3]
    # Acceleration: 2nd difference
    acc = (vel[:, 1:, :] - vel[:, :-1, :]) / dt          # [B, 6, 3]
    # Jerk: 3rd difference
    jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / dt          # [B, 5, 3]

    return jerk.pow(2).mean()


# =============================================================================
# Denoising network
# =============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timestep."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DenoiseBlock(nn.Module):
    """Denoising MLP block with FiLM conditioning and residual."""
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.cond_proj = nn.Linear(cond_dim, dim * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        scale_shift = self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h


class TrajectoryDenoiser(nn.Module):
    """
    Trajectory denoising network for dense 8-point trajectories.
    Input: flattened noisy trajectory (8×3=24) + condition + timestep
    Output: predicted clean trajectory offset from anchor (24 dim)
    """
    def __init__(
        self,
        num_waypoints: int = NUM_DENSE_WP,
        waypoint_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        traj_dim = num_waypoints * waypoint_dim  # 8 * 3 = 24

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trajectory input projection
        self.traj_proj = nn.Linear(traj_dim, hidden_dim)

        # Condition: cond_dim + time_dim
        full_cond_dim = cond_dim + hidden_dim

        # Denoising blocks
        self.blocks = nn.ModuleList([
            DenoiseBlock(hidden_dim, full_cond_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, traj_dim),
        )

    def forward(
        self,
        noisy_traj: torch.Tensor,   # [B, 24]
        timestep: torch.Tensor,      # [B]
        condition: torch.Tensor,     # [B, cond_dim]
    ) -> torch.Tensor:
        t_emb = self.time_embed(timestep)
        t_emb = self.time_proj(t_emb)
        cond = torch.cat([condition, t_emb], dim=-1)
        h = self.traj_proj(noisy_traj)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(h)  # [B, 24]


# =============================================================================
# Main Planner
# =============================================================================

class TruncatedDiffusionPlanner(nn.Module):
    """
    Truncated Diffusion Planner — Dense Trajectory Version.

    Output: 8 waypoints at 0.5s intervals (t=0.5, 1.0, ..., 4.0s)
    Anchors: B-spline initialized, representing driving intents
    Diffusion: anchor-based truncated diffusion with 2-step DDIM

    The condition comes from LLM planning tokens + QFormer tokens.
    """

    def __init__(
        self,
        llm_hidden_dim: int = 896,
        num_planning_tokens: int = 4,
        num_waypoints: int = NUM_DENSE_WP,   # 8
        waypoint_dim: int = 3,
        num_anchors: int = 8,
        cond_dim: int = 256,
        denoise_hidden: int = 256,
        denoise_blocks: int = 4,
        t_trunc: int = 5,
        n_infer_steps: int = 2,
        dropout: float = 0.1,
        smoothness_weight: float = 0.1,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.num_anchors = num_anchors
        self.traj_dim = num_waypoints * waypoint_dim  # 8 * 3 = 24
        self.t_trunc = t_trunc
        self.n_infer_steps = n_infer_steps
        self.smoothness_weight = smoothness_weight

        # ── Condition encoder ──
        self.cond_attn_pool = nn.Linear(llm_hidden_dim, 1)
        self.condition_encoder = nn.Sequential(
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(llm_hidden_dim),
            nn.Linear(llm_hidden_dim, cond_dim),
            nn.GELU(),
        )

        # ── Anchor trajectories ──
        default_anchors = generate_default_anchors(num_anchors, num_waypoints)
        self.anchors = nn.Parameter(default_anchors)  # [N_anchor, 8, 3]

        # ── Denoiser ──
        self.denoiser = TrajectoryDenoiser(
            num_waypoints=num_waypoints,
            waypoint_dim=waypoint_dim,
            hidden_dim=denoise_hidden,
            cond_dim=cond_dim,
            num_blocks=denoise_blocks,
            dropout=dropout,
        )

        # ── Anchor scorer ──
        self.anchor_scorer = nn.Sequential(
            nn.Linear(cond_dim + self.traj_dim, denoise_hidden),
            nn.GELU(),
            nn.LayerNorm(denoise_hidden),
            nn.Linear(denoise_hidden, 1),
        )

        # ── Direct regression head ──
        self.direct_head = nn.Sequential(
            nn.Linear(cond_dim, denoise_hidden),
            nn.GELU(),
            nn.Linear(denoise_hidden, denoise_hidden),
            nn.GELU(),
            nn.Linear(denoise_hidden, self.traj_dim),
        )

        # ── Noise schedule ──
        betas = torch.linspace(1e-4, 0.02, t_trunc)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # ── Dense time offsets (normalized) ──
        self.register_buffer(
            'waypoint_times',
            torch.tensor(DENSE_TIMES, dtype=torch.float32),
        )

    # ──────────────────────────────────────────────────────
    # Forward diffusion
    # ──────────────────────────────────────────────────────

    def _diffuse(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alphas_cumprod[t - 1].unsqueeze(-1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt, noise

    # ──────────────────────────────────────────────────────
    # Assign GT to closest anchor
    # ──────────────────────────────────────────────────────

    def _assign_anchors(
        self, gt_traj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        gt_traj: [B, 8, 3] (dense waypoints)
        Returns: anchor_idx [B], offset [B, traj_dim]
        """
        B = gt_traj.shape[0]
        gt_flat = gt_traj.reshape(B, -1)
        anchors_flat = self.anchors.reshape(self.num_anchors, -1)

        dist = torch.cdist(
            gt_flat.unsqueeze(0), anchors_flat.unsqueeze(0)
        ).squeeze(0)
        anchor_idx = dist.argmin(dim=1)

        assigned_anchor = self.anchors[anchor_idx].reshape(B, -1)
        offset = gt_flat - assigned_anchor

        return anchor_idx, offset

    # ──────────────────────────────────────────────────────
    # Training forward
    # ──────────────────────────────────────────────────────

    def forward(
        self,
        planning_tokens: torch.Tensor,
        gt_waypoints: Optional[torch.Tensor] = None,
        gt_is_dense: bool = False,
    ) -> dict:
        """
        Args:
            planning_tokens: [B, N_tokens, D_llm]
            gt_waypoints: [B, 4, 3] (sparse) or [B, 8, 3] (dense)
            gt_is_dense: if True, gt_waypoints is already 8-point dense
        """
        B = planning_tokens.shape[0]
        device = planning_tokens.device

        # 1. Condition encoding
        attn_weights = F.softmax(
            self.cond_attn_pool(planning_tokens), dim=1
        )
        pooled = (planning_tokens * attn_weights).sum(dim=1)
        cond = self.condition_encoder(pooled)

        if gt_waypoints is not None:
            # ── Training mode ──

            # Convert sparse GT to dense if needed
            # Convert sparse GT to dense if needed (auto-detect by shape)
            if gt_waypoints.shape[1] >= 8:
                gt_dense = gt_waypoints[:, :8, :]  # already dense [B, 8, 3]
            else:
                gt_dense = sparse_to_dense_gt(gt_waypoints)  # [B, 4, 3] → [B, 8, 3]

            # 2. Assign GT to anchor
            anchor_idx, offset = self._assign_anchors(gt_dense)

            # 3. Sample timestep
            t = torch.randint(1, self.t_trunc + 1, (B,), device=device)

            # 4. Add noise
            noisy_offset, noise = self._diffuse(offset, t)

            # 5. Denoise
            pred_offset = self.denoiser(noisy_offset, t, cond)

            # 6. Reconstruction loss (offset space)
            rec_loss = F.smooth_l1_loss(pred_offset, offset)

            # 7. Trajectory ADE loss
            pred_traj = self._offset_to_waypoints(pred_offset, anchor_idx)
            disp = torch.sqrt(
                (pred_traj[:, :, 0] - gt_dense[:, :, 0]) ** 2
                + (pred_traj[:, :, 1] - gt_dense[:, :, 1]) ** 2
                + 1e-6
            )  # [B, 8]

            # Weight: later points slightly more important
            wp_weights = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
                device=device,
            )
            ade_loss = (disp * wp_weights.unsqueeze(0)).mean()

            # 8. Smoothness loss on predicted trajectory
            smooth_loss = trajectory_smoothness_loss(pred_traj)

            # 9. Anchor scoring loss
            anchor_scores = self._score_all_anchors(cond)
            target_scores = F.one_hot(anchor_idx, self.num_anchors).float()
            score_loss = F.binary_cross_entropy_with_logits(
                anchor_scores, target_scores
            )

            # 10. Direct regression
            direct_pred = self.direct_head(cond).reshape(B, self.num_waypoints, self.waypoint_dim)
            direct_disp = torch.sqrt(
                (direct_pred[:, :, 0] - gt_dense[:, :, 0]) ** 2
                + (direct_pred[:, :, 1] - gt_dense[:, :, 1]) ** 2
                + 1e-6
            )
            direct_loss = (direct_disp * wp_weights.unsqueeze(0)).mean()
            direct_smooth_loss = trajectory_smoothness_loss(direct_pred)

            # 11. Total loss — diffusion is PRIMARY, direct is auxiliary
            loss = (
                1.0 * rec_loss
                + 2.0 * ade_loss
                + 0.5 * score_loss
                + 0.3 * direct_loss
                + self.smoothness_weight * (smooth_loss + 0.3 * direct_smooth_loss)
            )

            with torch.no_grad():
                pred_wp_log = self._offset_to_waypoints(pred_offset, anchor_idx)

            return {
                'loss': loss,
                'rec_loss': rec_loss,
                'ade_loss': ade_loss,
                'smooth_loss': smooth_loss,
                'score_loss': score_loss,
                'direct_loss': direct_loss,
                'pred_waypoints': pred_wp_log,   # [B, 8, 3]
                'anchor_scores': anchor_scores,
                'gt_dense': gt_dense,            # for logging
            }

        else:
            # ── Inference ──
            return self.generate(cond)

    # ──────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> dict:
        B = cond.shape[0]
        device = cond.device

        # ═══ Anchor-based diffusion (PRIMARY output) ═══
        anchor_scores = self._score_all_anchors(cond)
        best_anchor_idx = anchor_scores.argmax(dim=1)

        all_trajs = []
        for k in range(self.num_anchors):
            anchor_k = self.anchors[k].reshape(1, -1).expand(B, -1)

            # Truncated diffusion inference:
            # Training: denoiser sees xt = sqrt(αbar)*offset + sqrt(1-αbar)*noise
            #   At t=1: αbar≈0.9999, so xt ≈ offset (nearly clean)
            #   At t=5: αbar≈0.95,  so xt ≈ 0.975*offset + 0.224*noise
            #
            # Inference strategy: pass zero offset at t=1
            #   = "anchor is our current best guess, predict the correction"
            # This is correct because at t=1 the denoiser expects nearly-clean input
            t_one = torch.ones(B, dtype=torch.long, device=device)
            zero_offset = torch.zeros(B, self.traj_dim, device=device)
            pred_offset = self.denoiser(zero_offset, t_one, cond)

            traj_k = (anchor_k + pred_offset).reshape(
                B, self.num_waypoints, self.waypoint_dim
            )
            all_trajs.append(traj_k)

        all_trajs = torch.stack(all_trajs, dim=1)  # [B, N_anchor, 8, 3]
        diffusion_best = all_trajs[
            torch.arange(B, device=device), best_anchor_idx
        ]

        # ═══ Direct regression (auxiliary / fallback) ═══
        direct_pred = self.direct_head(cond)
        direct_traj = direct_pred.reshape(B, self.num_waypoints, self.waypoint_dim)

        return {
            'pred_waypoints': diffusion_best,    # PRIMARY: diffusion output
            'direct_traj': direct_traj,           # auxiliary: direct MLP
            'all_waypoints': all_trajs,           # all anchor trajectories
            'anchor_scores': anchor_scores,
            'best_anchor_idx': best_anchor_idx,
        }

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _score_all_anchors(self, cond: torch.Tensor) -> torch.Tensor:
        B = cond.shape[0]
        scores = []
        for k in range(self.num_anchors):
            anchor_flat = self.anchors[k].reshape(1, -1).expand(B, -1)
            inp = torch.cat([cond, anchor_flat], dim=-1)
            score = self.anchor_scorer(inp).squeeze(-1)
            scores.append(score)
        return torch.stack(scores, dim=1)

    def _offset_to_waypoints(
        self, offset: torch.Tensor, anchor_idx: torch.Tensor
    ) -> torch.Tensor:
        B = offset.shape[0]
        assigned_anchor = self.anchors[anchor_idx].reshape(B, -1)
        traj = assigned_anchor + offset
        return traj.reshape(B, self.num_waypoints, self.waypoint_dim)

    def load_anchors_from_data(self, all_dense_trajs: torch.Tensor,
                               save_dir: str = None):
        """
        Re-initialize anchors from training data (speed × curvature grid).
        Optionally visualize and save to save_dir.

        Args:
            all_dense_trajs: [N, 8, 3] — dense GT trajectories
            save_dir: if provided, save anchor visualization here
        """
        new_anchors = initialize_anchors_from_data(
            all_dense_trajs.cpu(), self.num_anchors
        )
        self.anchors.data.copy_(new_anchors.to(self.anchors.device))
        print(f"[Planner] Anchors re-initialized from {all_dense_trajs.shape[0]} samples")

        # Visualize
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'anchor_visualization.png')
            visualize_anchors(new_anchors, save_path, gt_trajs=all_dense_trajs.cpu())


# =============================================================================
# Planning loss function (external API)
# =============================================================================

def planning_loss_fn(
    pred_waypoints: torch.Tensor,
    gt_waypoints: torch.Tensor,
) -> torch.Tensor:
    """
    Compute weighted planning loss for 8-point dense trajectory.

    Args:
        pred_waypoints: [B, 8, 3]
        gt_waypoints:   [B, 8, 3] (dense) or [B, 4, 3] (sparse)

    Returns:
        scalar loss
    """
    # Auto-detect: if gt has 4 points, convert to dense
    if gt_waypoints.shape[1] == 4:
        gt_waypoints = sparse_to_dense_gt(gt_waypoints)

    gt = gt_waypoints.to(pred_waypoints.dtype).to(pred_waypoints.device)

    # Per-step weights
    step_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
        device=pred_waypoints.device,
    ).view(1, 8, 1)

    error = F.smooth_l1_loss(pred_waypoints, gt, reduction='none')
    loss = (error * step_weights).mean()

    # Add smoothness
    smooth = trajectory_smoothness_loss(pred_waypoints)
    loss = loss + 0.1 * smooth

    return loss