"""
Point Cloud Encoders:
  - VoxelNeXt-style encoder for LiDAR point clouds
  - RadarNeXt-style encoder for 4D mmWave radar point clouds

Based on:
  [1] VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection (Chen et al., 2023)
  [2] RadarNeXt: Real-Time and Reliable 3D Object Detector Based On 4D mmWave Radar (Jia et al., 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


# =============================================================================
# Common utilities
# =============================================================================

class PillarEncoder(nn.Module):
    """
    PointPillars-style encoder: converts raw point cloud into pillar features
    then scatters into a pseudo-image (BEV representation).
    
    Used by RadarNeXt for radar point clouds.
    Also provides a voxelization path for VoxelNeXt (LiDAR).
    """
    def __init__(self, 
                 in_channels: int,
                 feat_channels: int = 64,
                 pc_range: list = None,
                 pillar_size: list = None,
                 max_points_per_pillar: int = 32,
                 max_pillars: int = 16000):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.pc_range = pc_range  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.pillar_size = pillar_size  # [dx, dy, dz]
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        
        # Compute grid size
        self.grid_x = int((pc_range[3] - pc_range[0]) / pillar_size[0])
        self.grid_y = int((pc_range[4] - pc_range[1]) / pillar_size[1])
        
        # PointNet-like feature extractor for each pillar
        # Input: raw features + relative offset to pillar center + distance to origin
        augmented_channels = in_channels + 5  # +3 offset to center, +2 distance offset
        self.pfn = nn.Sequential(
            nn.Linear(augmented_channels, feat_channels),
            nn.LayerNorm(feat_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, points_batch: list) -> torch.Tensor:
        """
        Args:
            points_batch: list of [N_i, C] tensors, one per batch item
        Returns:
            pseudo_image: [B, feat_channels, grid_y, grid_x] BEV feature map
        """
        batch_size = len(points_batch)
        device = points_batch[0].device
        dtype = points_batch[0].dtype
        
        pseudo_images = []
        for b in range(batch_size):
            points = points_batch[b]  # [N, C]
            bev = self._create_pseudo_image_single(points, device, dtype)
            pseudo_images.append(bev)
        
        return torch.stack(pseudo_images, dim=0)  # [B, C, H, W]
    
    def _create_pseudo_image_single(self, points, device, dtype):
        """Create BEV pseudo-image for a single sample."""
        if points.shape[0] == 0:
            return torch.zeros(self.feat_channels, self.grid_y, self.grid_x,
                             device=device, dtype=dtype)
        
        # Clip to range
        x_min, y_min, z_min = self.pc_range[0], self.pc_range[1], self.pc_range[2]
        x_max, y_max, z_max = self.pc_range[3], self.pc_range[4], self.pc_range[5]
        
        mask = ((points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] < z_max))
        points = points[mask]
        
        if points.shape[0] == 0:
            return torch.zeros(self.feat_channels, self.grid_y, self.grid_x,
                             device=device, dtype=dtype)
        
        # Compute pillar indices
        ix = ((points[:, 0] - x_min) / self.pillar_size[0]).long()
        iy = ((points[:, 1] - y_min) / self.pillar_size[1]).long()
        ix = ix.clamp(0, self.grid_x - 1)
        iy = iy.clamp(0, self.grid_y - 1)
        
        # Pillar center coordinates
        cx = x_min + (ix.float() + 0.5) * self.pillar_size[0]
        cy = y_min + (iy.float() + 0.5) * self.pillar_size[1]
        cz = (z_min + z_max) / 2.0
        
        # Augmented features: [original, offset_x, offset_y, offset_z, dist_x, dist_y]
        offset_x = points[:, 0] - cx
        offset_y = points[:, 1] - cy
        offset_z = points[:, 2] - cz
        dist_x = points[:, 0]
        dist_y = points[:, 1]
        
        augmented = torch.cat([
            points,
            offset_x.unsqueeze(-1),
            offset_y.unsqueeze(-1),
            offset_z.unsqueeze(-1),
            dist_x.unsqueeze(-1),
            dist_y.unsqueeze(-1),
        ], dim=-1)  # [N, C+5]
        
        # Simple scatter: average pooling per pillar
        pillar_idx = iy * self.grid_x + ix  # [N]
        
        # Apply PFN per point then scatter
        # Linear: [N, augmented_channels] -> [N, feat_channels]
        feat = self.pfn[0](augmented)  # Linear
        feat = self.pfn[1](feat)       # LayerNorm
        feat = self.pfn[2](feat)       # ReLU
        
        # Scatter max into BEV grid
        bev = torch.zeros(self.grid_y * self.grid_x, self.feat_channels,
                         device=device, dtype=dtype)
        pillar_idx_expanded = pillar_idx.unsqueeze(-1).expand(-1, self.feat_channels)
        bev.scatter_reduce_(0, pillar_idx_expanded, feat, reduce='amax', include_self=True)
        
        bev = bev.view(self.grid_y, self.grid_x, self.feat_channels)
        bev = bev.permute(2, 0, 1)  # [C, H, W]
        
        return bev


# =============================================================================
# VoxelNeXt Encoder (for LiDAR)
# =============================================================================

class SparseConvBlock(nn.Module):
    """Simulates a sparse conv block using standard 2D convolutions on BEV.
    
    For simplicity and compatibility (no spconv dependency), we use dense 2D
    convolutions on the BEV pseudo-image. This captures the key ideas from
    VoxelNeXt: multi-stage downsampling to enlarge receptive fields, and
    sparse height compression (done in PillarEncoder).
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class VoxelNeXtEncoder(nn.Module):
    """
    VoxelNeXt-style encoder for LiDAR point clouds.
    
    Key ideas from the paper:
    1. Pillarization -> BEV pseudo-image
    2. 6-stage sparse CNN backbone with strides {1,2,4,8,16,32}
       (4 base stages + 2 additional downsampling for larger receptive fields)
    3. Multi-scale feature concatenation from last 3 stages (F4, F5, F6)
    4. Sparse height compression (handled by pillar encoder)
    
    LiDAR input features: x, y, z, intensity (4 channels)
    """
    def __init__(self, 
                 output_dim: int = 256,
                 pc_range: list = None,
                 pillar_size: list = None):
        super().__init__()
        
        # Default range for LiDAR
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if pillar_size is None:
            pillar_size = [0.2, 0.2, 8.0]
        
        self.pc_range = pc_range
        self.pillar_size = pillar_size
        
        # LiDAR: x, y, z, intensity = 4 channels
        self.pillar_encoder = PillarEncoder(
            in_channels=4,
            feat_channels=64,
            pc_range=pc_range,
            pillar_size=pillar_size,
            max_points_per_pillar=32,
            max_pillars=40000,
        )
        
        # VoxelNeXt backbone: 6 stages
        # Stage 1: stride 1 (64 -> 16ch)
        self.stage1 = nn.Sequential(
            SparseConvBlock(64, 16, stride=1),
            SparseConvBlock(16, 16, stride=1),
        )
        # Stage 2: stride 2 (16 -> 32ch)
        self.stage2 = nn.Sequential(
            SparseConvBlock(16, 32, stride=2),
            SparseConvBlock(32, 32, stride=1),
        )
        # Stage 3: stride 4 (32 -> 64ch)
        self.stage3 = nn.Sequential(
            SparseConvBlock(32, 64, stride=2),
            SparseConvBlock(64, 64, stride=1),
        )
        # Stage 4: stride 8 (64 -> 128ch)
        self.stage4 = nn.Sequential(
            SparseConvBlock(64, 128, stride=2),
            SparseConvBlock(128, 128, stride=1),
        )
        # Stage 5: stride 16 (128 -> 128ch) — additional downsampling
        self.stage5 = nn.Sequential(
            SparseConvBlock(128, 128, stride=2),
            SparseConvBlock(128, 128, stride=1),
        )
        # Stage 6: stride 32 (128 -> 128ch) — additional downsampling
        self.stage6 = nn.Sequential(
            SparseConvBlock(128, 128, stride=2),
            SparseConvBlock(128, 128, stride=1),
        )
        
        # Upsample F5 and F6 to match F4's spatial resolution
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Fuse F4 + F5 + F6 -> output_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(128 * 3, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling to get a fixed-size feature
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = output_dim
    
    def forward(self, points_batch: list) -> torch.Tensor:
        """
        Args:
            points_batch: list of [N_i, 4] tensors (x, y, z, intensity)
        Returns:
            features: [B, output_dim] global feature vector
        """
        bev = self.pillar_encoder(points_batch)  # [B, 64, H, W]
        
        f1 = self.stage1(bev)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)   # stride 8
        f5 = self.stage5(f4)   # stride 16
        f6 = self.stage6(f5)   # stride 32
        
        # Upsample F5, F6 to F4's resolution
        f5_up = self.up5(f5)
        f6_up = self.up6(f6)
        
        # Handle potential size mismatch
        h4, w4 = f4.shape[2], f4.shape[3]
        f5_up = F.interpolate(f5_up, size=(h4, w4), mode='bilinear', align_corners=False)
        f6_up = F.interpolate(f6_up, size=(h4, w4), mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        fused = torch.cat([f4, f5_up, f6_up], dim=1)  # [B, 384, H/8, W/8]
        fused = self.fuse(fused)  # [B, output_dim, H/8, W/8]
        
        # Global average pooling
        out = self.pool(fused).squeeze(-1).squeeze(-1)  # [B, output_dim]
        return out


# =============================================================================
# RadarNeXt Encoder (for 4D mmWave Radar)
# =============================================================================

class RepDWCBlock(nn.Module):
    """
    Re-parameterizable Depthwise Convolution block from MobileOne,
    as used in RadarNeXt backbone.
    
    Training: multi-branch (3x3 DW + 1x1 DW + identity)
    Inference: re-parameterized to single 3x3 conv
    
    For simplicity, we implement the training-mode multi-branch version.
    """
    def __init__(self, channels, stride=1):
        super().__init__()
        self.stride = stride
        self.channels = channels
        
        # Depthwise 3x3 branch
        self.dw3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1,
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # Depthwise 1x1 branch
        self.dw1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=stride, padding=0,
                     groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        # Identity branch (only when stride=1)
        self.has_identity = (stride == 1)
        if self.has_identity:
            self.identity_bn = nn.BatchNorm2d(channels)
        
        # Pointwise 1x1 to mix channels
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Depthwise: sum of branches
        out = self.dw3x3(x) + self.dw1x1(x)
        if self.has_identity:
            out = out + self.identity_bn(x)
        out = self.act(out)
        
        # Pointwise
        out = self.act(self.pw(out))
        return out


class RepDWCDownBlock(nn.Module):
    """Rep-DWC block with channel expansion and stride-2 downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1x1 to expand channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Stride-2 depthwise
        self.dw_down = RepDWCBlock(out_ch, stride=2)
    
    def forward(self, x):
        x = self.expand(x)
        x = self.dw_down(x)
        return x


class DCNv3Simple(nn.Module):
    """
    Simplified Deformable Convolution v3 for foreground enhancement.
    
    Key idea from RadarNeXt: use multi-group deformable convolutions
    with learnable offsets and modulation weights to adaptively select
    features, enhancing foreground object representations.
    
    This is a simplified version using standard deformable conv concepts.
    """
    def __init__(self, channels, num_groups=4, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Offset prediction: 2 * K * K offsets per group
        self.offset_conv = nn.Conv2d(
            channels,
            num_groups * 2 * kernel_size * kernel_size,
            3, padding=1,
        )
        # Modulation weight prediction
        self.mask_conv = nn.Conv2d(
            channels,
            num_groups * kernel_size * kernel_size,
            3, padding=1,
        )
        # Actual convolution (group conv)
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size, padding=self.padding,
            groups=num_groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        
        # Initialize offsets to zero
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)
    
    def forward(self, x):
        """Simplified DCNv3: predict offsets and masks, apply modulated conv."""
        # Predict offsets and masks
        offset = self.offset_conv(x)  # [B, G*2*K*K, H, W]
        mask = torch.sigmoid(self.mask_conv(x))  # [B, G*K*K, H, W]
        
        # For simplicity, we apply the modulation as attention on the input
        # then use the group conv. This captures the spirit of DCNv3
        # (adaptive receptive fields + learnable weights) without requiring
        # torchvision's DeformConv2d.
        
        # Apply mask as spatial attention
        B, C, H, W = x.shape
        G = self.num_groups
        CperG = C // G
        
        # Reshape mask to channel attention
        mask_expanded = mask.view(B, G, self.kernel_size * self.kernel_size, H, W)
        mask_mean = mask_expanded.mean(dim=2)  # [B, G, H, W]
        mask_mean = mask_mean.unsqueeze(2).expand(-1, -1, CperG, -1, -1)
        mask_mean = mask_mean.reshape(B, C, H, W)
        
        x_modulated = x * mask_mean
        
        out = self.bn(self.conv(x_modulated))
        return F.relu(out + x)  # residual connection


class MDFEN(nn.Module):
    """
    Multi-path Deformable Foreground Enhancement Network (MDFEN).
    
    From RadarNeXt paper:
    - Combines DCNv3 with PAN (Path Aggregation Network)
    - DCNv3 on the largest-scale concatenated feature map (position 4)
    - FPN top-down + PAN bottom-up fusion
    """
    def __init__(self, channels):
        super().__init__()
        C = channels  # e.g., 64
        
        # FPN: top-down path
        self.up_8to4 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2, bias=False)
        self.up_4to2 = nn.ConvTranspose2d(2*C, C, 2, stride=2, bias=False)
        
        # DCNv3 on concatenated feature at largest scale (position 4 in paper)
        # After concat of upsampled 4C + 2C features at scale H/2
        self.dcnv3 = DCNv3Simple(2*C, num_groups=4)
        
        # Rep-DWC for channel fusion after DCNv3
        self.rep_dwc_fuse = RepDWCBlock(C, stride=1)
        
        # PAN: bottom-up path
        self.down_2to4 = nn.Conv2d(C, 2*C, 3, stride=2, padding=1, bias=False)
        self.down_4to8 = nn.Conv2d(2*C, 4*C, 3, stride=2, padding=1, bias=False)
        self.bn_down1 = nn.BatchNorm2d(2*C)
        self.bn_down2 = nn.BatchNorm2d(4*C)
        
        # Final fusion: concat all scales -> output
        # After PAN, we have features at 3 scales: H/2(C), H/4(2C), H/8(4C)
        # Upsample all to H/4 and concat
        self.final_up = nn.ConvTranspose2d(C, 2*C, 2, stride=2, bias=False)
        self.final_down = nn.Conv2d(4*C, 2*C, 3, stride=2, padding=1, bias=False)
        # Concat: 2C + 2C + 2C = 6C -> output 2C
        self.final_fuse = nn.Sequential(
            nn.Conv2d(6*C, 2*C, 1, bias=False),
            nn.BatchNorm2d(2*C),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, f_half, f_quarter, f_eighth):
        """
        Args:
            f_half: [B, C, H/2, W/2]      - scale 1
            f_quarter: [B, 2C, H/4, W/4]  - scale 2
            f_eighth: [B, 4C, H/8, W/8]   - scale 3
        Returns:
            fused: [B, 2C, H/4, W/4]
        """
        # FPN top-down
        up1 = self.up_8to4(f_eighth)  # [B, 2C, H/4, W/4]
        up1 = F.interpolate(up1, size=f_quarter.shape[2:], mode='bilinear', align_corners=False)
        td_quarter = up1 + f_quarter
        
        up2 = self.up_4to2(td_quarter)  # [B, C, H/2, W/2]
        up2 = F.interpolate(up2, size=f_half.shape[2:], mode='bilinear', align_corners=False)
        
        # Concat at largest scale and apply DCNv3 (position 4)
        cat_feat = torch.cat([up2, f_half], dim=1)  # [B, 2C, H/2, W/2]
        enhanced = self.dcnv3(cat_feat)  # [B, 2C, H/2, W/2]
        
        # Reduce to C channels
        # Split and take first C channels
        enhanced_c = enhanced[:, :enhanced.shape[1]//2, :, :]  # [B, C, H/2, W/2]
        enhanced_c = self.rep_dwc_fuse(enhanced_c)
        
        # PAN bottom-up
        pan_quarter = F.relu(self.bn_down1(self.down_2to4(enhanced_c))) + td_quarter  # [B, 2C, H/4, W/4]
        pan_eighth = F.relu(self.bn_down2(self.down_4to8(pan_quarter))) + f_eighth[:, :f_eighth.shape[1]//2*2, :, :]
        # handle channel mismatch
        if pan_eighth.shape[1] != f_eighth.shape[1]:
            pan_eighth = F.interpolate(pan_eighth.unsqueeze(0), size=(f_eighth.shape[1], pan_eighth.shape[2], pan_eighth.shape[3])).squeeze(0)
        
        # Final multi-scale fusion at H/4
        # Upsample enhanced_c from H/2 to H/4... wait, it's already larger
        # Downsample H/2 -> H/4
        f1_at_quarter = F.avg_pool2d(enhanced_c, 2)  # [B, C, H/4, W/4]
        # Pad channels to 2C
        f1_at_quarter = torch.cat([f1_at_quarter, f1_at_quarter], dim=1)  # [B, 2C, H/4, W/4]
        
        # Upsample H/8 -> H/4
        f3_at_quarter = F.interpolate(pan_eighth, size=pan_quarter.shape[2:], mode='bilinear', align_corners=False)
        # Adjust channels to 2C
        if f3_at_quarter.shape[1] != pan_quarter.shape[1]:
            f3_at_quarter = f3_at_quarter[:, :pan_quarter.shape[1], :, :]
        
        fused = torch.cat([f1_at_quarter, pan_quarter, f3_at_quarter], dim=1)  # [B, 6C, H/4, W/4]
        fused = self.final_fuse(fused)  # [B, 2C, H/4, W/4]
        
        return fused


class RadarNeXtEncoder(nn.Module):
    """
    RadarNeXt-style encoder for 4D mmWave radar point clouds.
    
    Key ideas from the paper:
    1. Pillarization to create BEV pseudo-image
    2. Rep-DWC backbone (re-parameterizable depthwise convolutions)
       - 3 stages extracting multi-scale features
    3. MDFEN neck for foreground enhancement using DCNv3
    
    Radar input features: x, y, z, doppler, power, recoveredSpeed (6 channels)
    """
    def __init__(self,
                 output_dim: int = 256,
                 pc_range: list = None,
                 pillar_size: list = None):
        super().__init__()
        
        # Default range for radar (longer range than LiDAR)
        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        if pillar_size is None:
            pillar_size = [0.32, 0.32, 8.0]  # Larger pillars for sparser radar
        
        self.pc_range = pc_range
        self.pillar_size = pillar_size
        
        # Radar: x, y, z, doppler, power, recoveredSpeed = 6 channels
        C = 64
        self.pillar_encoder = PillarEncoder(
            in_channels=6,
            feat_channels=C,
            pc_range=pc_range,
            pillar_size=pillar_size,
            max_points_per_pillar=32,
            max_pillars=16000,
        )
        
        # Rep-DWC backbone: 3 stages
        # Stage 1: 3 blocks, stride 2 for first
        self.stage1 = nn.Sequential(
            RepDWCDownBlock(C, C),        # H/2, W/2, C
            RepDWCBlock(C, stride=1),
            RepDWCBlock(C, stride=1),
        )
        # Stage 2: 5 blocks, stride 2 for first
        self.stage2 = nn.Sequential(
            RepDWCDownBlock(C, 2*C),      # H/4, W/4, 2C
            RepDWCBlock(2*C, stride=1),
            RepDWCBlock(2*C, stride=1),
            RepDWCBlock(2*C, stride=1),
            RepDWCBlock(2*C, stride=1),
        )
        # Stage 3: 5 blocks, stride 2 for first
        self.stage3 = nn.Sequential(
            RepDWCDownBlock(2*C, 4*C),    # H/8, W/8, 4C
            RepDWCBlock(4*C, stride=1),
            RepDWCBlock(4*C, stride=1),
            RepDWCBlock(4*C, stride=1),
            RepDWCBlock(4*C, stride=1),
        )
        
        # MDFEN neck
        self.mdfen = MDFEN(C)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(2*C, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = output_dim
    
    def forward(self, points_batch: list) -> torch.Tensor:
        """
        Args:
            points_batch: list of [N_i, 6] tensors (x, y, z, doppler, power, speed)
        Returns:
            features: [B, output_dim] global feature vector
        """
        bev = self.pillar_encoder(points_batch)  # [B, C, H, W]
        
        f1 = self.stage1(bev)   # [B, C, H/2, W/2]
        f2 = self.stage2(f1)    # [B, 2C, H/4, W/4]
        f3 = self.stage3(f2)    # [B, 4C, H/8, W/8]
        
        # MDFEN foreground enhancement
        fused = self.mdfen(f1, f2, f3)  # [B, 2C, H/4, W/4]
        
        out = self.out_proj(fused)      # [B, output_dim, H/4, W/4]
        out = self.pool(out).squeeze(-1).squeeze(-1)  # [B, output_dim]
        
        return out


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing VoxelNeXt Encoder (LiDAR)...")
    lidar_enc = VoxelNeXtEncoder(output_dim=256).to(device)
    lidar_points = [torch.randn(10000, 4).to(device) * 20 for _ in range(2)]
    lidar_feat = lidar_enc(lidar_points)
    print(f"  LiDAR output: {lidar_feat.shape}")  # [2, 256]
    print(f"  LiDAR params: {sum(p.numel() for p in lidar_enc.parameters()):,}")
    
    print("\nTesting RadarNeXt Encoder (Radar)...")
    radar_enc = RadarNeXtEncoder(output_dim=256).to(device)
    radar_points = [torch.randn(5000, 6).to(device) * 20 for _ in range(2)]
    radar_feat = radar_enc(radar_points)
    print(f"  Radar output: {radar_feat.shape}")  # [2, 256]
    print(f"  Radar params: {sum(p.numel() for p in radar_enc.parameters()):,}")