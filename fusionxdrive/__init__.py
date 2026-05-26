"""
FusionXDrive: Multi-Modal VLM for Driving Scene Understanding and Trajectory Planning.

Architecture:
  Image Encoder:  DINOv3-base (frozen)      -> patch features [B, 256, 768]
  LiDAR Encoder:  VoxelNeXt-style           -> global feature [B, 256]
  Radar Encoder:  RadarNeXt-style           -> global feature [B, 256]
  Bridge:         Q-Former / MoRo-Former    -> query tokens  [B, 64, 896]
  LLM:            Qwen2.5-0.5B + LoRA      -> text generation
  Planner:        Truncated Diffusion       -> trajectory waypoints [B, 8, 3]
"""

__version__ = "1.0.0"

from fusionxdrive.point_cloud_encoders import VoxelNeXtEncoder, RadarNeXtEncoder
from fusionxdrive.qformer import QFormerBridge
from fusionxdrive.moro_former import MoRoFormer
from fusionxdrive.diffusion_planner import TruncatedDiffusionPlanner
from fusionxdrive.dataset import (
    UniscpDataset,
    UniscpDataCollator,
    read_pcd,
    get_rural_calibration,
    load_timestamps,
    find_nearest_idx,
)
from fusionxdrive.model_vqa import MultiModalVLM, MultiModalVLMConfig
