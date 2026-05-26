# """
# Dataset for UNISCP multi-modal driving scene understanding.

# Loads synchronized:
#   - Left camera image (720x540 PNG)
#   - LiDAR point cloud (.pcd, fields: x,y,z,intensity,t,reflectivity,ring,ambient,range)
#   - 4D Radar point cloud (.pcd, fields: x,y,z,alpha,beta,range,doppler,power,recoveredSpeed,...)
#   - Caption JSON (scene understanding labels)

# Crops point clouds to camera FOV using calibration matrices.
# """

# import os
# import json
# import struct
# import numpy as np

# # Pure-Python LZF decompressor (no external dependency)
# def lzf_decompress(data: bytes, expected_length: int) -> bytes:
#     """Decompress LZF-compressed data (pure Python implementation)."""
#     out = bytearray()
#     i = 0
#     data_len = len(data)
    
#     while i < data_len:
#         ctrl = data[i]
#         i += 1
        
#         if ctrl < 32:  # literal run
#             run_len = ctrl + 1
#             out.extend(data[i:i + run_len])
#             i += run_len
#         else:  # back reference
#             length = (ctrl >> 5) + 2
#             if length == 9:  # extended length
#                 length += data[i]
#                 i += 1
#             offset = ((ctrl & 0x1f) << 8) + data[i] + 1
#             i += 1
            
#             # Copy from back reference
#             start = len(out) - offset
#             for j in range(length):
#                 out.append(out[start + j])
    
#     if len(out) != expected_length:
#         # Truncate or pad to expected length
#         if len(out) > expected_length:
#             out = out[:expected_length]
#         else:
#             out.extend(b'\x00' * (expected_length - len(out)))
    
#     return bytes(out)
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
# import logging

# logger = logging.getLogger(__name__)


# # =============================================================================
# # PCD file reader
# # =============================================================================

# def read_pcd(filepath: str, fields_to_load: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
#     """
#     Read a PCD file (supports binary and binary_compressed formats).
    
#     Args:
#         filepath: path to .pcd file
#         fields_to_load: optional list of field names to load (None = all)
    
#     Returns:
#         dict mapping field names to numpy arrays
#     """
#     with open(filepath, 'rb') as f:
#         # Parse header
#         header = {}
#         while True:
#             line = f.readline().decode('ascii', errors='ignore').strip()
#             if line.startswith('DATA'):
#                 header['DATA'] = line.split()[-1]
#                 break
#             if line.startswith('#') or not line:
#                 continue
#             parts = line.split()
#             if len(parts) >= 2:
#                 key = parts[0]
#                 values = parts[1:]
#                 header[key] = values
        
#         fields = header.get('FIELDS', [])
#         sizes = [int(s) for s in header.get('SIZE', [])]
#         types = header.get('TYPE', [])
#         counts = [int(c) for c in header.get('COUNT', [])]
#         width = int(header.get('WIDTH', ['0'])[0])
#         height = int(header.get('HEIGHT', ['1'])[0])
#         num_points = int(header.get('POINTS', [str(width * height)])[0])
#         data_format = header.get('DATA', 'ascii')
        
#         # Build dtype for each field
#         type_map = {
#             ('F', 4): np.float32,
#             ('F', 8): np.float64,
#             ('U', 1): np.uint8,
#             ('U', 2): np.uint16,
#             ('U', 4): np.uint32,
#             ('I', 1): np.int8,
#             ('I', 2): np.int16,
#             ('I', 4): np.int32,
#         }
        
#         dtypes = []
#         for i, field in enumerate(fields):
#             dt = type_map.get((types[i], sizes[i]), np.float32)
#             dtypes.append((field, dt))
        
#         point_size = sum(sizes)
        
#         if data_format == 'binary':
#             raw = f.read(num_points * point_size)
#             data = np.frombuffer(raw, dtype=np.dtype(dtypes))
            
#         elif data_format == 'binary_compressed':
#             # Read compressed and decompressed sizes
#             compressed_size = struct.unpack('<I', f.read(4))[0]
#             decompressed_size = struct.unpack('<I', f.read(4))[0]
            
#             compressed_data = f.read(compressed_size)
            
#             try:
#                 decompressed = lzf_decompress(compressed_data, decompressed_size)
#             except Exception:
#                 # Fallback: return empty
#                 logger.warning(f"Failed to decompress PCD: {filepath}")
#                 result = {}
#                 for field, dt in dtypes:
#                     result[field] = np.zeros(0, dtype=dt)
#                 return result
            
#             # binary_compressed stores data column-by-column
#             data_dict = {}
#             offset = 0
#             for i, (field, dt) in enumerate(dtypes):
#                 field_bytes = sizes[i] * num_points
#                 field_data = np.frombuffer(
#                     decompressed[offset:offset + field_bytes], dtype=dt
#                 )
#                 data_dict[field] = field_data
#                 offset += field_bytes
            
#             if fields_to_load:
#                 return {k: data_dict[k] for k in fields_to_load if k in data_dict}
#             return data_dict
            
#         elif data_format == 'ascii':
#             lines_data = []
#             for _ in range(num_points):
#                 line = f.readline().decode('ascii', errors='ignore').strip()
#                 if line:
#                     lines_data.append([float(x) for x in line.split()])
#             arr = np.array(lines_data, dtype=np.float32)
#             data_dict = {}
#             for i, (field, dt) in enumerate(dtypes):
#                 if i < arr.shape[1]:
#                     data_dict[field] = arr[:, i].astype(dt)
#             if fields_to_load:
#                 return {k: data_dict[k] for k in fields_to_load if k in data_dict}
#             return data_dict
#         else:
#             raise ValueError(f"Unknown PCD data format: {data_format}")
        
#         # For binary (non-compressed) format
#         data_dict = {}
#         for field, dt in dtypes:
#             data_dict[field] = data[field].copy()
        
#         if fields_to_load:
#             return {k: data_dict[k] for k in fields_to_load if k in data_dict}
#         return data_dict


# # =============================================================================
# # Calibration utilities
# # =============================================================================

# class Calibration:
#     """
#     Camera-LiDAR-Radar calibration.
    
#     Stores:
#       - Camera intrinsic matrix K (3x3)
#       - LiDAR-to-camera extrinsic: R_cl (3x3), t_cl (3,)
#       - Radar-to-camera extrinsic: R_cr (3x3), t_cr (3,)
#       - Distortion coefficients D (optional)
#     """
#     def __init__(self,
#                  K: np.ndarray,
#                  R_cl: np.ndarray, t_cl: np.ndarray,
#                  R_cr: np.ndarray, t_cr: np.ndarray,
#                  D: Optional[np.ndarray] = None,
#                  image_size: Tuple[int, int] = (720, 540)):
#         self.K = K        # [3, 3]
#         self.R_cl = R_cl  # [3, 3] rotation: LiDAR -> camera
#         self.t_cl = t_cl  # [3]   translation: LiDAR -> camera
#         self.R_cr = R_cr  # [3, 3] rotation: Radar -> camera
#         self.t_cr = t_cr  # [3]   translation: Radar -> camera
#         self.D = D
#         self.img_w, self.img_h = image_size
    
#     def project_lidar_to_image(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Project LiDAR points to image plane. Returns (uv, depth)."""
#         return self._project(points_xyz, self.R_cl, self.t_cl)
    
#     def project_radar_to_image(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Project Radar points to image plane. Returns (uv, depth)."""
#         return self._project(points_xyz, self.R_cr, self.t_cr)
    
#     def _project(self, points_xyz, R, t):
#         """Project 3D points to image using extrinsic R,t and intrinsic K."""
#         # Transform to camera coordinate: P_cam = R @ P_sensor + t
#         P_cam = (R @ points_xyz.T).T + t  # [N, 3]
        
#         # Filter: keep only points in front of camera (z > 0)
#         depth = P_cam[:, 2]
        
#         # Project to image: p = K @ P_cam
#         uv_h = (self.K @ P_cam.T).T  # [N, 3]
#         uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-8)  # [N, 2]
        
#         return uv, depth
    
#     def crop_lidar_to_fov(self, points: np.ndarray, margin: int = 0) -> np.ndarray:
#         """
#         Crop LiDAR point cloud to camera FOV.
#         Args:
#             points: [N, C] where first 3 cols are x, y, z
#             margin: pixel margin for FOV
#         Returns:
#             cropped points [M, C]
#         """
#         return self._crop_to_fov(points, self.R_cl, self.t_cl, margin)
    
#     def crop_radar_to_fov(self, points: np.ndarray, margin: int = 0) -> np.ndarray:
#         """Crop Radar point cloud to camera FOV."""
#         return self._crop_to_fov(points, self.R_cr, self.t_cr, margin)
    
#     def _crop_to_fov(self, points, R, t, margin):
#         if len(points) == 0:
#             return points
#         xyz = points[:, :3]
#         uv, depth = self._project(xyz, R, t)
        
#         mask = (
#             (depth > 0.1) &
#             (uv[:, 0] >= -margin) & (uv[:, 0] < self.img_w + margin) &
#             (uv[:, 1] >= -margin) & (uv[:, 1] < self.img_h + margin)
#         )
#         return points[mask]


# def get_rural_calibration() -> Calibration:
#     """
#     Returns calibration for RURAL sequences based on the provided calibration data.
#     """
#     # Camera LEFT intrinsics
#     K = np.array([
#         [647.429551329957, 0, 374.754981608475],
#         [0, 647.909150378696, 266.214772068844],
#         [0, 0, 1],
#     ], dtype=np.float64)
    
#     D = np.array([-0.220566189060877, 0.102696172815567, 0.0, 0.0], dtype=np.float64)
    
#     # LiDAR -> Camera rotation and translation
#     R_cl = np.array([
#         [0.00830729606316660, -0.999964176331806, 0.00162323293799360],
#         [-0.00514736101274921, -0.00166602951617612, -0.999985364403027],
#         [0.999952245613121, 0.00829881911498959, -0.00516101681583692],
#     ], dtype=np.float64)
    
#     t_cl = np.array([0.2347512330691049, -0.3165704095566238, -0.7598884687506345],
#                      dtype=np.float64)
    
#     # Radar -> Camera rotation and translation
#     R_cr = np.array([
#         [-0.999599255745628, -0.0244619965156001, -0.0142456533461353],
#         [-0.0253476700264381, 0.997515830991367, 0.0657241397496601],
#         [0.0126025210580381, 0.0660588952986107, -0.997736136869317],
#     ], dtype=np.float64)
    
#     t_cr = np.array([-0.2159934537928913, -1.376102767377935, -1.693106180476492],
#                      dtype=np.float64)
    
#     return Calibration(K, R_cl, t_cl, R_cr, t_cr, D, image_size=(720, 540))


# # =============================================================================
# # Timestamp matching
# # =============================================================================

# def load_timestamps(filepath: str) -> List[Tuple[str, float]]:
#     """
#     Load timestamp file.
#     Format: <index> <seconds> <nanoseconds>
#     e.g.: 000000 1677224778 947625704
    
#     Returns: list of (index_str, timestamp_float)
#     """
#     timestamps = []
#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 3:
#                 idx_str = parts[0]
#                 ts = float(parts[1]) + float(parts[2]) / 1e9
#                 timestamps.append((idx_str, ts))
#     return timestamps


# def find_nearest_idx(query_ts: float, target_timestamps: List[Tuple[str, float]], 
#                      max_diff: float = 0.1) -> Optional[str]:
#     """Find the nearest timestamp index within max_diff seconds."""
#     best_idx = None
#     best_diff = float('inf')
#     for idx_str, ts in target_timestamps:
#         diff = abs(ts - query_ts)
#         if diff < best_diff:
#             best_diff = diff
#             best_idx = idx_str
#     if best_diff <= max_diff:
#         return best_idx
#     return None


# # =============================================================================
# # Prompts
# # =============================================================================

# SYSTEM_PROMPT = """You are an urban driving scene understanding module.
# Your task is to analyze the input image and output structured labels for urban driving visual question answering (VQA).
# IMPORTANT RULES:
# - Only choose answers from the provided options for all closed-set fields.
# - Do not explain your reasoning except for the final explanation field.
# - The final explanation must be one short sentence only.
# - If the scene is ambiguous or uncertain, choose the closest valid option.
# - Output only one valid JSON object.
# - Do not output any extra text before or after the JSON."""

# USER_PROMPT = """Analyze the urban driving scene and answer the following questions.
# [1. Weather and Illumination]
# Choose ONE weather label and ONE illumination label.
# Weather options: {sunny, cloudy, rainy, snowy, foggy}
# Illumination options: {daytime, dusk, night}
# [2. Traffic Light]
# Determine whether there is a traffic light relevant to the ego vehicle.
# Presence options: {yes, no}
# State options: {red, yellow, green, unknown, none}
# [3. Traffic Sign]
# Determine whether there is a traffic sign relevant to driving.
# Presence options: {yes, no}
# Sign options: {stop, speed_limit, no_entry, turn_left, turn_right, pedestrian_crossing, parking, warning, unknown, none}
# [4. Key Traffic Participants]
# Identify the main traffic participants that may affect the ego vehicle.
# Count options: {0, 1, 2, 3_plus}
# List up to 3 key participants, ordered by their potential influence on the ego vehicle.
# For each participant:
# Category options: {vehicle, pedestrian, bicycle}
# Direction options: {front, front_left, front_right, left, right, rear, unknown}
# Intent options: {stationary, moving_left, moving_right, moving_forward, moving_toward_ego, moving_away, unknown}
# [5. Hazard Region]
# Determine whether there is a hazardous region that needs special attention.
# Presence options: {yes, no}
# Hazard options: {pedestrian_cluster, accident_area, road_construction, parked_vehicle_risk, intersection_conflict, lane_obstruction, unknown, none}
# Direction options: {front, front_left, front_right, left, right, near, far, unknown, none}
# [6. Forward Drivability]
# Evaluate the forward drivability condition.
# Status options: {clear, slow_down_needed, blocked}
# [7. Lane Keeping]
# Evaluate whether current lane keeping is reasonable.
# Reasonableness options: {reasonable, slightly_unreasonable, unreasonable, unknown}
# Deviation options: {centered, slight_left, slight_right, severe_left, severe_right, unknown}
# [8. Driving Advice]
# Choose the main driving advice.
# Action options: {accelerate, decelerate, turn_left, turn_right, keep, brake}
# [9. Explanation]
# Explain the main reason for the driving advice in one short sentence.
# Output in the following JSON format:
# {
#   "weather": {"condition": "", "illumination": ""},
#   "traffic_light": {"present": "", "state": ""},
#   "traffic_sign": {"present": "", "category": ""},
#   "participants": {"count": "", "objects": [{"category": "", "direction": "", "intent": ""}]},
#   "hazard_region": {"present": "", "type": "", "direction": ""},
#   "forward_drivability": {"status": ""},
#   "lane_keeping": {"status": "", "deviation": ""},
#   "driving_advice": {"action": ""},
#   "explanation": {"reason": ""}
# }"""


# # =============================================================================
# # Dataset
# # =============================================================================

# class UniscpDataset(Dataset):
#     """
#     Multi-modal dataset for UNISCP driving scene understanding.
    
#     For each sample:
#       1. Load left camera image
#       2. Load LiDAR PCD, crop to camera FOV, extract (x,y,z,intensity)
#       3. Load Radar PCD, crop to camera FOV, extract (x,y,z,doppler,power,recoveredSpeed)
#       4. Load caption JSON as target
#     """
    
#     # Sequence directory structure patterns
#     RURAL_SEQUENCES = [
#         'RURAL_A0', 'RURAL_A1', 'RURAL_A2',
#         'RURAL_B0', 'RURAL_B1', 'RURAL_B2',
#     ]
#     OTHER_SEQUENCES = [
#         'FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO',
#         'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE',
#     ]
    
#     def __init__(self,
#                  data_root: str,
#                  sequences: Optional[List[str]] = None,
#                  tokenizer=None,
#                  processor=None,
#                  image_pad_num: int = 64,
#                  max_lidar_points: int = 40000,
#                  max_radar_points: int = 16000,
#                  lidar_pc_range: list = None,
#                  radar_pc_range: list = None,
#                  ):
#         super().__init__()
#         self.data_root = Path(data_root)
#         self.tokenizer = tokenizer
#         self.processor = processor
#         self.image_pad_num = image_pad_num
#         self.max_lidar_points = max_lidar_points
#         self.max_radar_points = max_radar_points
        
#         self.lidar_pc_range = lidar_pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
#         self.radar_pc_range = radar_pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
#         # Calibration (using RURAL calibration for all sequences for now)
#         self.calib = get_rural_calibration()
        
#         # Build sample index
#         if sequences is None:
#             sequences = self.RURAL_SEQUENCES + self.OTHER_SEQUENCES
        
#         self.samples = []
#         for seq_name in sequences:
#             seq_dir = self.data_root / seq_name
#             if not seq_dir.exists():
#                 logger.warning(f"Sequence {seq_name} not found, skipping")
#                 continue
#             self._index_sequence(seq_name, seq_dir)
        
#         logger.info(f"Loaded {len(self.samples)} samples from {len(sequences)} sequences")
#         print(f"Loaded {len(self.samples)} samples from sequences")
    
#     def _find_subdir(self, seq_dir: Path, prefix: str) -> Optional[Path]:
#         """Find the actual data subdirectory (handles nested dirs like 1_IMAGE/1_IMAGE/)."""
#         top = seq_dir / prefix
#         if not top.exists():
#             return None
#         # Check for nested same-name subdir
#         nested = top / prefix
#         if nested.exists():
#             return nested
#         return top
    
#     def _index_sequence(self, seq_name: str, seq_dir: Path):
#         """Build index of valid samples for one sequence."""
#         # Find directories
#         img_dir = self._find_subdir(seq_dir, '1_IMAGE')
#         lidar_dir = self._find_subdir(seq_dir, '2_LIDAR')
#         radar_dir = self._find_subdir(seq_dir, '3_RADAR')
#         caption_dir = seq_dir / '6_CAPTION'
        
#         if img_dir is None or lidar_dir is None or radar_dir is None:
#             logger.warning(f"Missing sensor dirs in {seq_name}")
#             return
#         if not caption_dir.exists():
#             logger.warning(f"Missing caption dir in {seq_name}")
#             return
        
#         # Load timestamps
#         ts_img_file = img_dir / 'timestamp_image_left.txt'
#         ts_lidar_file = lidar_dir / 'timestamp_lidar.txt'
#         ts_radar_file = radar_dir / 'timestamp_radar.txt'
        
#         if not all(f.exists() for f in [ts_img_file, ts_lidar_file, ts_radar_file]):
#             logger.warning(f"Missing timestamp files in {seq_name}")
#             return
        
#         ts_img = load_timestamps(str(ts_img_file))
#         ts_lidar = load_timestamps(str(ts_lidar_file))
#         ts_radar = load_timestamps(str(ts_radar_file))
        
#         # Find LEFT image dir
#         left_dir = img_dir / 'LEFT'
#         if not left_dir.exists():
#             logger.warning(f"Missing LEFT image dir in {seq_name}")
#             return
        
#         # LiDAR/Radar PCD dirs
#         lidar_pcd_dir = lidar_dir / 'PCD'
#         radar_pcd_dir = radar_dir / 'PCD'
        
#         if not lidar_pcd_dir.exists() or not radar_pcd_dir.exists():
#             logger.warning(f"Missing PCD dirs in {seq_name}")
#             return
        
#         # For each image frame, find matching lidar/radar and caption
#         for img_idx_str, img_ts in ts_img:
#             # Check image exists
#             img_path = left_dir / f"{img_idx_str}.png"
#             if not img_path.exists():
#                 continue
            
#             # Check caption exists (captions are aligned with images)
#             caption_path = caption_dir / f"{img_idx_str}.json"
#             if not caption_path.exists():
#                 continue
            
#             # Find nearest LiDAR and Radar
#             lidar_idx = find_nearest_idx(img_ts, ts_lidar, max_diff=0.15)
#             radar_idx = find_nearest_idx(img_ts, ts_radar, max_diff=0.15)
            
#             if lidar_idx is None or radar_idx is None:
#                 continue
            
#             lidar_path = lidar_pcd_dir / f"{lidar_idx}.pcd"
#             radar_path = radar_pcd_dir / f"{radar_idx}.pcd"
            
#             if not lidar_path.exists() or not radar_path.exists():
#                 continue
            
#             self.samples.append({
#                 'seq': seq_name,
#                 'img_idx': img_idx_str,
#                 'img_path': str(img_path),
#                 'lidar_path': str(lidar_path),
#                 'radar_path': str(radar_path),
#                 'caption_path': str(caption_path),
#             })
    
#     def __len__(self):
#         return len(self.samples)
    
#     def _load_lidar(self, path: str) -> np.ndarray:
#         """Load LiDAR PCD and extract (x, y, z, intensity)."""
#         try:
#             data = read_pcd(path, fields_to_load=['x', 'y', 'z', 'intensity'])
#             x = data.get('x', np.zeros(0, dtype=np.float32))
#             y = data.get('y', np.zeros(0, dtype=np.float32))
#             z = data.get('z', np.zeros(0, dtype=np.float32))
#             intensity = data.get('intensity', np.zeros_like(x))
            
#             if len(x) == 0:
#                 return np.zeros((0, 4), dtype=np.float32)
            
#             points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
            
#             # Remove NaN/Inf
#             valid = np.all(np.isfinite(points), axis=-1)
#             points = points[valid]
            
#             # Remove zero points (invalid returns)
#             nonzero = np.any(points[:, :3] != 0, axis=-1)
#             points = points[nonzero]
            
#             return points
#         except Exception as e:
#             logger.warning(f"Failed to load LiDAR {path}: {e}")
#             return np.zeros((0, 4), dtype=np.float32)
    
#     def _load_radar(self, path: str) -> np.ndarray:
#         """Load Radar PCD and extract (x, y, z, doppler, power, recoveredSpeed)."""
#         try:
#             data = read_pcd(path, fields_to_load=[
#                 'x', 'y', 'z', 'doppler', 'power', 'recoveredSpeed'
#             ])
#             x = data.get('x', np.zeros(0, dtype=np.float32))
#             y = data.get('y', np.zeros(0, dtype=np.float32))
#             z = data.get('z', np.zeros(0, dtype=np.float32))
#             doppler = data.get('doppler', np.zeros_like(x))
#             power = data.get('power', np.zeros_like(x))
#             speed = data.get('recoveredSpeed', np.zeros_like(x))
            
#             if len(x) == 0:
#                 return np.zeros((0, 6), dtype=np.float32)
            
#             points = np.stack([x, y, z, doppler, power, speed], axis=-1).astype(np.float32)
            
#             # Remove NaN/Inf
#             valid = np.all(np.isfinite(points), axis=-1)
#             points = points[valid]
            
#             return points
#         except Exception as e:
#             logger.warning(f"Failed to load Radar {path}: {e}")
#             return np.zeros((0, 6), dtype=np.float32)
    
#     def _subsample(self, points: np.ndarray, max_points: int) -> np.ndarray:
#         """Random subsample if too many points."""
#         if len(points) > max_points:
#             idx = np.random.choice(len(points), max_points, replace=False)
#             return points[idx]
#         return points
    
#     def _pad_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
#         """Pad with zeros to fixed size for batching."""
#         n, c = points.shape
#         if n >= max_points:
#             return points[:max_points]
#         padded = np.zeros((max_points, c), dtype=np.float32)
#         padded[:n] = points
#         return padded
    
#     def __getitem__(self, index):
#         sample = self.samples[index]
        
#         # 1. Load image
#         try:
#             image = Image.open(sample['img_path']).convert('RGB')
#         except Exception as e:
#             logger.warning(f"Failed to load image {sample['img_path']}: {e}")
#             image = Image.new('RGB', (720, 540), color='black')
        
#         # 2. Load and crop LiDAR
#         lidar_points = self._load_lidar(sample['lidar_path'])
#         if len(lidar_points) > 0:
#             lidar_points = self.calib.crop_lidar_to_fov(lidar_points)
#         lidar_points = self._subsample(lidar_points, self.max_lidar_points)
        
#         # 3. Load and crop Radar
#         radar_points = self._load_radar(sample['radar_path'])
#         if len(radar_points) > 0:
#             radar_points = self.calib.crop_radar_to_fov(radar_points)
#         radar_points = self._subsample(radar_points, self.max_radar_points)
        
#         # 4. Load caption
#         try:
#             with open(sample['caption_path'], 'r') as f:
#                 caption = json.load(f)
#         except Exception as e:
#             logger.warning(f"Failed to load caption {sample['caption_path']}: {e}")
#             caption = self._get_error_template()
        
#         # 5. Process image for DINOv3
#         pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
#         # 6. Build text input/output
#         answer_text = json.dumps(caption, ensure_ascii=False)
        
#         # Build conversation
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": "<image>\n" + USER_PROMPT},
#         ]
        
#         q_text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         ).replace('<image>', '<|image_pad|>' * self.image_pad_num)
        
#         q_input_ids = self.tokenizer(q_text)["input_ids"]
#         a_text = answer_text + self.tokenizer.eos_token
#         a_input_ids = self.tokenizer(a_text)["input_ids"]
        
#         input_ids = q_input_ids + a_input_ids
#         labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
#         input_ids = input_ids[:-1]
#         labels = labels[1:]
        
#         # Record actual point counts for masking in collate
#         n_lidar = len(lidar_points)
#         n_radar = len(radar_points)
        
#         return {
#             "input_ids": input_ids,
#             "labels": labels,
#             "pixel_values": pixel_values,
#             "lidar_points": lidar_points.astype(np.float32),
#             "radar_points": radar_points.astype(np.float32),
#             "n_lidar": n_lidar,
#             "n_radar": n_radar,
#         }
    
#     @staticmethod
#     def _get_error_template():
#         return {
#             "weather": {"condition": "unknown", "illumination": "unknown"},
#             "traffic_light": {"present": "unknown", "state": "unknown"},
#             "traffic_sign": {"present": "unknown", "category": "unknown"},
#             "participants": {"count": "0", "objects": []},
#             "hazard_region": {"present": "unknown", "type": "unknown", "direction": "unknown"},
#             "forward_drivability": {"status": "unknown"},
#             "lane_keeping": {"status": "unknown", "deviation": "unknown"},
#             "driving_advice": {"action": "unknown"},
#             "explanation": {"reason": "unknown"},
#         }


# class UniscpDataCollator:
#     """
#     Custom data collator that handles variable-length point clouds.
    
#     Point clouds are passed as lists (not padded tensors) because
#     the encoders handle variable-length inputs natively.
#     """
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
    
#     def __call__(self, features: List[Dict]) -> Dict:
#         max_len = max(len(f["input_ids"]) for f in features)
        
#         input_ids = []
#         labels = []
#         pixel_values = []
#         lidar_points_list = []
#         radar_points_list = []
        
#         for f in features:
#             # Pad text
#             pad_len = max_len - len(f["input_ids"])
#             input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
#             labels.append(f["labels"] + [self.tokenizer.pad_token_id] * pad_len)
            
#             pixel_values.append(f["pixel_values"])
            
#             # Point clouds as tensors (variable length, passed as list)
#             lidar_pts = torch.from_numpy(f["lidar_points"]).float()
#             radar_pts = torch.from_numpy(f["radar_points"]).float()
            
#             # Only keep actual points (remove padding zeros)
#             n_l = f["n_lidar"]
#             n_r = f["n_radar"]
#             lidar_points_list.append(lidar_pts[:n_l] if n_l > 0 else lidar_pts[:1] * 0)
#             radar_points_list.append(radar_pts[:n_r] if n_r > 0 else radar_pts[:1] * 0)
        
#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "labels": torch.tensor(labels, dtype=torch.long),
#             "pixel_values": torch.stack(pixel_values, dim=0),
#             "lidar_points": lidar_points_list,  # list of [N_i, 4] tensors
#             "radar_points": radar_points_list,   # list of [N_i, 6] tensors
#         }


"""
Dataset for UNISCP multi-modal driving scene understanding.

Loads synchronized:
  - Left camera image (720x540 PNG)
  - LiDAR point cloud (.pcd, fields: x,y,z,intensity,t,reflectivity,ring,ambient,range)
  - 4D Radar point cloud (.pcd, fields: x,y,z,alpha,beta,range,doppler,power,recoveredSpeed,...)
  - Caption JSON (scene understanding labels)

Crops point clouds to camera FOV using calibration matrices.
"""

import os
import json
import struct
import numpy as np

# Pure-Python LZF decompressor (no external dependency)
def lzf_decompress(data: bytes, expected_length: int) -> bytes:
    """Decompress LZF-compressed data (pure Python implementation)."""
    out = bytearray()
    i = 0
    data_len = len(data)
    
    while i < data_len:
        ctrl = data[i]
        i += 1
        
        if ctrl < 32:  # literal run
            run_len = ctrl + 1
            out.extend(data[i:i + run_len])
            i += run_len
        else:  # back reference
            length = (ctrl >> 5) + 2
            if length == 9:  # extended length
                length += data[i]
                i += 1
            offset = ((ctrl & 0x1f) << 8) + data[i] + 1
            i += 1
            
            # Copy from back reference
            start = len(out) - offset
            for j in range(length):
                out.append(out[start + j])
    
    if len(out) != expected_length:
        # Truncate or pad to expected length
        if len(out) > expected_length:
            out = out[:expected_length]
        else:
            out.extend(b'\x00' * (expected_length - len(out)))
    
    return bytes(out)
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PCD file reader
# =============================================================================

def read_pcd(filepath: str, fields_to_load: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Read a PCD file (supports binary and binary_compressed formats).
    
    Args:
        filepath: path to .pcd file
        fields_to_load: optional list of field names to load (None = all)
    
    Returns:
        dict mapping field names to numpy arrays
    """
    with open(filepath, 'rb') as f:
        # Parse header
        header = {}
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('DATA'):
                header['DATA'] = line.split()[-1]
                break
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                values = parts[1:]
                header[key] = values
        
        fields = header.get('FIELDS', [])
        sizes = [int(s) for s in header.get('SIZE', [])]
        types = header.get('TYPE', [])
        counts = [int(c) for c in header.get('COUNT', [])]
        width = int(header.get('WIDTH', ['0'])[0])
        height = int(header.get('HEIGHT', ['1'])[0])
        num_points = int(header.get('POINTS', [str(width * height)])[0])
        data_format = header.get('DATA', 'ascii')
        
        # Build dtype for each field
        type_map = {
            ('F', 4): np.float32,
            ('F', 8): np.float64,
            ('U', 1): np.uint8,
            ('U', 2): np.uint16,
            ('U', 4): np.uint32,
            ('I', 1): np.int8,
            ('I', 2): np.int16,
            ('I', 4): np.int32,
        }
        
        dtypes = []
        for i, field in enumerate(fields):
            dt = type_map.get((types[i], sizes[i]), np.float32)
            dtypes.append((field, dt))
        
        point_size = sum(sizes)
        
        if data_format == 'binary':
            raw = f.read(num_points * point_size)
            data = np.frombuffer(raw, dtype=np.dtype(dtypes))
            
        elif data_format == 'binary_compressed':
            # Read compressed and decompressed sizes
            compressed_size = struct.unpack('<I', f.read(4))[0]
            decompressed_size = struct.unpack('<I', f.read(4))[0]
            
            compressed_data = f.read(compressed_size)
            
            try:
                decompressed = lzf_decompress(compressed_data, decompressed_size)
            except Exception:
                # Fallback: return empty
                logger.warning(f"Failed to decompress PCD: {filepath}")
                result = {}
                for field, dt in dtypes:
                    result[field] = np.zeros(0, dtype=dt)
                return result
            
            # binary_compressed stores data column-by-column
            data_dict = {}
            offset = 0
            for i, (field, dt) in enumerate(dtypes):
                field_bytes = sizes[i] * num_points
                field_data = np.frombuffer(
                    decompressed[offset:offset + field_bytes], dtype=dt
                )
                data_dict[field] = field_data
                offset += field_bytes
            
            if fields_to_load:
                return {k: data_dict[k] for k in fields_to_load if k in data_dict}
            return data_dict
            
        elif data_format == 'ascii':
            lines_data = []
            for _ in range(num_points):
                line = f.readline().decode('ascii', errors='ignore').strip()
                if line:
                    lines_data.append([float(x) for x in line.split()])
            arr = np.array(lines_data, dtype=np.float32)
            data_dict = {}
            for i, (field, dt) in enumerate(dtypes):
                if i < arr.shape[1]:
                    data_dict[field] = arr[:, i].astype(dt)
            if fields_to_load:
                return {k: data_dict[k] for k in fields_to_load if k in data_dict}
            return data_dict
        else:
            raise ValueError(f"Unknown PCD data format: {data_format}")
        
        # For binary (non-compressed) format
        data_dict = {}
        for field, dt in dtypes:
            data_dict[field] = data[field].copy()
        
        if fields_to_load:
            return {k: data_dict[k] for k in fields_to_load if k in data_dict}
        return data_dict


# =============================================================================
# Calibration utilities
# =============================================================================

class Calibration:
    """
    Camera-LiDAR-Radar calibration.
    
    Stores:
      - Camera intrinsic matrix K (3x3)
      - LiDAR-to-camera extrinsic: R_cl (3x3), t_cl (3,)
      - Radar-to-camera extrinsic: R_cr (3x3), t_cr (3,)
      - Distortion coefficients D (optional)
    """
    def __init__(self,
                 K: np.ndarray,
                 R_cl: np.ndarray, t_cl: np.ndarray,
                 R_cr: np.ndarray, t_cr: np.ndarray,
                 D: Optional[np.ndarray] = None,
                 image_size: Tuple[int, int] = (720, 540)):
        self.K = K        # [3, 3]
        self.R_cl = R_cl  # [3, 3] rotation: LiDAR -> camera
        self.t_cl = t_cl  # [3]   translation: LiDAR -> camera
        self.R_cr = R_cr  # [3, 3] rotation: Radar -> camera
        self.t_cr = t_cr  # [3]   translation: Radar -> camera
        self.D = D
        self.img_w, self.img_h = image_size
    
    def project_lidar_to_image(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project LiDAR points to image plane. Returns (uv, depth)."""
        return self._project(points_xyz, self.R_cl, self.t_cl)
    
    def project_radar_to_image(self, points_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project Radar points to image plane. Returns (uv, depth)."""
        return self._project(points_xyz, self.R_cr, self.t_cr)
    
    def _project(self, points_xyz, R, t):
        """Project 3D points to image using extrinsic R,t and intrinsic K."""
        # Transform to camera coordinate: P_cam = R @ P_sensor + t
        P_cam = (R @ points_xyz.T).T + t  # [N, 3]
        
        # Filter: keep only points in front of camera (z > 0)
        depth = P_cam[:, 2]
        
        # Project to image: p = K @ P_cam
        uv_h = (self.K @ P_cam.T).T  # [N, 3]
        uv = uv_h[:, :2] / (uv_h[:, 2:3] + 1e-8)  # [N, 2]
        
        return uv, depth
    
    def crop_lidar_to_fov(self, points: np.ndarray, margin: int = 0) -> np.ndarray:
        """
        Crop LiDAR point cloud to camera FOV.
        Args:
            points: [N, C] where first 3 cols are x, y, z
            margin: pixel margin for FOV
        Returns:
            cropped points [M, C]
        """
        return self._crop_to_fov(points, self.R_cl, self.t_cl, margin)
    
    def crop_radar_to_fov(self, points: np.ndarray, margin: int = 0) -> np.ndarray:
        """Crop Radar point cloud to camera FOV."""
        return self._crop_to_fov(points, self.R_cr, self.t_cr, margin)
    
    def _crop_to_fov(self, points, R, t, margin):
        if len(points) == 0:
            return points
        xyz = points[:, :3]
        uv, depth = self._project(xyz, R, t)
        
        mask = (
            (depth > 0.1) &
            (uv[:, 0] >= -margin) & (uv[:, 0] < self.img_w + margin) &
            (uv[:, 1] >= -margin) & (uv[:, 1] < self.img_h + margin)
        )
        return points[mask]


def get_rural_calibration() -> Calibration:
    """
    Returns calibration for RURAL sequences based on the provided calibration data.
    """
    # Camera LEFT intrinsics
    K = np.array([
        [647.429551329957, 0, 374.754981608475],
        [0, 647.909150378696, 266.214772068844],
        [0, 0, 1],
    ], dtype=np.float64)
    
    D = np.array([-0.220566189060877, 0.102696172815567, 0.0, 0.0], dtype=np.float64)
    
    # LiDAR -> Camera rotation and translation
    R_cl = np.array([
        [0.00830729606316660, -0.999964176331806, 0.00162323293799360],
        [-0.00514736101274921, -0.00166602951617612, -0.999985364403027],
        [0.999952245613121, 0.00829881911498959, -0.00516101681583692],
    ], dtype=np.float64)
    
    t_cl = np.array([0.2347512330691049, -0.3165704095566238, -0.7598884687506345],
                     dtype=np.float64)
    
    # Radar -> Camera rotation and translation
    R_cr = np.array([
        [-0.999599255745628, -0.0244619965156001, -0.0142456533461353],
        [-0.0253476700264381, 0.997515830991367, 0.0657241397496601],
        [0.0126025210580381, 0.0660588952986107, -0.997736136869317],
    ], dtype=np.float64)
    
    t_cr = np.array([-0.2159934537928913, -1.376102767377935, -1.693106180476492],
                     dtype=np.float64)
    
    return Calibration(K, R_cl, t_cl, R_cr, t_cr, D, image_size=(720, 540))


# =============================================================================
# Timestamp matching
# =============================================================================

def load_timestamps(filepath: str) -> List[Tuple[str, float]]:
    """
    Load timestamp file.
    Format: <index> <seconds> <nanoseconds>
    e.g.: 000000 1677224778 947625704
    
    Returns: list of (index_str, timestamp_float)
    """
    timestamps = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                idx_str = parts[0]
                ts = float(parts[1]) + float(parts[2]) / 1e9
                timestamps.append((idx_str, ts))
    return timestamps


def find_nearest_idx(query_ts: float, target_timestamps: List[Tuple[str, float]], 
                     max_diff: float = 0.1) -> Optional[str]:
    """Find the nearest timestamp index within max_diff seconds."""
    best_idx = None
    best_diff = float('inf')
    for idx_str, ts in target_timestamps:
        diff = abs(ts - query_ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = idx_str
    if best_diff <= max_diff:
        return best_idx
    return None


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an urban driving scene understanding module.
Your task is to analyze the input image and output structured labels for urban driving visual question answering (VQA).
IMPORTANT RULES:
- Only choose answers from the provided options for all closed-set fields.
- Do not explain your reasoning except for the final explanation field.
- The final explanation must be one short sentence only.
- If the scene is ambiguous or uncertain, choose the closest valid option.
- Output only one valid JSON object.
- Do not output any extra text before or after the JSON."""

USER_PROMPT = """Analyze the urban driving scene and answer the following questions.
[1. Weather and Illumination]
Choose ONE weather label and ONE illumination label.
Weather options: {sunny, cloudy, rainy, snowy, foggy}
Illumination options: {daytime, dusk, night}
[2. Traffic Light]
Determine whether there is a traffic light relevant to the ego vehicle.
Presence options: {yes, no}
State options: {red, yellow, green, unknown, none}
[3. Traffic Sign]
Determine whether there is a traffic sign relevant to driving.
Presence options: {yes, no}
Sign options: {stop, speed_limit, no_entry, turn_left, turn_right, pedestrian_crossing, parking, warning, unknown, none}
[4. Key Traffic Participants]
Identify the main traffic participants that may affect the ego vehicle.
Count options: {0, 1, 2, 3_plus}
List up to 3 key participants, ordered by their potential influence on the ego vehicle.
For each participant:
Category options: {vehicle, pedestrian, bicycle}
Direction options: {front, front_left, front_right, left, right, rear, unknown}
Intent options: {stationary, moving_left, moving_right, moving_forward, moving_toward_ego, moving_away, unknown}
[5. Hazard Region]
Determine whether there is a hazardous region that needs special attention.
Presence options: {yes, no}
Hazard options: {pedestrian_cluster, accident_area, road_construction, parked_vehicle_risk, intersection_conflict, lane_obstruction, unknown, none}
Direction options: {front, front_left, front_right, left, right, near, far, unknown, none}
[6. Forward Drivability]
Evaluate the forward drivability condition.
Status options: {clear, slow_down_needed, blocked}
[7. Lane Keeping]
Evaluate whether current lane keeping is reasonable.
Reasonableness options: {reasonable, slightly_unreasonable, unreasonable, unknown}
Deviation options: {centered, slight_left, slight_right, severe_left, severe_right, unknown}
[8. Driving Advice]
Choose the main driving advice.
Action options: {accelerate, decelerate, turn_left, turn_right, keep, brake}
[9. Explanation]
Explain the main reason for the driving advice in one short sentence.
Output in the following JSON format:
{
  "weather": {"condition": "", "illumination": ""},
  "traffic_light": {"present": "", "state": ""},
  "traffic_sign": {"present": "", "category": ""},
  "participants": {"count": "", "objects": [{"category": "", "direction": "", "intent": ""}]},
  "hazard_region": {"present": "", "type": "", "direction": ""},
  "forward_drivability": {"status": ""},
  "lane_keeping": {"status": "", "deviation": ""},
  "driving_advice": {"action": ""},
  "explanation": {"reason": ""}
}"""


# =============================================================================
# Dataset
# =============================================================================

class UniscpDataset(Dataset):
    """
    Multi-modal dataset for UNISCP driving scene understanding.
    
    For each sample:
      1. Load left camera image
      2. Load LiDAR PCD, crop to camera FOV, extract (x,y,z,intensity)
      3. Load Radar PCD, crop to camera FOV, extract (x,y,z,doppler,power,recoveredSpeed)
      4. Load caption JSON as target
    """
    
    # Sequence directory structure patterns
    RURAL_SEQUENCES = [
        'RURAL_A0', 'RURAL_A1', 'RURAL_A2',
        'RURAL_B0', 'RURAL_B1', 'RURAL_B2',
    ]
    OTHER_SEQUENCES = [
        'FENDUAN_1', 'KUNSHAN_LUCE6', 'NIGHT_GAOJIAOQIAO',
        'CP_MSCLIKE', 'GARDEN_MSCLIKE', 'LOOP1_MSCLIKE',
    ]
    
    def __init__(self,
                 data_root: str,
                 sequences: Optional[List[str]] = None,
                 tokenizer=None,
                 processor=None,
                 image_pad_num: int = 64,
                 max_lidar_points: int = 40000,
                 max_radar_points: int = 16000,
                 lidar_pc_range: list = None,
                 radar_pc_range: list = None,
                 ):
        super().__init__()
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_pad_num = image_pad_num
        self.max_lidar_points = max_lidar_points
        self.max_radar_points = max_radar_points
        
        self.lidar_pc_range = lidar_pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.radar_pc_range = radar_pc_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
        # Calibration (using RURAL calibration for all sequences for now)
        self.calib = get_rural_calibration()
        
        # Build sample index
        if sequences is None:
            sequences = self.RURAL_SEQUENCES + self.OTHER_SEQUENCES
        
        self.samples = []
        for seq_name in sequences:
            seq_dir = self.data_root / seq_name
            if not seq_dir.exists():
                logger.warning(f"Sequence {seq_name} not found, skipping")
                continue
            self._index_sequence(seq_name, seq_dir)
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(sequences)} sequences")
        print(f"Loaded {len(self.samples)} samples from sequences")
    
    def _find_subdir(self, seq_dir: Path, prefix: str) -> Optional[Path]:
        """Find the actual data subdirectory (handles nested dirs like 1_IMAGE/1_IMAGE/)."""
        top = seq_dir / prefix
        if not top.exists():
            return None
        # Check for nested same-name subdir
        nested = top / prefix
        if nested.exists():
            return nested
        return top
    
    def _index_sequence(self, seq_name: str, seq_dir: Path):
        """Build index of valid samples for one sequence."""
        # Find directories
        img_dir = self._find_subdir(seq_dir, '1_IMAGE')
        lidar_dir = self._find_subdir(seq_dir, '2_LIDAR')
        radar_dir = self._find_subdir(seq_dir, '3_RADAR')
        caption_dir = seq_dir / '6_CAPTION'
        
        if img_dir is None or lidar_dir is None or radar_dir is None:
            logger.warning(f"Missing sensor dirs in {seq_name}")
            return
        if not caption_dir.exists():
            logger.warning(f"Missing caption dir in {seq_name}")
            return
        
        # Load timestamps
        ts_img_file = img_dir / 'timestamp_image_left.txt'
        ts_lidar_file = lidar_dir / 'timestamp_lidar.txt'
        ts_radar_file = radar_dir / 'timestamp_radar.txt'
        
        if not all(f.exists() for f in [ts_img_file, ts_lidar_file, ts_radar_file]):
            logger.warning(f"Missing timestamp files in {seq_name}")
            return
        
        ts_img = load_timestamps(str(ts_img_file))
        ts_lidar = load_timestamps(str(ts_lidar_file))
        ts_radar = load_timestamps(str(ts_radar_file))
        
        # Find LEFT image dir
        left_dir = img_dir / 'LEFT'
        if not left_dir.exists():
            logger.warning(f"Missing LEFT image dir in {seq_name}")
            return
        
        # LiDAR/Radar PCD dirs
        lidar_pcd_dir = lidar_dir / 'PCD'
        radar_pcd_dir = radar_dir / 'PCD'
        
        if not lidar_pcd_dir.exists() or not radar_pcd_dir.exists():
            logger.warning(f"Missing PCD dirs in {seq_name}")
            return
        
        # For each image frame, find matching lidar/radar and caption
        for img_idx_str, img_ts in ts_img:
            # Check image exists
            img_path = left_dir / f"{img_idx_str}.png"
            if not img_path.exists():
                continue
            
            # Check caption exists (captions are aligned with images)
            caption_path = caption_dir / f"{img_idx_str}.json"
            if not caption_path.exists():
                continue
            
            # Find nearest LiDAR and Radar
            lidar_idx = find_nearest_idx(img_ts, ts_lidar, max_diff=0.15)
            radar_idx = find_nearest_idx(img_ts, ts_radar, max_diff=0.15)
            
            if lidar_idx is None or radar_idx is None:
                continue
            
            lidar_path = lidar_pcd_dir / f"{lidar_idx}.pcd"
            radar_path = radar_pcd_dir / f"{radar_idx}.pcd"
            
            if not lidar_path.exists() or not radar_path.exists():
                continue
            
            self.samples.append({
                'seq': seq_name,
                'img_idx': img_idx_str,
                'img_path': str(img_path),
                'lidar_path': str(lidar_path),
                'radar_path': str(radar_path),
                'caption_path': str(caption_path),
            })
    
    def __len__(self):
        return len(self.samples)
    
    def _load_lidar(self, path: str) -> np.ndarray:
        """Load LiDAR PCD and extract (x, y, z, intensity)."""
        try:
            data = read_pcd(path, fields_to_load=['x', 'y', 'z', 'intensity'])
            x = data.get('x', np.zeros(0, dtype=np.float32))
            y = data.get('y', np.zeros(0, dtype=np.float32))
            z = data.get('z', np.zeros(0, dtype=np.float32))
            intensity = data.get('intensity', np.zeros_like(x))
            
            if len(x) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            
            points = np.stack([x, y, z, intensity], axis=-1).astype(np.float32)
            
            # Remove NaN/Inf
            valid = np.all(np.isfinite(points), axis=-1)
            points = points[valid]
            
            # Remove zero points (invalid returns)
            nonzero = np.any(points[:, :3] != 0, axis=-1)
            points = points[nonzero]
            
            return points
        except Exception as e:
            logger.warning(f"Failed to load LiDAR {path}: {e}")
            return np.zeros((0, 4), dtype=np.float32)
    
    def _load_radar(self, path: str) -> np.ndarray:
        """Load Radar PCD and extract (x, y, z, doppler, power, recoveredSpeed).
        
        Radar PCD coordinate convention: Z=forward, X=lateral, Y=vertical.
        The calibration extrinsic R_cr expects -Z=forward (radar -Z maps to camera +Z).
        Fix: negate Z so that the stored +Z (forward) becomes -Z for the extrinsic.
        After this fix, crop_radar_to_fov and project_radar_to_image work correctly.
        """
        try:
            data = read_pcd(path, fields_to_load=[
                'x', 'y', 'z', 'doppler', 'power', 'recoveredSpeed'
            ])
            x = data.get('x', np.zeros(0, dtype=np.float32))
            y = data.get('y', np.zeros(0, dtype=np.float32))
            z = data.get('z', np.zeros(0, dtype=np.float32))
            doppler = data.get('doppler', np.zeros_like(x))
            power = data.get('power', np.zeros_like(x))
            speed = data.get('recoveredSpeed', np.zeros_like(x))
            
            if len(x) == 0:
                return np.zeros((0, 6), dtype=np.float32)
            
            # Radar PCD uses a mirrored coordinate system vs calibration convention.
            # Fix: negate all of x, y, z. Also keep only front-facing points (z > 0).
            # After negation: p_calib = (-x, -y, -z) with z_pcd > 0 → z_calib < 0 (= -Z forward).
            front = z > 0
            x, y, z = -x[front], -y[front], -z[front]
            doppler = doppler[front]
            power   = power[front]
            speed   = speed[front]

            points = np.stack([x, y, z, doppler, power, speed], axis=-1).astype(np.float32)
            
            # Remove NaN/Inf
            valid = np.all(np.isfinite(points), axis=-1)
            points = points[valid]
            
            return points
        except Exception as e:
            logger.warning(f"Failed to load Radar {path}: {e}")
            return np.zeros((0, 6), dtype=np.float32)
    
    def _subsample(self, points: np.ndarray, max_points: int) -> np.ndarray:
        """Random subsample if too many points."""
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            return points[idx]
        return points
    
    def _pad_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        """Pad with zeros to fixed size for batching."""
        n, c = points.shape
        if n >= max_points:
            return points[:max_points]
        padded = np.zeros((max_points, c), dtype=np.float32)
        padded[:n] = points
        return padded
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 1. Load image
        try:
            image = Image.open(sample['img_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {sample['img_path']}: {e}")
            image = Image.new('RGB', (720, 540), color='black')
        
        # 2. Load and crop LiDAR
        lidar_points = self._load_lidar(sample['lidar_path'])
        if len(lidar_points) > 0:
            lidar_points = self.calib.crop_lidar_to_fov(lidar_points)
        lidar_points = self._subsample(lidar_points, self.max_lidar_points)
        
        # 3. Load and crop Radar
        radar_points = self._load_radar(sample['radar_path'])
        if len(radar_points) > 0:
            radar_points = self.calib.crop_radar_to_fov(radar_points)
        radar_points = self._subsample(radar_points, self.max_radar_points)
        
        # 4. Load caption
        try:
            with open(sample['caption_path'], 'r') as f:
                caption = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load caption {sample['caption_path']}: {e}")
            caption = self._get_error_template()
        
        # 5. Process image for DINOv3
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        # 6. Build text input/output
        answer_text = json.dumps(caption, ensure_ascii=False)
        
        # Build conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<image>\n" + USER_PROMPT},
        ]
        
        q_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).replace('<image>', '<|image_pad|>' * self.image_pad_num)
        
        q_input_ids = self.tokenizer(q_text)["input_ids"]
        a_text = answer_text + self.tokenizer.eos_token
        a_input_ids = self.tokenizer(a_text)["input_ids"]
        
        input_ids = q_input_ids + a_input_ids
        labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
        input_ids = input_ids[:-1]
        labels = labels[1:]
        
        # Record actual point counts for masking in collate
        n_lidar = len(lidar_points)
        n_radar = len(radar_points)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "lidar_points": lidar_points.astype(np.float32),
            "radar_points": radar_points.astype(np.float32),
            "n_lidar": n_lidar,
            "n_radar": n_radar,
        }
    
    @staticmethod
    def _get_error_template():
        return {
            "weather": {"condition": "unknown", "illumination": "unknown"},
            "traffic_light": {"present": "unknown", "state": "unknown"},
            "traffic_sign": {"present": "unknown", "category": "unknown"},
            "participants": {"count": "0", "objects": []},
            "hazard_region": {"present": "unknown", "type": "unknown", "direction": "unknown"},
            "forward_drivability": {"status": "unknown"},
            "lane_keeping": {"status": "unknown", "deviation": "unknown"},
            "driving_advice": {"action": "unknown"},
            "explanation": {"reason": "unknown"},
        }


class UniscpDataCollator:
    """
    Custom data collator that handles variable-length point clouds.
    
    Point clouds are passed as lists (not padded tensors) because
    the encoders handle variable-length inputs natively.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict]) -> Dict:
        max_len = max(len(f["input_ids"]) for f in features)
        
        input_ids = []
        labels = []
        pixel_values = []
        lidar_points_list = []
        radar_points_list = []
        
        for f in features:
            # Pad text
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            labels.append(f["labels"] + [self.tokenizer.pad_token_id] * pad_len)
            
            pixel_values.append(f["pixel_values"])
            
            # Point clouds as tensors (variable length, passed as list)
            lidar_pts = torch.from_numpy(f["lidar_points"]).float()
            radar_pts = torch.from_numpy(f["radar_points"]).float()
            
            # Only keep actual points (remove padding zeros)
            n_l = f["n_lidar"]
            n_r = f["n_radar"]
            lidar_points_list.append(lidar_pts[:n_l] if n_l > 0 else lidar_pts[:1] * 0)
            radar_points_list.append(radar_pts[:n_r] if n_r > 0 else radar_pts[:1] * 0)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": torch.stack(pixel_values, dim=0),
            "lidar_points": lidar_points_list,  # list of [N_i, 4] tensors
            "radar_points": radar_points_list,   # list of [N_i, 6] tensors
        }