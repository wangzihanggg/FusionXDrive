# # """
# # Waypoint Generation Script for UNISCP Dataset (FIXED)
# # =========================================================
# # Reads GNSS JSON files from 7_PLANNING/GNSS/,
# # uses IMU quaternion for yaw to transform future GPS points
# # into ego-centric (vehicle body frame) waypoints.

# # KEY FIXES from v1:
# #   1. IMU quaternion column order is (qw, qx, qy, qz), NOT (qx, qy, qz, qw)
# #   2. IMU yaw is in NED convention (from North, clockwise positive)
# #   3. Rotation formula corrected accordingly

# # Coordinate convention (ego frame):
# #   - x: forward (vehicle heading direction)
# #   - y: left
# #   - z: up
# #   - origin: current vehicle position (0, 0)

# # Output: 7_PLANNING/WAYPOINTS/<image_idx>.json

# # Usage:
# #     python gen_waypoints.py --dataset_root /path/to/UNISCP
# #     python gen_waypoints.py --dataset_root /path/to/UNISCP --sequences RURAL_A0
# # """

# # import os
# # import json
# # import math
# # import argparse
# # import numpy as np
# # from pathlib import Path
# # from tqdm import tqdm


# # # ============================================================
# # # Coordinate conversion utilities
# # # ============================================================

# # EARTH_RADIUS = 6371000.0  # meters (WGS84 mean)


# # def gps_to_enu(lat0, lon0, lat1, lon1):
# #     """
# #     Convert GPS difference to local ENU (East, North) in meters.
# #     Uses equirectangular approximation — accurate for short distances.
# #     """
# #     mean_lat = math.radians((lat0 + lat1) / 2.0)
# #     north = math.radians(lat1 - lat0) * EARTH_RADIUS
# #     east = math.radians(lon1 - lon0) * EARTH_RADIUS * math.cos(mean_lat)
# #     return east, north


# # def quaternion_to_yaw_ned(qw, qx, qy, qz):
# #     """
# #     Extract yaw from quaternion in NED convention.

# #     IMU quaternion order: (qw, qx, qy, qz)  — columns 9,10,11,12 of IMU.txt
# #     NED yaw: angle from North, clockwise positive (in radians)

# #     NOTE: If qw < 0, negate all components (quaternion double cover)
# #     to ensure consistent yaw extraction.
# #     """
# #     if qw < 0:
# #         qw, qx, qy, qz = -qw, -qx, -qy, -qz

# #     yaw = math.atan2(2.0 * (qw * qz + qx * qy),
# #                      1.0 - 2.0 * (qy**2 + qz**2))
# #     return yaw


# # def enu_to_ego_ned(east, north, yaw_ned):
# #     """
# #     Transform ENU displacement (east, north) to ego frame (forward, left)
# #     using NED yaw (from North, clockwise positive).

# #     In NED convention:
# #       heading_north = cos(yaw), heading_east = sin(yaw)

# #     Body frame rotation:
# #       x_forward =  cos(yaw) * north + sin(yaw) * east
# #       y_left    =  sin(yaw) * north - cos(yaw) * east
# #     """
# #     cos_y = math.cos(yaw_ned)
# #     sin_y = math.sin(yaw_ned)

# #     x_fwd  =  cos_y * north + sin_y * east
# #     y_left =  sin_y * north - cos_y * east

# #     return x_fwd, y_left


# # # ============================================================
# # # IMU parsing
# # # ============================================================

# # def parse_imu_file(filepath):
# #     """
# #     Parse IMU.txt
# #     Format: INDEX SEC NSEC ax ay az gx gy gz qw qx qy qz
# #     (13 columns)

# #     IMPORTANT: Quaternion order is (qw, qx, qy, qz) — cols 9,10,11,12
# #     """
# #     timestamps = []
# #     qw_arr, qx_arr, qy_arr, qz_arr = [], [], [], []

# #     with open(filepath, 'r') as f:
# #         for line in f:
# #             parts = line.strip().split()
# #             if len(parts) != 13:
# #                 continue
# #             sec = int(parts[1])
# #             nsec = int(parts[2])
# #             timestamps.append(sec + nsec * 1e-9)
# #             # Columns 9-12: qw, qx, qy, qz
# #             qw_arr.append(float(parts[9]))
# #             qx_arr.append(float(parts[10]))
# #             qy_arr.append(float(parts[11]))
# #             qz_arr.append(float(parts[12]))

# #     return (
# #         np.array(timestamps),
# #         np.array(qw_arr),
# #         np.array(qx_arr),
# #         np.array(qy_arr),
# #         np.array(qz_arr),
# #     )


# # def find_nearest_imu_yaw(imu_ts, imu_qw, imu_qx, imu_qy, imu_qz, query_time):
# #     """Find the nearest IMU measurement and return its NED yaw."""
# #     idx = np.searchsorted(imu_ts, query_time)
# #     if idx == 0:
# #         best = 0
# #     elif idx >= len(imu_ts):
# #         best = len(imu_ts) - 1
# #     else:
# #         if abs(imu_ts[idx - 1] - query_time) <= abs(imu_ts[idx] - query_time):
# #             best = idx - 1
# #         else:
# #             best = idx

# #     yaw = quaternion_to_yaw_ned(
# #         imu_qw[best], imu_qx[best], imu_qy[best], imu_qz[best]
# #     )
# #     time_diff = imu_ts[best] - query_time
# #     return yaw, time_diff


# # # ============================================================
# # # Main processing
# # # ============================================================

# # def process_sequence(seq_path):
# #     """Process one sequence: read GNSS JSONs + IMU, output waypoints."""
# #     seq_name = seq_path.name

# #     gnss_dir = seq_path / "7_PLANNING" / "GNSS"
# #     if not gnss_dir.exists():
# #         print(f"[SKIP] {seq_name}: No 7_PLANNING/GNSS/ directory")
# #         return 0

# #     imu_candidates = [
# #         seq_path / "4_NAVIGATION" / "4_NAVIGATION" / "IMU.txt",
# #         seq_path / "4_NAVIGATION" / "IMU.txt",
# #     ]
# #     imu_file = None
# #     for c in imu_candidates:
# #         if c.exists():
# #             imu_file = c
# #             break

# #     if imu_file is None:
# #         print(f"[SKIP] {seq_name}: No IMU.txt found")
# #         return 0

# #     print(f"  Loading IMU from {imu_file.relative_to(seq_path)} ...")
# #     imu_ts, imu_qw, imu_qx, imu_qy, imu_qz = parse_imu_file(imu_file)
# #     print(f"  IMU entries: {len(imu_ts)}")

# #     wp_dir = seq_path / "7_PLANNING" / "WAYPOINTS"
# #     wp_dir.mkdir(parents=True, exist_ok=True)

# #     gnss_files = sorted(gnss_dir.glob("*.json"))
# #     if not gnss_files:
# #         print(f"[SKIP] {seq_name}: No GNSS JSON files found")
# #         return 0

# #     saved = 0
# #     for gf in tqdm(gnss_files, desc=f"  {seq_name}"):
# #         with open(gf, 'r') as f:
# #             gnss_data = json.load(f)

# #         image_index = gnss_data["image_index"]
# #         image_ts = gnss_data["image_timestamp"]
# #         gnss = gnss_data["gnss"]

# #         t0 = gnss.get("t0", {})
# #         if not t0.get("available", False):
# #             continue

# #         lat0 = t0["latitude"]
# #         lon0 = t0["longitude"]
# #         alt0 = t0["altitude"]

# #         yaw_ned, imu_dt = find_nearest_imu_yaw(
# #             imu_ts, imu_qw, imu_qx, imu_qy, imu_qz, image_ts
# #         )

# #         waypoints = []

# #         # Current point: always (0, 0, 0)
# #         waypoints.append({
# #             "label": "t0",
# #             "offset_s": 0,
# #             "x": 0.0,
# #             "y": 0.0,
# #             "z": 0.0,
# #         })

# #         # Future points
# #         for key, offset in [("t+1s", 1), ("t+2s", 2), ("t+5s", 5), ("t+10s", 10)]:
# #             future = gnss.get(key, {})
# #             if not future.get("available", False):
# #                 waypoints.append({
# #                     "label": key,
# #                     "offset_s": offset,
# #                     "x": None,
# #                     "y": None,
# #                     "z": None,
# #                     "available": False,
# #                 })
# #                 continue

# #             lat1 = future["latitude"]
# #             lon1 = future["longitude"]
# #             alt1 = future["altitude"]

# #             east, north = gps_to_enu(lat0, lon0, lat1, lon1)
# #             dz = alt1 - alt0

# #             x_fwd, y_left = enu_to_ego_ned(east, north, yaw_ned)

# #             waypoints.append({
# #                 "label": key,
# #                 "offset_s": offset,
# #                 "x": round(x_fwd, 4),
# #                 "y": round(y_left, 4),
# #                 "z": round(dz, 4),
# #                 "available": True,
# #             })

# #         result = {
# #             "image_index": image_index,
# #             "image_timestamp": image_ts,
# #             "yaw_rad": round(yaw_ned, 6),
# #             "yaw_deg": round(math.degrees(yaw_ned), 2),
# #             "yaw_convention": "NED (from North, clockwise positive)",
# #             "imu_time_diff_s": round(imu_dt, 6),
# #             "reference_gps": {
# #                 "latitude": lat0,
# #                 "longitude": lon0,
# #                 "altitude": alt0,
# #             },
# #             "coordinate_frame": "ego (x=forward, y=left, z=up)",
# #             "waypoints": waypoints,
# #         }

# #         out_file = wp_dir / f"{image_index}.json"
# #         with open(out_file, 'w', encoding='utf-8') as f:
# #             json.dump(result, f, indent=2, ensure_ascii=False)
# #         saved += 1

# #     return saved


# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Generate ego-frame waypoints from GNSS + IMU for UNISCP"
# #     )
# #     parser.add_argument("--dataset_root", type=str, required=True,
# #                         help="Path to UNISCP root directory")
# #     parser.add_argument("--sequences", type=str, nargs="*", default=None,
# #                         help="Specific sequences (e.g., RURAL_A0). Default: all RURAL_*")
# #     args = parser.parse_args()

# #     dataset_root = Path(args.dataset_root)
# #     if not dataset_root.exists():
# #         print(f"ERROR: {dataset_root} not found")
# #         return

# #     if args.sequences:
# #         seq_folders = [dataset_root / s for s in args.sequences]
# #         seq_folders = [s for s in seq_folders if s.exists()]
# #     else:
# #         seq_folders = sorted([
# #             d for d in dataset_root.iterdir()
# #             if d.is_dir() and d.name.startswith("RURAL_")
# #         ])

# #     print("=" * 60)
# #     print("UNISCP Waypoint Generation (GNSS + IMU Yaw)")
# #     print("=" * 60)
# #     print(f"Sequences: {[s.name for s in seq_folders]}")
# #     print(f"Ego frame: x=forward, y=left, z=up")
# #     print(f"Yaw convention: NED (from North, CW positive)")
# #     print()

# #     total = 0
# #     for seq_path in seq_folders:
# #         print(f"\n[Processing] {seq_path.name}")
# #         count = process_sequence(seq_path)
# #         total += count
# #         print(f"  Saved: {count} waypoint files")

# #     print("\n" + "=" * 60)
# #     print(f"DONE. Total: {total} files")
# #     print(f"Output: <sequence>/7_PLANNING/WAYPOINTS/")
# #     print("=" * 60)


# # if __name__ == "__main__":
# #     main()


# """
# Waypoint Generation Script for UNISCP Dataset (v3 - GPS Heading)
# ====================================================================
# Reads GNSS JSON files from 7_PLANNING/GNSS/,
# uses GPS trajectory to compute heading (NOT IMU quaternion),
# then transforms future GPS points into ego-centric waypoints.

# ROOT CAUSE OF v2 BUG:
#   IMU quaternion does NOT contain correct absolute heading.
#   The yaw from IMU stays ~178-180° regardless of actual driving direction,
#   indicating the IMU lacks magnetometer calibration or heading initialization.

# FIX:
#   Compute heading from GPS trajectory: heading = atan2(delta_east, delta_north)
#   using t0 -> t+1s displacement. Falls back to longer intervals if needed.

# Coordinate convention (ego frame):
#   - x: forward (vehicle heading direction)
#   - y: left
#   - z: up
#   - origin: current vehicle position (0, 0)

# Output: 7_PLANNING/WAYPOINTS/<image_idx>.json

# Usage:
#     python gen_waypoints_fixed.py --dataset_root /path/to/UNISCP
#     python gen_waypoints_fixed.py --dataset_root /path/to/UNISCP --sequences RURAL_A0
#     python gen_waypoints_fixed.py --dataset_root /path/to/UNISCP --use_rtk
# """

# import os
# import json
# import math
# import argparse
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm


# # ============================================================
# # Coordinate conversion utilities
# # ============================================================

# EARTH_RADIUS = 6371000.0  # meters (WGS84 mean)

# # Waypoint keys in order of preference for heading calculation
# FUTURE_KEYS = ["t+1s", "t+2s", "t+5s", "t+10s"]


# def gps_to_enu(lat0, lon0, lat1, lon1):
#     """
#     Convert GPS difference to local ENU (East, North) in meters.
#     Uses equirectangular approximation — accurate for short distances.
#     """
#     mean_lat = math.radians((lat0 + lat1) / 2.0)
#     north = math.radians(lat1 - lat0) * EARTH_RADIUS
#     east = math.radians(lon1 - lon0) * EARTH_RADIUS * math.cos(mean_lat)
#     return east, north


# def enu_to_ego_ned(east, north, yaw_ned):
#     """
#     Transform ENU displacement (east, north) to ego frame (forward, left)
#     using NED yaw (from North, clockwise positive).

#     Body frame rotation:
#       x_forward =  cos(yaw) * north + sin(yaw) * east
#       y_left    =  sin(yaw) * north - cos(yaw) * east
#     """
#     cos_y = math.cos(yaw_ned)
#     sin_y = math.sin(yaw_ned)
#     x_fwd = cos_y * north + sin_y * east
#     y_left = sin_y * north - cos_y * east
#     return x_fwd, y_left


# def compute_gps_heading(gnss_data, lat0, lon0):
#     """
#     Compute heading from GPS trajectory.
#     Uses the shortest available future point (t+1s preferred).
#     Falls back to longer intervals if t+1s is unavailable or too close.

#     Returns:
#         heading_ned (float): NED heading in radians, or None if can't compute
#         source_key (str): which future point was used
#     """
#     MIN_DIST = 0.5  # minimum displacement (meters) to trust heading

#     for key in FUTURE_KEYS:
#         future = gnss_data.get(key, {})
#         if not future.get("available", False):
#             continue

#         lat1 = future["latitude"]
#         lon1 = future["longitude"]
#         east, north = gps_to_enu(lat0, lon0, lat1, lon1)
#         dist = math.sqrt(east ** 2 + north ** 2)

#         if dist < MIN_DIST:
#             continue  # too close, heading unreliable

#         heading = math.atan2(east, north)  # NED: atan2(E, N)
#         return heading, key

#     return None, None


# # ============================================================
# # Optional: IMU parsing for pitch/roll (kept for reference)
# # ============================================================

# def parse_imu_file(filepath):
#     """
#     Parse IMU.txt
#     Format: INDEX SEC NSEC ax ay az gx gy gz qw qx qy qz
#     (13 columns)
#     """
#     timestamps = []
#     qw_arr, qx_arr, qy_arr, qz_arr = [], [], [], []

#     with open(filepath, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 13:
#                 continue
#             sec = int(parts[1])
#             nsec = int(parts[2])
#             timestamps.append(sec + nsec * 1e-9)
#             qw_arr.append(float(parts[9]))
#             qx_arr.append(float(parts[10]))
#             qy_arr.append(float(parts[11]))
#             qz_arr.append(float(parts[12]))

#     return (
#         np.array(timestamps),
#         np.array(qw_arr),
#         np.array(qx_arr),
#         np.array(qy_arr),
#         np.array(qz_arr),
#     )


# # ============================================================
# # GPS smoothing (optional, for heading stability)
# # ============================================================

# def load_all_gnss(gnss_dir):
#     """Load all GNSS JSONs and return sorted list."""
#     gnss_files = sorted(gnss_dir.glob("*.json"))
#     all_data = []
#     for gf in gnss_files:
#         with open(gf, 'r') as f:
#             data = json.load(f)
#         all_data.append(data)
#     return all_data


# def smooth_heading_from_neighbors(all_gnss, idx, window=5):
#     """
#     For very slow-moving or stationary frames, use neighboring frames
#     to estimate heading more robustly.
#     """
#     headings = []
#     center = all_gnss[idx]
#     lat0 = center["gnss"]["t0"]["latitude"]
#     lon0 = center["gnss"]["t0"]["longitude"]

#     for di in range(1, window + 1):
#         if idx + di < len(all_gnss):
#             future = all_gnss[idx + di]["gnss"]["t0"]
#             if future.get("available", True):
#                 e, n = gps_to_enu(lat0, lon0, future["latitude"], future["longitude"])
#                 dist = math.sqrt(e**2 + n**2)
#                 if dist > 0.3:
#                     headings.append(math.atan2(e, n))
#                     break

#     if headings:
#         return headings[0]
#     return None


# # ============================================================
# # Main processing
# # ============================================================

# def process_sequence(seq_path, use_rtk=False):
#     """Process one sequence: read GNSS JSONs, compute GPS heading, output waypoints."""
#     seq_name = seq_path.name

#     gnss_dir = seq_path / "7_PLANNING" / "GNSS"
#     if not gnss_dir.exists():
#         print(f"[SKIP] {seq_name}: No 7_PLANNING/GNSS/ directory")
#         return 0

#     wp_dir = seq_path / "7_PLANNING" / "WAYPOINTS"
#     wp_dir.mkdir(parents=True, exist_ok=True)

#     # Load all GNSS data for neighbor-based heading fallback
#     all_gnss = load_all_gnss(gnss_dir)
#     if not all_gnss:
#         print(f"[SKIP] {seq_name}: No GNSS JSON files found")
#         return 0

#     # Build index map
#     idx_map = {d["image_index"]: i for i, d in enumerate(all_gnss)}

#     saved = 0
#     no_heading = 0

#     for i, gnss_data in enumerate(tqdm(all_gnss, desc=f"  {seq_name}")):
#         image_index = gnss_data["image_index"]
#         image_ts = gnss_data["image_timestamp"]
#         gnss = gnss_data["gnss"]

#         t0 = gnss.get("t0", {})
#         if not t0.get("available", False):
#             continue

#         lat0 = t0["latitude"]
#         lon0 = t0["longitude"]
#         alt0 = t0["altitude"]

#         # Compute heading from GPS trajectory
#         heading_ned, heading_source = compute_gps_heading(gnss, lat0, lon0)

#         # Fallback: use neighboring frames
#         if heading_ned is None:
#             heading_ned = smooth_heading_from_neighbors(all_gnss, i)
#             heading_source = "neighbor_frames"

#         if heading_ned is None:
#             no_heading += 1
#             continue

#         # Build waypoints
#         waypoints = [{
#             "label": "t0",
#             "offset_s": 0,
#             "x": 0.0,
#             "y": 0.0,
#             "z": 0.0,
#         }]

#         for key, offset in [("t+1s", 1), ("t+2s", 2), ("t+5s", 5), ("t+10s", 10)]:
#             future = gnss.get(key, {})
#             if not future.get("available", False):
#                 waypoints.append({
#                     "label": key,
#                     "offset_s": offset,
#                     "x": None,
#                     "y": None,
#                     "z": None,
#                     "available": False,
#                 })
#                 continue

#             lat1 = future["latitude"]
#             lon1 = future["longitude"]
#             alt1 = future["altitude"]

#             east, north = gps_to_enu(lat0, lon0, lat1, lon1)
#             dz = alt1 - alt0
#             x_fwd, y_left = enu_to_ego_ned(east, north, heading_ned)

#             waypoints.append({
#                 "label": key,
#                 "offset_s": offset,
#                 "x": round(x_fwd, 4),
#                 "y": round(y_left, 4),
#                 "z": round(dz, 4),
#                 "available": True,
#             })

#         result = {
#             "image_index": image_index,
#             "image_timestamp": image_ts,
#             "yaw_rad": round(heading_ned, 6),
#             "yaw_deg": round(math.degrees(heading_ned), 2),
#             "yaw_source": f"gps_trajectory ({heading_source})",
#             "yaw_convention": "NED (from North, clockwise positive)",
#             "reference_gps": {
#                 "latitude": lat0,
#                 "longitude": lon0,
#                 "altitude": alt0,
#             },
#             "coordinate_frame": "ego (x=forward, y=left, z=up)",
#             "waypoints": waypoints,
#         }

#         out_file = wp_dir / f"{image_index}.json"
#         with open(out_file, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)
#         saved += 1

#     if no_heading > 0:
#         print(f"  WARNING: {no_heading} frames skipped (couldn't determine heading)")

#     return saved


# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate ego-frame waypoints from GNSS (GPS heading) for UNISCP"
#     )
#     parser.add_argument("--dataset_root", type=str, required=True,
#                         help="Path to UNISCP root directory")
#     parser.add_argument("--sequences", type=str, nargs="*", default=None,
#                         help="Specific sequences (e.g., RURAL_A0). Default: all RURAL_*")
#     parser.add_argument("--use_rtk", action="store_true",
#                         help="Use RTK_GPS for higher precision (if available)")
#     args = parser.parse_args()

#     dataset_root = Path(args.dataset_root)
#     if not dataset_root.exists():
#         print(f"ERROR: {dataset_root} not found")
#         return

#     if args.sequences:
#         seq_folders = [dataset_root / s for s in args.sequences]
#         seq_folders = [s for s in seq_folders if s.exists()]
#     else:
#         seq_folders = sorted([
#             d for d in dataset_root.iterdir()
#             if d.is_dir() and d.name.startswith("RURAL_")
#         ])

#     print("=" * 60)
#     print("UNISCP Waypoint Generation (GPS Heading - FIXED)")
#     print("=" * 60)
#     print(f"Sequences: {[s.name for s in seq_folders]}")
#     print(f"Heading source: GPS trajectory (NOT IMU)")
#     print(f"Ego frame: x=forward, y=left, z=up")
#     print()

#     total = 0
#     for seq_path in seq_folders:
#         print(f"\n[Processing] {seq_path.name}")
#         count = process_sequence(seq_path, use_rtk=args.use_rtk)
#         total += count
#         print(f"  Saved: {count} waypoint files")

#     print("\n" + "=" * 60)
#     print(f"DONE. Total: {total} files")
#     print(f"Output: <sequence>/7_PLANNING/WAYPOINTS/")
#     print("=" * 60)


# if __name__ == "__main__":
#     main()

"""
Generate Dense GT Waypoints for UNISCP Dataset
===================================================
Reads existing sparse waypoints (t+1s, t+2s, t+5s, t+10s) from WAYPOINTS/,
applies B-spline interpolation to generate dense 8-point trajectories
at 0.5s intervals (t=0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0s).

Input:  7_PLANNING/WAYPOINTS/<idx>.json  (4 sparse waypoints)
Output: 7_PLANNING/WAYPOINTS/<idx>.json  (8 dense waypoints)

Prerequisites:
    - Run gen_waypoints_fixed.py first (GPS heading corrected waypoints)
    - Requires scipy for B-spline interpolation

Usage:
    python gen_dense_waypoints.py --dataset_root /path/to/UNISCP
    python gen_dense_waypoints.py --dataset_root /path/to/UNISCP --sequences RURAL_A0
"""

import os
import json
import math
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from scipy.interpolate import make_interp_spline
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


# Output time points
DENSE_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# Input time points (including t0)
SPARSE_TIMES = [0.0, 1.0, 2.0, 5.0, 10.0]


def interpolate_waypoints(sparse_wp_list, sparse_times=None, dense_times=None):
    """
    B-spline interpolate sparse waypoints to dense.

    Args:
        sparse_wp_list: list of [x, y, z] for each sparse time
                        First element should be t0 = [0, 0, 0]
        sparse_times: time stamps, default [0, 1, 2, 5, 10]
        dense_times: output times, default [0.5, 1.0, ..., 4.0]

    Returns:
        list of {"label": "t+0.5s", "x": ..., "y": ..., "z": ..., ...}
    """
    if sparse_times is None:
        sparse_times = np.array(SPARSE_TIMES)
    if dense_times is None:
        dense_times = np.array(DENSE_TIMES)

    # Extract xyz arrays
    xs = np.array([wp[0] for wp in sparse_wp_list])
    ys = np.array([wp[1] for wp in sparse_wp_list])
    zs = np.array([wp[2] for wp in sparse_wp_list])

    # Check for None/NaN
    if any(x is None or np.isnan(x) for x in xs):
        return None

    k = min(3, len(sparse_times) - 1)

    try:
        spline_x = make_interp_spline(sparse_times, xs, k=k)
        spline_y = make_interp_spline(sparse_times, ys, k=k)
        spline_z = make_interp_spline(sparse_times, zs, k=k)

        dense_wp = []
        for t in dense_times:
            dense_wp.append({
                "label": f"t+{t:.1f}s",
                "offset_s": float(t),
                "x": round(float(spline_x(t)), 4),
                "y": round(float(spline_y(t)), 4),
                "z": round(float(spline_z(t)), 4),
                "available": True,
            })
        return dense_wp

    except Exception as e:
        print(f"  WARNING: B-spline failed: {e}, using linear interpolation")
        dense_wp = []
        for t in dense_times:
            # Linear interpolation fallback
            idx = np.searchsorted(sparse_times, t)
            idx = max(1, min(idx, len(sparse_times) - 1))
            t0, t1 = sparse_times[idx - 1], sparse_times[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-8)
            alpha = max(0, min(1, alpha))

            dense_wp.append({
                "label": f"t+{t:.1f}s",
                "offset_s": float(t),
                "x": round(float((1 - alpha) * xs[idx - 1] + alpha * xs[idx]), 4),
                "y": round(float((1 - alpha) * ys[idx - 1] + alpha * ys[idx]), 4),
                "z": round(float((1 - alpha) * zs[idx - 1] + alpha * zs[idx]), 4),
                "available": True,
            })
        return dense_wp


def compute_trajectory_stats(dense_wp, dt=0.5):
    """Compute velocity, acceleration, smoothness metrics."""
    if dense_wp is None:
        return {}

    xs = [wp["x"] for wp in dense_wp]
    ys = [wp["y"] for wp in dense_wp]

    # Velocity at each segment
    vx = [(xs[i + 1] - xs[i]) / dt for i in range(len(xs) - 1)]
    vy = [(ys[i + 1] - ys[i]) / dt for i in range(len(ys) - 1)]
    speeds = [math.sqrt(vx[i] ** 2 + vy[i] ** 2) for i in range(len(vx))]

    # Acceleration
    ax = [(vx[i + 1] - vx[i]) / dt for i in range(len(vx) - 1)]
    ay = [(vy[i + 1] - vy[i]) / dt for i in range(len(vy) - 1)]

    return {
        "avg_speed_ms": round(sum(speeds) / len(speeds), 2),
        "max_speed_ms": round(max(speeds), 2),
        "total_distance_m": round(xs[-1] if xs else 0, 2),
    }


def process_sequence(seq_path, overwrite=False):
    """Process one sequence."""
    seq_name = seq_path.name

    sparse_dir = seq_path / "7_PLANNING" / "WAYPOINTS"
    if not sparse_dir.exists():
        print(f"[SKIP] {seq_name}: No WAYPOINTS directory")
        return 0

    dense_dir = seq_path / "7_PLANNING" / "WAYPOINTS"
    dense_dir.mkdir(parents=True, exist_ok=True)

    sparse_files = sorted(sparse_dir.glob("*.json"))
    if not sparse_files:
        print(f"[SKIP] {seq_name}: No waypoint files")
        return 0

    saved = 0
    skipped = 0

    for sf in tqdm(sparse_files, desc=f"  {seq_name}"):
        out_file = dense_dir / sf.name

        if out_file.exists() and not overwrite:
            saved += 1
            continue

        with open(sf, 'r') as f:
            data = json.load(f)

        # Extract sparse waypoints (t0 + t+1s, t+2s, t+5s, t+10s)
        wp_dict = {w["label"]: w for w in data.get("waypoints", [])}

        sparse_list = [[0.0, 0.0, 0.0]]  # t0
        all_available = True

        for label in ["t+1s", "t+2s", "t+5s", "t+10s"]:
            wp = wp_dict.get(label, {})
            if wp.get("available", False) and wp.get("x") is not None:
                sparse_list.append([wp["x"], wp["y"], wp.get("z", 0.0)])
            else:
                all_available = False
                break

        if not all_available or len(sparse_list) != 5:
            skipped += 1
            continue

        # B-spline interpolation
        dense_wp = interpolate_waypoints(sparse_list)
        if dense_wp is None:
            skipped += 1
            continue

        # Compute stats
        stats = compute_trajectory_stats(dense_wp)

        # Build output JSON
        result = {
            "image_index": data["image_index"],
            "image_timestamp": data["image_timestamp"],
            "yaw_rad": data.get("yaw_rad"),
            "yaw_deg": data.get("yaw_deg"),
            "yaw_source": data.get("yaw_source", "unknown"),
            "reference_gps": data.get("reference_gps"),
            "coordinate_frame": "ego (x=forward, y=left, z=up)",
            "trajectory_type": "dense_bspline",
            "time_range_s": [0.5, 4.0],
            "time_step_s": 0.5,
            "num_waypoints": len(dense_wp),
            "interpolation": {
                "method": "cubic_bspline",
                "source_times_s": SPARSE_TIMES,
                "output_times_s": DENSE_TIMES,
            },
            "stats": stats,
            "waypoints": dense_wp,
            # Keep original sparse for reference
            "sparse_waypoints": data.get("waypoints", []),
        }

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved += 1

    if skipped > 0:
        print(f"  Skipped {skipped} frames (missing waypoints)")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Generate dense B-spline interpolated waypoints"
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--sequences", type=str, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    if args.sequences:
        seq_folders = [dataset_root / s for s in args.sequences]
        seq_folders = [s for s in seq_folders if s.exists()]
    else:
        seq_folders = sorted([
            d for d in dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("RURAL_")
        ])

    print("=" * 60)
    print("Dense Waypoint Generation (B-spline Interpolation)")
    print("=" * 60)
    print(f"Sequences: {[s.name for s in seq_folders]}")
    print(f"Output: 0.5s intervals, t=0.5s to t=4.0s ({len(DENSE_TIMES)} points)")
    print(f"Method: Cubic B-spline from sparse 5-point GT")
    print()

    total = 0
    for seq_path in seq_folders:
        print(f"\n[Processing] {seq_path.name}")
        count = process_sequence(seq_path, overwrite=args.overwrite)
        total += count
        print(f"  Saved: {count} dense waypoint files")

    print("\n" + "=" * 60)
    print(f"DONE. Total: {total} files")
    print(f"Output: <sequence>/7_PLANNING/WAYPOINTS/")
    print("=" * 60)


if __name__ == "__main__":
    main()