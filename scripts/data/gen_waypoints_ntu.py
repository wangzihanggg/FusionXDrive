"""
Dense GT Waypoint Generation for UNISCP Dataset (End-to-End Fixed)
=====================================================================
Reads GNSS JSON files from 7_PLANNING/GNSS/,
computes heading from GPS trajectory (NOT IMU quaternion),
generates sparse ego-frame waypoints, then B-spline interpolates
to produce dense 8-point trajectories at 0.5s intervals.

ROOT CAUSE OF PREVIOUS BUG:
    IMU quaternion does NOT contain correct absolute heading (lacks
    magnetometer calibration). Using IMU yaw ≈ 0° when vehicle heads
    south (true heading ≈ 180°) flips the sign of x, making all
    waypoints appear BEHIND the vehicle.

FIX:
    Compute heading from GPS trajectory:
        heading_ned = atan2(delta_east, delta_north)
    using t0 → t+1s displacement (falls back to longer intervals).

Coordinate convention (ego frame):
    x: forward (vehicle heading direction)
    y: left
    z: up
    origin: current vehicle position (0, 0, 0)

Output: 7_PLANNING/WAYPOINTS/<image_idx>.json  (overwrites existing files)

Usage:
    python gen_waypoints_fixed.py --dataset_root /path/to/UNISCP
    python gen_waypoints_fixed.py --dataset_root /path/to/UNISCP --sequences RURAL_A0
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
    HAS_SCIPY = True
except ImportError:
    print("[WARN] scipy not found, falling back to linear interpolation")
    print("       Install scipy for cubic B-spline: pip install scipy")
    HAS_SCIPY = False


# ============================================================
# Configuration
# ============================================================

EARTH_RADIUS = 6371000.0  # meters (WGS84 mean)

# Sparse waypoint time points (from GNSS JSON)
SPARSE_LABELS = ["t+1s", "t+2s", "t+5s", "t+10s"]
SPARSE_TIMES = [0.0, 1.0, 2.0, 5.0, 10.0]  # including t0

# Dense output time points
DENSE_TIMES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Minimum displacement (meters) to trust GPS heading
MIN_HEADING_DIST = 0.5

# Future keys in order of preference for heading calculation
HEADING_KEYS = ["t+1s", "t+2s", "t+5s", "t+10s"]


# ============================================================
# Coordinate conversion utilities
# ============================================================

def gps_to_enu(lat0, lon0, lat1, lon1):
    """
    Convert GPS difference (WGS84 degrees) to local ENU (East, North) in meters.
    Uses equirectangular approximation — accurate for short distances.
    """
    mean_lat = math.radians((lat0 + lat1) / 2.0)
    north = math.radians(lat1 - lat0) * EARTH_RADIUS
    east = math.radians(lon1 - lon0) * EARTH_RADIUS * math.cos(mean_lat)
    return east, north


def local_to_enu(lat0, lon0, lat1, lon1):
    """
    For datasets where lat/lon are already in local metric coordinates (meters).
    latitude  -> north axis
    longitude -> east axis
    Simply compute the difference.
    """
    north = lat1 - lat0
    east = lon1 - lon0
    return east, north


def detect_coord_type(all_gnss):
    """
    Auto-detect whether coordinates are WGS84 degrees or local meters.

    Heuristic: check if treating coordinates as degrees produces physically
    plausible speeds. If the resulting speed exceeds MAX_PLAUSIBLE_SPEED,
    the coordinates are likely already in meters (local frame).

    Returns:
        "wgs84" or "local"
    """
    MAX_PLAUSIBLE_SPEED = 100.0  # m/s (~360 km/h)

    for gnss_data in all_gnss[:10]:  # check first 10 frames
        gnss = gnss_data["gnss"]
        t0 = gnss.get("t0", {})
        if not t0.get("available", False):
            continue

        lat0, lon0 = t0["latitude"], t0["longitude"]

        # Check t+1s displacement
        t1 = gnss.get("t+1s", {})
        if not t1.get("available", False):
            continue

        # If WGS84, compute distance in meters
        east, north = gps_to_enu(lat0, lon0, t1["latitude"], t1["longitude"])
        dist_as_wgs84 = math.sqrt(east ** 2 + north ** 2)

        # If local, distance is just the coordinate difference
        de = t1["longitude"] - lon0
        dn = t1["latitude"] - lat0
        dist_as_local = math.sqrt(de ** 2 + dn ** 2)

        # WGS84 speed for 1 second
        if dist_as_wgs84 > MAX_PLAUSIBLE_SPEED:
            return "local"

        # Also check: if coordinates are very small (near origin),
        # likely local frame
        if abs(lat0) < 1.0 and abs(lon0) < 1.0 and dist_as_local < MAX_PLAUSIBLE_SPEED:
            return "local"

        return "wgs84"

    return "wgs84"  # default


def get_enu_func(coord_type):
    """Return the appropriate coordinate conversion function."""
    if coord_type == "local":
        return local_to_enu
    return gps_to_enu


def enu_to_ego(east, north, heading_ned):
    """
    Transform ENU displacement to ego frame (forward, left)
    using NED heading (from North, clockwise positive).

    Derivation:
        Vehicle heading in NED: forward = (cos(h), sin(h)) in (N, E)
        Vehicle left direction:  left   = (sin(h), -cos(h)) in (N, E)

        x_forward = dot((north, east), (cos(h), sin(h)))
        y_left    = dot((north, east), (sin(h), -cos(h)))
    """
    cos_h = math.cos(heading_ned)
    sin_h = math.sin(heading_ned)
    x_fwd = cos_h * north + sin_h * east
    y_left = sin_h * north - cos_h * east
    return x_fwd, y_left


# ============================================================
# GPS heading computation
# ============================================================

def compute_gps_heading(gnss, lat0, lon0, to_enu):
    """
    Compute heading from GPS trajectory.
    Uses the shortest available future point (t+1s preferred).
    Falls back to longer intervals if unavailable or too close.

    Returns:
        heading_ned (float): NED heading in radians, or None
        source_key (str): which future point was used, or None
    """
    for key in HEADING_KEYS:
        future = gnss.get(key, {})
        if not future.get("available", False):
            continue

        east, north = to_enu(lat0, lon0,
                             future["latitude"], future["longitude"])
        dist = math.sqrt(east ** 2 + north ** 2)

        if dist < MIN_HEADING_DIST:
            continue  # too close, heading unreliable

        heading = math.atan2(east, north)  # NED convention: atan2(E, N)
        return heading, key

    return None, None


def compute_heading_from_neighbors(all_gnss, idx, to_enu, max_look=10):
    """
    Fallback: compute heading from neighboring GNSS frames
    when current frame's future points are unavailable or too close.
    Looks forward in the sequence to find sufficient displacement.
    """
    current = all_gnss[idx]["gnss"].get("t0", {})
    if not current.get("available", False):
        return None

    lat0 = current["latitude"]
    lon0 = current["longitude"]

    for di in range(1, max_look + 1):
        if idx + di >= len(all_gnss):
            break
        future_t0 = all_gnss[idx + di]["gnss"].get("t0", {})
        if not future_t0.get("available", True):
            continue

        east, north = to_enu(lat0, lon0,
                             future_t0["latitude"],
                             future_t0["longitude"])
        dist = math.sqrt(east ** 2 + north ** 2)
        if dist >= MIN_HEADING_DIST:
            return math.atan2(east, north)

    return None


# ============================================================
# Interpolation
# ============================================================

def interpolate_bspline(sparse_pts, sparse_times=None, dense_times=None):
    """
    B-spline interpolate sparse waypoints to dense.

    Args:
        sparse_pts: list of [x, y, z] for each sparse time (including t0=[0,0,0])
        sparse_times: timestamps, default [0, 1, 2, 5, 10]
        dense_times: output times, default [0.5, 1.0, ..., 4.0]

    Returns:
        list of (x, y, z) tuples at dense_times, or None on failure
    """
    if sparse_times is None:
        sparse_times = np.array(SPARSE_TIMES, dtype=np.float64)
    if dense_times is None:
        dense_times = np.array(DENSE_TIMES, dtype=np.float64)

    xs = np.array([p[0] for p in sparse_pts], dtype=np.float64)
    ys = np.array([p[1] for p in sparse_pts], dtype=np.float64)
    zs = np.array([p[2] for p in sparse_pts], dtype=np.float64)

    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
        return None

    k = min(3, len(sparse_times) - 1)

    if HAS_SCIPY:
        try:
            sx = make_interp_spline(sparse_times, xs, k=k)
            sy = make_interp_spline(sparse_times, ys, k=k)
            sz = make_interp_spline(sparse_times, zs, k=k)
            return [(float(sx(t)), float(sy(t)), float(sz(t))) for t in dense_times]
        except Exception as e:
            print(f"    B-spline failed ({e}), falling back to linear")

    # Linear interpolation fallback
    result = []
    for t in dense_times:
        idx = np.searchsorted(sparse_times, t)
        idx = max(1, min(idx, len(sparse_times) - 1))
        t0, t1 = sparse_times[idx - 1], sparse_times[idx]
        alpha = float((t - t0) / (t1 - t0 + 1e-8))
        alpha = max(0.0, min(1.0, alpha))
        result.append((
            float((1 - alpha) * xs[idx - 1] + alpha * xs[idx]),
            float((1 - alpha) * ys[idx - 1] + alpha * ys[idx]),
            float((1 - alpha) * zs[idx - 1] + alpha * zs[idx]),
        ))
    return result


# ============================================================
# Trajectory statistics
# ============================================================

def compute_stats(dense_pts, dt=0.5):
    """Compute speed and distance metrics from dense waypoints."""
    if not dense_pts or len(dense_pts) < 2:
        return {}

    xs = [p[0] for p in dense_pts]
    ys = [p[1] for p in dense_pts]

    # Speed at each segment
    speeds = []
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        speeds.append(math.sqrt(dx ** 2 + dy ** 2) / dt)

    # Total trajectory length
    total_dist = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        total_dist += math.sqrt(dx ** 2 + dy ** 2)

    return {
        "avg_speed_ms": round(sum(speeds) / len(speeds), 2) if speeds else 0,
        "max_speed_ms": round(max(speeds), 2) if speeds else 0,
        "total_distance_m": round(total_dist, 2),
        "endpoint_x_m": round(xs[-1], 2),
    }


# ============================================================
# Main processing
# ============================================================

def load_all_gnss(gnss_dir):
    """Load all GNSS JSON files sorted by filename."""
    gnss_files = sorted(gnss_dir.glob("*.json"))
    all_data = []
    for gf in gnss_files:
        with open(gf, 'r') as f:
            all_data.append(json.load(f))
    return all_data


def process_sequence(seq_path, output_subdir="WAYPOINTS"):
    """
    Process one sequence end-to-end:
      1. Load all GNSS JSONs
      2. Compute GPS heading for each frame
      3. Generate sparse ego-frame waypoints
      4. B-spline interpolate to dense waypoints
      5. Save output JSON
    """
    seq_name = seq_path.name

    gnss_dir = seq_path / "7_PLANNING" / "GNSS"
    if not gnss_dir.exists():
        print(f"[SKIP] {seq_name}: No 7_PLANNING/GNSS/ directory")
        return 0

    out_dir = seq_path / "7_PLANNING" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gnss = load_all_gnss(gnss_dir)
    if not all_gnss:
        print(f"[SKIP] {seq_name}: No GNSS JSON files found")
        return 0

    # Auto-detect coordinate type
    coord_type = detect_coord_type(all_gnss)
    to_enu = get_enu_func(coord_type)
    print(f"  Coordinate type: {coord_type}"
          f"{' (lat/lon treated as meters)' if coord_type == 'local' else ' (WGS84 degrees)'}")

    saved = 0
    no_heading = 0
    incomplete = 0

    for i, gnss_data in enumerate(tqdm(all_gnss, desc=f"  {seq_name}")):
        image_index = gnss_data["image_index"]
        image_ts = gnss_data["image_timestamp"]
        gnss = gnss_data["gnss"]

        t0 = gnss.get("t0", {})
        if not t0.get("available", False):
            continue

        lat0 = t0["latitude"]
        lon0 = t0["longitude"]
        alt0 = t0["altitude"]

        # ---- Step 1: Compute GPS heading ----
        heading_ned, heading_src = compute_gps_heading(gnss, lat0, lon0, to_enu)

        # Fallback: use neighboring frames
        if heading_ned is None:
            heading_ned = compute_heading_from_neighbors(all_gnss, i, to_enu)
            heading_src = "neighbor_frames"

        if heading_ned is None:
            no_heading += 1
            continue

        # ---- Step 2: Sparse ego-frame waypoints ----
        sparse_pts = [[0.0, 0.0, 0.0]]  # t0
        sparse_waypoints_out = [{
            "label": "t0", "offset_s": 0,
            "x": 0.0, "y": 0.0, "z": 0.0,
        }]
        all_available = True

        for key, offset in [("t+1s", 1), ("t+2s", 2), ("t+5s", 5), ("t+10s", 10)]:
            future = gnss.get(key, {})
            if not future.get("available", False):
                sparse_waypoints_out.append({
                    "label": key, "offset_s": offset,
                    "x": None, "y": None, "z": None,
                    "available": False,
                })
                all_available = False
                continue

            east, north = to_enu(lat0, lon0,
                                future["latitude"], future["longitude"])
            dz = future["altitude"] - alt0
            x_fwd, y_left = enu_to_ego(east, north, heading_ned)

            sparse_pts.append([x_fwd, y_left, dz])
            sparse_waypoints_out.append({
                "label": key, "offset_s": offset,
                "x": round(x_fwd, 4),
                "y": round(y_left, 4),
                "z": round(dz, 4),
                "available": True,
            })

        # ---- Step 3: Dense interpolation ----
        if not all_available or len(sparse_pts) != 5:
            incomplete += 1
            # Still save sparse-only result
            result = {
                "image_index": image_index,
                "image_timestamp": image_ts,
                "heading_rad": round(heading_ned, 6),
                "heading_deg": round(math.degrees(heading_ned), 2),
                "heading_source": f"gps_trajectory ({heading_src})",
                "heading_convention": "NED (from North, clockwise positive)",
                "reference_gps": {
                    "latitude": lat0, "longitude": lon0, "altitude": alt0,
                },
                "coordinate_frame": "ego (x=forward, y=left, z=up)",
                "input_coord_type": coord_type,
                "trajectory_type": "sparse_only",
                "num_waypoints": len(sparse_waypoints_out),
                "waypoints": sparse_waypoints_out,
            }
            out_file = out_dir / f"{image_index}.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            saved += 1
            continue

        dense_pts = interpolate_bspline(sparse_pts)
        if dense_pts is None:
            incomplete += 1
            continue

        # Build dense waypoint list
        dense_waypoints = []
        for t, (x, y, z) in zip(DENSE_TIMES, dense_pts):
            dense_waypoints.append({
                "label": f"t+{t:.1f}s",
                "offset_s": float(t),
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "available": True,
            })

        stats = compute_stats(dense_pts)

        result = {
            "image_index": image_index,
            "image_timestamp": image_ts,
            "heading_rad": round(heading_ned, 6),
            "heading_deg": round(math.degrees(heading_ned), 2),
            "heading_source": f"gps_trajectory ({heading_src})",
            "heading_convention": "NED (from North, clockwise positive)",
            "reference_gps": {
                "latitude": lat0, "longitude": lon0, "altitude": alt0,
            },
            "coordinate_frame": "ego (x=forward, y=left, z=up)",
                "input_coord_type": coord_type,
            "trajectory_type": "dense_bspline",
            "time_range_s": [DENSE_TIMES[0], DENSE_TIMES[-1]],
            "time_step_s": 0.5,
            "num_waypoints": len(dense_waypoints),
            "interpolation": {
                "method": "cubic_bspline" if HAS_SCIPY else "linear",
                "source_times_s": SPARSE_TIMES,
                "output_times_s": DENSE_TIMES,
            },
            "stats": stats,
            "waypoints": dense_waypoints,
            "sparse_waypoints": sparse_waypoints_out,
        }

        out_file = out_dir / f"{image_index}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved += 1

    if no_heading > 0:
        print(f"  WARNING: {no_heading} frames skipped (couldn't determine heading)")
    if incomplete > 0:
        print(f"  INFO: {incomplete} frames have incomplete future waypoints (sparse only)")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Generate dense ego-frame waypoints from GNSS (GPS heading) for UNISCP"
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to UNISCP root directory")
    parser.add_argument("--sequences", type=str, nargs="*", default=None,
                        help="Specific sequences (e.g., RURAL_A0). Default: all subdirs with GNSS data")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: {dataset_root} not found")
        return

    if args.sequences:
        seq_folders = [dataset_root / s for s in args.sequences]
        seq_folders = [s for s in seq_folders if s.exists()]
    else:
        seq_folders = sorted([
            d for d in dataset_root.iterdir()
            if d.is_dir() and (d / "7_PLANNING" / "GNSS").exists()
        ])

    print("=" * 60)
    print("UNISCP Dense Waypoint Generation (GPS Heading - Fixed)")
    print("=" * 60)
    print(f"Sequences:    {[s.name for s in seq_folders]}")
    print(f"Heading:      GPS trajectory (NOT IMU)")
    print(f"Ego frame:    x=forward, y=left, z=up")
    print(f"Dense output: {DENSE_TIMES[0]}s to {DENSE_TIMES[-1]}s, "
          f"step={DENSE_TIMES[1]-DENSE_TIMES[0]}s ({len(DENSE_TIMES)} points)")
    print(f"Interpolation: {'cubic B-spline (scipy)' if HAS_SCIPY else 'linear (no scipy)'}")
    print(f"Output dir:   <seq>/7_PLANNING/WAYPOINTS/ (overwrite)")
    print()

    total = 0
    for seq_path in seq_folders:
        print(f"\n[Processing] {seq_path.name}")
        count = process_sequence(seq_path)
        total += count
        print(f"  Saved: {count} waypoint files")

    print("\n" + "=" * 60)
    print(f"DONE. Total: {total} files")
    print("=" * 60)


if __name__ == "__main__":
    main()