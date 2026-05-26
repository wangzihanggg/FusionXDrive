"""
GNSS Alignment Script for UNISCP Dataset
=============================================
For each image frame, finds the closest GNSS (RTK_GPS) measurement at:
  - current time (t=0)
  - t+1s, t+2s, t+5s, t+10s (nearest neighbor)
and saves as JSON files in RURAL_XX/7_PLANNING/GNSS/

Usage:
    python align_gnss.py --dataset_root /path/to/UNISCP
    python align_gnss.py --dataset_root /path/to/UNISCP --sequences RURAL_A0 RURAL_A1
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


FUTURE_OFFSETS = [0, 1, 2, 5, 10]


def parse_timestamp_file(filepath):
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            idx_str = parts[0]
            sec = int(parts[1])
            nsec = int(parts[2])
            timestamp = sec + nsec * 1e-9
            entries.append((idx_str, timestamp))
    return entries


def parse_rtk_gps_file(filepath):
    timestamps, lats, lons, alts = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 6:
                # RURAL 格式: idx sec nsec lat lon alt
                sec = int(parts[1])
                nsec = int(parts[2])
                timestamps.append(sec + nsec * 1e-9)
                lats.append(float(parts[3]))
                lons.append(float(parts[4]))
                alts.append(float(parts[5]))
            elif len(parts) == 10:
                # CP_MSCLIKE 格式: idx sec nsec x y z qw qx qy qz
                sec = int(parts[1])
                nsec = int(parts[2])
                timestamps.append(sec + nsec * 1e-9)
                lats.append(float(parts[3]))
                lons.append(float(parts[4]))
                alts.append(float(parts[5]))
            elif len(parts) == 14:
                # NIGHT 格式: idx sec nsec yaw pitch roll lat lon alt Ve Vn Vu speed gpstime
                sec = int(parts[1])
                nsec = int(parts[2])
                timestamps.append(sec + nsec * 1e-9)
                lats.append(float(parts[6]))
                lons.append(float(parts[7]))
                alts.append(float(parts[8]))
    return np.array(timestamps), np.array(lats), np.array(lons), np.array(alts)


def find_nearest_index(gps_timestamps, query_time):
    idx = np.searchsorted(gps_timestamps, query_time)
    if idx == 0:
        best = 0
    elif idx >= len(gps_timestamps):
        best = len(gps_timestamps) - 1
    else:
        if abs(gps_timestamps[idx - 1] - query_time) <= abs(gps_timestamps[idx] - query_time):
            best = idx - 1
        else:
            best = idx
    return best, gps_timestamps[best], gps_timestamps[best] - query_time


def process_sequence(seq_path, max_time_gap=1.0):
    seq_name = seq_path.name

    ts_candidates = [
        seq_path / "1_IMAGE" / "1_IMAGE" / "timestamp_image_left.txt",
        seq_path / "1_IMAGE" / "timestamp_image_left.txt",
    ]
    ts_file = next((c for c in ts_candidates if c.exists()), None)
    if ts_file is None:
        print(f"[SKIP] {seq_name}: No timestamp_image_left.txt found")
        return 0

    gps_candidates = [
        seq_path / "4_NAVIGATION" / "4_NAVIGATION" / "RTK_GPS.txt",
        seq_path / "4_NAVIGATION" / "RTK_GPS.txt",
    ]
    gps_file = next((c for c in gps_candidates if c.exists()), None)
    if gps_file is None:
        print(f"[SKIP] {seq_name}: No RTK_GPS.txt found")
        return 0

    image_entries = parse_timestamp_file(ts_file)
    gps_ts, gps_lat, gps_lon, gps_alt = parse_rtk_gps_file(gps_file)

    if len(image_entries) == 0 or len(gps_ts) == 0:
        print(f"[SKIP] {seq_name}: Empty timestamp or GPS data")
        return 0

    print(f"  Images: {len(image_entries)}, GPS points: {len(gps_ts)}")
    print(f"  GPS time range: {gps_ts[-1] - gps_ts[0]:.1f}s")

    output_dir = seq_path / "7_PLANNING" / "GNSS"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx_str, img_ts in tqdm(image_entries, desc=f"  {seq_name}"):
        result = {"image_index": idx_str, "image_timestamp": img_ts, "gnss": {}}

        for offset in FUTURE_OFFSETS:
            query_time = img_ts + offset
            key = f"t+{offset}s" if offset > 0 else "t0"
            best_idx, best_ts, time_diff = find_nearest_index(gps_ts, query_time)

            if abs(time_diff) > max_time_gap:
                result["gnss"][key] = {
                    "available": False,
                    "reason": f"nearest GPS is {abs(time_diff):.3f}s away (>{max_time_gap}s)",
                    "offset_target_s": offset,
                }
            else:
                result["gnss"][key] = {
                    "available": True,
                    "latitude": float(gps_lat[best_idx]),
                    "longitude": float(gps_lon[best_idx]),
                    "altitude": float(gps_alt[best_idx]),
                    "timestamp": float(best_ts),
                    "time_diff_s": round(float(time_diff), 6),
                    "offset_target_s": offset,
                }

        out_file = output_dir / f"{idx_str}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved_count += 1

    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Align GNSS data to image frames for UNISCP")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--sequences", type=str, nargs="*", default=None)
    parser.add_argument("--max_time_gap", type=float, default=1.0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return

    if args.sequences:
        seq_folders = [dataset_root / s for s in args.sequences if (dataset_root / s).exists()]
    else:
        seq_folders = sorted([
            d for d in dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("RURAL_")
        ])

    print("=" * 60)
    print("UNISCP GNSS Alignment")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Sequences: {[s.name for s in seq_folders]}")
    print(f"Future offsets: {FUTURE_OFFSETS}s")
    print(f"Max time gap: {args.max_time_gap}s\n")

    total_saved = 0
    for seq_path in seq_folders:
        print(f"\n[Processing] {seq_path.name}")
        count = process_sequence(seq_path, max_time_gap=args.max_time_gap)
        total_saved += count
        print(f"  Saved: {count} JSON files")

    print("\n" + "=" * 60)
    print(f"DONE. Total JSON files saved: {total_saved}")
    print(f"Output location: <sequence>/7_PLANNING/GNSS/")
    print("=" * 60)


if __name__ == "__main__":
    main()
