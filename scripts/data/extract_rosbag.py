"""
ROS Bag(s) + gt_odom.txt → UNISCP Format Extraction Script
===============================================================
Uses `rosbags` 0.11+ (pure Python, NO ROS needed).

API notes for rosbags 0.11:
  - typestore.deserialize_ros1(rawdata, msgtype)
  - get_types_from_msg(msg_text, msgtype_name) → dict
  - typestore.register(types_dict)

Install:
    pip install rosbags opencv-python numpy

Usage:
    # Single bag:
    python extract_rosbag.py \
        --bags cp_2022-02-26.bag \
        --gt_odom gt_odom.txt \
        --output_dir ./ --seq_name CP_A0

    # Multiple bags (same session):
    python extract_rosbag.py \
        --bags garden_2022-05-13_0.bag garden_2022-05-13_1.bag \
        --gt_odom gt_odom.txt \
        --output_dir ./ --seq_name GARDEN_A0
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_typestore, get_types_from_msg, Stores
except ImportError:
    print("ERROR: rosbags not found. Install: pip install rosbags")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv not found. Install: pip install opencv-python")
    sys.exit(1)


# ============================================================
# Topic configuration
# ============================================================
TOPIC_IMAGE = "/rgb_cam/image_raw/compressed"
TOPIC_LIDAR = "/livox/lidar"
TOPIC_RADAR = "/radar_enhanced_pcl"
TOPIC_IMU = "/vectornav/imu"


# ============================================================
# Type store + Livox registration
# ============================================================

def get_typestore_with_livox():
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # Register CustomPoint first (dependency of CustomMsg)
    livox_custom_point_msg = (
        "uint32 offset_time\n"
        "float32 x\n"
        "float32 y\n"
        "float32 z\n"
        "uint8 reflectivity\n"
        "uint8 tag\n"
        "uint8 line\n"
    )
    typestore.register(
        get_types_from_msg(
            livox_custom_point_msg,
            'livox_ros_driver/msg/CustomPoint',
        )
    )

    # Register CustomMsg (depends on CustomPoint)
    livox_custom_msg = (
        "std_msgs/Header header\n"
        "uint64 timebase\n"
        "uint32 point_num\n"
        "uint8 lidar_id\n"
        "uint8[3] rsvd\n"
        "livox_ros_driver/CustomPoint[] points\n"
    )
    typestore.register(
        get_types_from_msg(
            livox_custom_msg,
            'livox_ros_driver/msg/CustomMsg',
        )
    )
    return typestore


# ============================================================
# Multi-bag message iterator
# ============================================================

def iter_messages_multi(bag_paths, topic, typestore):
    """Yield (deserialized_msg, timestamp_nsec) from multiple bags in order."""
    for bag_path in bag_paths:
        with Reader(bag_path) as reader:
            connections = [c for c in reader.connections if c.topic == topic]
            if not connections:
                print(f"    NOTE: {topic} not in {bag_path.name}, skipping")
                continue
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                try:
                    msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    yield msg, timestamp
                except Exception as e:
                    print(f"    WARN: deserialize failed at t={timestamp}: {e}")
                    continue


# ============================================================
# Header stamp extraction
# ============================================================

def header_to_parts(msg):
    return msg.header.stamp.sec, msg.header.stamp.nanosec


# ============================================================
# PCD writer
# ============================================================

def write_pcd(filepath, points, fields):
    n = points.shape[0]
    nf = len(fields)
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        f"SIZE {' '.join(['4'] * nf)}",
        f"TYPE {' '.join(['F'] * nf)}",
        f"COUNT {' '.join(['1'] * nf)}",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA ascii",
    ]
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        for row in points:
            f.write(' '.join(f'{v:.6f}' for v in row) + '\n')


# ============================================================
# 1. Image extraction
# ============================================================

def extract_images(bag_paths, typestore, output_dir):
    img_dir = output_dir / "1_IMAGE" / "1_IMAGE" / "LEFT"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts_file = output_dir / "1_IMAGE" / "1_IMAGE" / "timestamp_image_left.txt"

    timestamps = []
    count = 0
    print(f"\n  [IMAGE] {TOPIC_IMAGE}")

    for msg, ts_nsec in iter_messages_multi(bag_paths, TOPIC_IMAGE, typestore):
        np_arr = np.frombuffer(bytes(msg.data), np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        secs, nsecs = header_to_parts(msg)
        cv2.imwrite(str(img_dir / f"{count:06d}.png"), image)
        timestamps.append(f"{count:06d} {secs} {nsecs:09d}")
        count += 1
        if count % 500 == 0:
            print(f"    ... {count} images")

    with open(ts_file, 'w') as f:
        f.write('\n'.join(timestamps) + '\n')
    print(f"    Done: {count} images")
    return count


# ============================================================
# 2. LiDAR (Livox CustomMsg → PCD)
# ============================================================

def extract_lidar(bag_paths, typestore, output_dir):
    pcd_dir = output_dir / "2_LIDAR" / "2_LIDAR" / "PCD"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    ts_file = output_dir / "2_LIDAR" / "2_LIDAR" / "timestamp_lidar.txt"

    timestamps = []
    count = 0
    print(f"\n  [LIDAR] {TOPIC_LIDAR}")

    for msg, ts_nsec in iter_messages_multi(bag_paths, TOPIC_LIDAR, typestore):
        secs, nsecs = header_to_parts(msg)
        pts = msg.points
        if len(pts) == 0:
            continue

        try:
            points = np.column_stack([
                np.asarray(pts['x'], dtype=np.float32),
                np.asarray(pts['y'], dtype=np.float32),
                np.asarray(pts['z'], dtype=np.float32),
                np.asarray(pts['reflectivity'], dtype=np.float32),
            ])
        except (KeyError, TypeError, IndexError):
            points = np.array(
                [[p.x, p.y, p.z, float(p.reflectivity)] for p in pts],
                dtype=np.float32,
            )

        write_pcd(pcd_dir / f"{count:06d}.pcd", points,
                  fields=["x", "y", "z", "intensity"])
        timestamps.append(f"{count:06d} {secs} {nsecs:09d}")
        count += 1

    with open(ts_file, 'w') as f:
        f.write('\n'.join(timestamps) + '\n')
    print(f"    Done: {count} PCD files")
    return count


# ============================================================
# 3. Radar (sensor_msgs/PointCloud → PCD)
# ============================================================

def extract_radar(bag_paths, typestore, output_dir):
    pcd_dir = output_dir / "3_RADAR" / "3_RADAR" / "PCD"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    ts_file = output_dir / "3_RADAR" / "3_RADAR" / "timestamp_radar.txt"

    timestamps = []
    count = 0
    print(f"\n  [RADAR] {TOPIC_RADAR}")

    for msg, ts_nsec in iter_messages_multi(bag_paths, TOPIC_RADAR, typestore):
        secs, nsecs = header_to_parts(msg)
        pts = msg.points
        if len(pts) == 0:
            continue

        n_pts = len(pts)
        base = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)

        channel_names = []
        channel_cols = []
        for ch in msg.channels:
            channel_names.append(ch.name)
            vals = np.array(ch.values, dtype=np.float32)
            if len(vals) < n_pts:
                vals = np.pad(vals, (0, n_pts - len(vals)))
            channel_cols.append(vals[:n_pts])

        if channel_cols:
            points = np.hstack([base, np.column_stack(channel_cols)])
        else:
            points = base

        fields = ["x", "y", "z"] + channel_names
        write_pcd(pcd_dir / f"{count:06d}.pcd", points, fields=fields)
        timestamps.append(f"{count:06d} {secs} {nsecs:09d}")
        count += 1

    with open(ts_file, 'w') as f:
        f.write('\n'.join(timestamps) + '\n')
    print(f"    Done: {count} PCD files")
    return count


# ============================================================
# 4. IMU → IMU.txt
# ============================================================

def extract_imu(bag_paths, typestore, output_dir):
    """Output: INDEX SECS NSECS ax ay az gx gy gz qw qx qy qz"""
    nav_dir = output_dir / "4_NAVIGATION" / "4_NAVIGATION"
    nav_dir.mkdir(parents=True, exist_ok=True)
    imu_file = nav_dir / "IMU.txt"

    lines = []
    count = 0
    print(f"\n  [IMU] {TOPIC_IMU}")

    for msg, ts_nsec in iter_messages_multi(bag_paths, TOPIC_IMU, typestore):
        secs, nsecs = header_to_parts(msg)

        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z

        line = (
            f"{count:06d} {secs} {nsecs:09d} "
            f"{ax:.8f} {ay:.8f} {az:.8f} "
            f"{gx:.8f} {gy:.8f} {gz:.8f} "
            f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}"
        )
        lines.append(line)
        count += 1

    with open(imu_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"    Done: {count} IMU entries")
    return count


# ============================================================
# 5. gt_odom.txt → GT_ODOM.txt
# ============================================================

def convert_gt_odom(gt_odom_path, output_dir):
    """
    Input:  # timestamp tx ty tz qx qy qz qw
    Output: INDEX SECS NSECS tx ty tz qw qx qy qz
    """
    nav_dir = output_dir / "4_NAVIGATION" / "4_NAVIGATION"
    nav_dir.mkdir(parents=True, exist_ok=True)
    out_file = nav_dir / "GT_ODOM.txt"

    lines = []
    count = 0
    print(f"\n  [GT_ODOM] {gt_odom_path.name}")

    with open(gt_odom_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue

            ts_float = float(parts[0])
            secs = int(ts_float)
            nsecs = int(round((ts_float - secs) * 1e9))

            tx, ty, tz = parts[1], parts[2], parts[3]
            qx, qy, qz, qw = parts[4], parts[5], parts[6], parts[7]

            out_line = (
                f"{count:06d} {secs} {nsecs:09d} "
                f"{tx} {ty} {tz} "
                f"{qw} {qx} {qy} {qz}"
            )
            lines.append(out_line)
            count += 1

    with open(out_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"    Done: {count} GT odom entries")
    return count


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract rosbag(s) + gt_odom → UNISCP format"
    )
    parser.add_argument("--bags", type=str, nargs="+", required=True)
    parser.add_argument("--gt_odom", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--seq_name", type=str, default="CP_A0")
    parser.add_argument("--skip_image", action="store_true")
    parser.add_argument("--skip_lidar", action="store_true")
    parser.add_argument("--skip_radar", action="store_true")
    parser.add_argument("--skip_imu", action="store_true")
    parser.add_argument("--skip_odom", action="store_true")

    args = parser.parse_args()

    bag_paths = [Path(b) for b in sorted(args.bags)]
    for bp in bag_paths:
        if not bp.exists():
            print(f"ERROR: Bag not found: {bp}")
            return

    gt_odom_path = Path(args.gt_odom)
    if not gt_odom_path.exists():
        print(f"ERROR: gt_odom not found: {gt_odom_path}")
        return

    output_dir = Path(args.output_dir) / args.seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ROS Bag(s) + gt_odom → UNISCP Format")
    print("=" * 60)
    print(f"Bags ({len(bag_paths)}):")
    for bp in bag_paths:
        print(f"  - {bp.name}")
    print(f"GT Odom:  {gt_odom_path}")
    print(f"Output:   {output_dir}")

    typestore = get_typestore_with_livox()
    results = {}

    if not args.skip_image:
        results["images"] = extract_images(bag_paths, typestore, output_dir)
    if not args.skip_lidar:
        results["lidar"] = extract_lidar(bag_paths, typestore, output_dir)
    if not args.skip_radar:
        results["radar"] = extract_radar(bag_paths, typestore, output_dir)
    if not args.skip_imu:
        results["imu"] = extract_imu(bag_paths, typestore, output_dir)
    if not args.skip_odom:
        results["gt_odom"] = convert_gt_odom(gt_odom_path, output_dir)

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for key, count in results.items():
        print(f"  {key:>10}: {count} entries")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
