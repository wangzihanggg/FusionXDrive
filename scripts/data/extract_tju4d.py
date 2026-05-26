"""
TJU4D ROS Bag → UNISCP-like Format Extraction Script
=========================================================
Handles:
  1. /camera/image_raw        (sensor_msgs/Image)         → PNG images
  2. /rslidar_packets          (rslidar_msgs/rslidarScan)  → PCD (RS-16 decode)
  3. /radar_enhanced_pcl       (sensor_msgs/PointCloud)    → PCD
  4. /imu_data                 (sensor_msgs/Imu)           → IMU.txt
  5. /nmea_sentence            (nmea_msgs/GpsImuforRadar)  → GPS_IMU.txt

Uses `rosbags` 0.11+ (pure Python, NO ROS needed).

Install:
    pip install rosbags opencv-python numpy

Usage:
    python extract_tju4d.py \
        --bag Night_gaojiaoqiao.bag \
        --output_dir ./ --seq_name NIGHT_GAOJIAOQIAO
"""

import os
import sys
import struct
import argparse
import math
import numpy as np
from pathlib import Path

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_typestore, get_types_from_msg, Stores
except ImportError:
    print("ERROR: rosbags not found.  pip install rosbags")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv not found.  pip install opencv-python")
    sys.exit(1)


# ============================================================
# Topic configuration
# ============================================================
TOPIC_IMAGE = "/camera/image_raw"
TOPIC_LIDAR = "/rslidar_packets"
TOPIC_RADAR = "/radar_enhanced_pcl"
TOPIC_IMU   = "/imu_data"
TOPIC_GPS   = "/nmea_sentence"


# ============================================================
# RS-LiDAR-32 packet structure constants
# ============================================================
RS_PKT_HEADER_SIZE = 42
RS_BLOCK_SIZE = 100         # 2 flag + 2 azimuth + 32*3 channels
RS_BLOCKS_PER_PKT = 12
RS_CHANNELS_PER_BLOCK = 32  # 32 channels per block
RS_BLOCK_FLAG = 0xFFEE
RS_DISTANCE_RESOLUTION = 0.005  # 5mm per unit
RS_MIN_DISTANCE = 0.2           # ignore points closer than this


def read_rs32_calibration_from_difop(bag_path, typestore):
    """
    Read vertical & horizontal angle calibration from the first DIFOP packet.

    DIFOP layout (RS-32):
      Bytes 468-563: 32 × 3 bytes → vertical (pitch) angles
      Bytes 564-659: 32 × 3 bytes → horizontal (yaw) corrections
      Format per entry: sign(uint8) + value(uint16 BE), unit = 0.001 degree
    """
    PITCH_OFFSET = 468
    YAW_OFFSET = 564

    vert_angles_deg = [0.0] * 32
    horiz_angles_deg = [0.0] * 32

    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == "/rslidar_packets_difop"]
        if not conns:
            print("    WARN: No DIFOP topic, using default RS-32 angles")
            # Fallback defaults
            defaults = [
                -25.0, -14.638, -3.844, 4.72, -22.5, -12.139, -1.375, 7.196,
                -20.0, -9.666, 1.094, 9.669, -17.5, -7.221, 3.563, 12.123,
                -15.0, -4.81, 6.028, 14.529, -12.5, -2.437, 8.488, 16.843,
                -10.0, -0.108, 10.934, 19.0, -7.5, 2.184, 13.254, 20.843,
            ]
            return [math.radians(a) for a in defaults], [0.0] * 32

        for conn, ts, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            raw = bytes(msg.data)

            # Parse vertical angles
            for ch in range(32):
                off = PITCH_OFFSET + ch * 3
                sign = raw[off]
                val = struct.unpack_from('>H', raw, off + 1)[0]
                angle = val * 0.001  # unit = 0.001 degree
                if sign:  # sign != 0 means negative
                    angle = -angle
                vert_angles_deg[ch] = angle

            # Parse horizontal corrections
            for ch in range(32):
                off = YAW_OFFSET + ch * 3
                sign = raw[off]
                val = struct.unpack_from('>H', raw, off + 1)[0]
                angle = val * 0.001
                if sign:
                    angle = -angle
                horiz_angles_deg[ch] = angle

            break  # only need first DIFOP packet

    print(f"    DIFOP vertical angles (deg):")
    for i in range(32):
        print(f"      ch{i:2d}: vert={vert_angles_deg[i]:+8.3f}°  horiz={horiz_angles_deg[i]:+8.3f}°")

    vert_angles_rad = [math.radians(a) for a in vert_angles_deg]
    horiz_angles_rad = [math.radians(a) for a in horiz_angles_deg]
    return vert_angles_rad, horiz_angles_rad


# ============================================================
# Type store + custom message registration
# ============================================================

def get_typestore_with_custom():
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # --- GpsImuforRadar ---
    gpsimu_msg = (
        "std_msgs/Header header\n"
        "float32 yaw\n"
        "float32 pitch\n"
        "float32 roll\n"
        "float32 lat\n"
        "float32 lon\n"
        "float32 alt\n"
        "float32 Ve\n"
        "float32 Vn\n"
        "float32 Vu\n"
        "float32 speed\n"
        "float64 gpstime\n"
    )
    typestore.register(
        get_types_from_msg(gpsimu_msg, 'nmea_msgs/msg/GpsImuforRadar')
    )

    # --- rslidarPacket ---
    rslidar_packet_msg = (
        "time stamp\n"
        "uint8[1248] data\n"
    )
    typestore.register(
        get_types_from_msg(rslidar_packet_msg, 'rslidar_msgs/msg/rslidarPacket')
    )

    # --- rslidarScan ---
    rslidar_scan_msg = (
        "std_msgs/Header header\n"
        "rslidar_msgs/rslidarPacket[] packets\n"
    )
    typestore.register(
        get_types_from_msg(rslidar_scan_msg, 'rslidar_msgs/msg/rslidarScan')
    )

    return typestore


# ============================================================
# Message iterator
# ============================================================

def iter_messages(bag_path, topic, typestore):
    """Yield (deserialized_msg, timestamp_nsec) from a single bag."""
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"    NOTE: {topic} not in {bag_path.name}, skipping")
            return
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                yield msg, timestamp
            except Exception as e:
                print(f"    WARN: deserialize failed at t={timestamp}: {e}")
                continue


# ============================================================
# Helpers
# ============================================================

def header_to_parts(msg):
    return msg.header.stamp.sec, msg.header.stamp.nanosec


def write_pcd(filepath, points, fields):
    """Write an ASCII PCD file."""
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
# 1. Image extraction (sensor_msgs/Image → PNG)
# ============================================================

def extract_images(bag_path, typestore, output_dir):
    img_dir = output_dir / "1_IMAGE" / "1_IMAGE" / "LEFT"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts_file = output_dir / "1_IMAGE" / "1_IMAGE" / "timestamp_image_left.txt"

    timestamps = []
    count = 0
    print(f"\n  [IMAGE] {TOPIC_IMAGE}")

    for msg, ts_nsec in iter_messages(bag_path, TOPIC_IMAGE, typestore):
        secs, nsecs = header_to_parts(msg)

        h, w = msg.height, msg.width
        encoding = msg.encoding if hasattr(msg, 'encoding') else 'bgr8'

        # Convert raw bytes to numpy array
        raw = bytes(msg.data)

        if encoding in ('bgr8', 'rgb8'):
            image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            if encoding == 'rgb8':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif encoding == 'mono8':
            image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        elif encoding in ('bayer_rggb8', 'bayer_bggr8', 'bayer_gbrg8', 'bayer_grbg8'):
            bayer_map = {
                'bayer_rggb8': cv2.COLOR_BayerRG2BGR,
                'bayer_bggr8': cv2.COLOR_BayerBG2BGR,
                'bayer_gbrg8': cv2.COLOR_BayerGB2BGR,
                'bayer_grbg8': cv2.COLOR_BayerGR2BGR,
            }
            image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
            image = cv2.cvtColor(image, bayer_map[encoding])
        elif encoding == '16UC1':
            image = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
        else:
            # Best effort: try 3-channel
            try:
                image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            except ValueError:
                image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)

        if image is None:
            continue

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
# 2. LiDAR: RS-32 raw packet decode → PCD
# ============================================================

def decode_rs32_scan(msg, vert_angles_rad, horiz_angles_rad):
    """
    Decode one rslidarScan message (multiple packets) into an Nx4 point cloud
    [x, y, z, intensity].

    Uses per-channel vertical and horizontal angles from DIFOP calibration.
    Dedup blocks with identical azimuth (dual-return artifact).
    Filters out 0xFFFF (no return) and enforces min/max distance.
    """
    RS_INVALID_DIST = 65535     # 0xFFFF = no return
    RS_MAX_DISTANCE = 200.0     # RS-32 max range ~200m

    all_points = []

    for pkt in msg.packets:
        raw = bytes(pkt.data)
        if len(raw) != 1248:
            continue

        prev_azimuth_raw = -1  # track for dual-return dedup

        for blk_idx in range(RS_BLOCKS_PER_PKT):
            offset = RS_PKT_HEADER_SIZE + blk_idx * RS_BLOCK_SIZE

            # Check block flag
            flag = struct.unpack_from('>H', raw, offset)[0]
            if flag != RS_BLOCK_FLAG:
                continue

            # Azimuth in 0.01 degree units, big-endian
            azimuth_raw = struct.unpack_from('>H', raw, offset + 2)[0]

            # Dedup: skip if same azimuth as previous block
            if azimuth_raw == prev_azimuth_raw:
                continue
            prev_azimuth_raw = azimuth_raw

            azimuth_base = math.radians(azimuth_raw * 0.01)

            # 32 channels per block
            for ch_idx in range(RS_CHANNELS_PER_BLOCK):
                ch_offset = offset + 4 + ch_idx * 3
                dist_raw = struct.unpack_from('>H', raw, ch_offset)[0]
                intensity = raw[ch_offset + 2]

                # Filter invalid returns
                if dist_raw == 0 or dist_raw >= RS_INVALID_DIST:
                    continue

                distance = dist_raw * RS_DISTANCE_RESOLUTION
                if distance < RS_MIN_DISTANCE or distance > RS_MAX_DISTANCE:
                    continue

                # Per-channel vertical angle + horizontal correction from DIFOP
                vert_angle = vert_angles_rad[ch_idx]
                azimuth = azimuth_base + horiz_angles_rad[ch_idx]

                cos_vert = math.cos(vert_angle)
                x = distance * cos_vert * math.sin(azimuth)
                y = distance * cos_vert * math.cos(azimuth)
                z = distance * math.sin(vert_angle)

                all_points.append([x, y, z, float(intensity)])

    if not all_points:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(all_points, dtype=np.float32)


def extract_lidar(bag_path, typestore, output_dir):
    pcd_dir = output_dir / "2_LIDAR" / "2_LIDAR" / "PCD"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    ts_file = output_dir / "2_LIDAR" / "2_LIDAR" / "timestamp_lidar.txt"

    # Read calibration from DIFOP packet
    print(f"\n  [LIDAR] Reading DIFOP calibration...")
    vert_angles_rad, horiz_angles_rad = read_rs32_calibration_from_difop(
        bag_path, typestore
    )

    timestamps = []
    count = 0
    print(f"  [LIDAR] {TOPIC_LIDAR}  (RS-32 raw decode with DIFOP calibration)")

    for msg, ts_nsec in iter_messages(bag_path, TOPIC_LIDAR, typestore):
        secs, nsecs = header_to_parts(msg)

        points = decode_rs32_scan(msg, vert_angles_rad, horiz_angles_rad)
        if points.shape[0] == 0:
            continue

        write_pcd(pcd_dir / f"{count:06d}.pcd", points,
                  fields=["x", "y", "z", "intensity"])
        timestamps.append(f"{count:06d} {secs} {nsecs:09d}")
        count += 1
        if count % 100 == 0:
            print(f"    ... {count} scans  (last: {points.shape[0]} pts)")

    with open(ts_file, 'w') as f:
        f.write('\n'.join(timestamps) + '\n')
    print(f"    Done: {count} PCD files")
    return count


# ============================================================
# 3. Radar (sensor_msgs/PointCloud → PCD)
# ============================================================

def extract_radar(bag_path, typestore, output_dir):
    pcd_dir = output_dir / "3_RADAR" / "3_RADAR" / "PCD"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    ts_file = output_dir / "3_RADAR" / "3_RADAR" / "timestamp_radar.txt"

    timestamps = []
    count = 0
    print(f"\n  [RADAR] {TOPIC_RADAR}")

    for msg, ts_nsec in iter_messages(bag_path, TOPIC_RADAR, typestore):
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

def extract_imu(bag_path, typestore, output_dir):
    """Output: INDEX SECS NSECS ax ay az gx gy gz qw qx qy qz"""
    nav_dir = output_dir / "4_NAVIGATION" / "4_NAVIGATION"
    nav_dir.mkdir(parents=True, exist_ok=True)
    imu_file = nav_dir / "IMU.txt"

    lines = []
    count = 0
    print(f"\n  [IMU] {TOPIC_IMU}")

    for msg, ts_nsec in iter_messages(bag_path, TOPIC_IMU, typestore):
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
# 5. GPS/IMU (GpsImuforRadar) → GPS_IMU.txt
# ============================================================

def extract_gpsimu(bag_path, typestore, output_dir):
    """
    Output: INDEX SECS NSECS yaw pitch roll lat lon alt Ve Vn Vu speed gpstime
    """
    nav_dir = output_dir / "4_NAVIGATION" / "4_NAVIGATION"
    nav_dir.mkdir(parents=True, exist_ok=True)
    out_file = nav_dir / "GPS_IMU.txt"

    lines = []
    count = 0
    print(f"\n  [GPS_IMU] {TOPIC_GPS}")

    for msg, ts_nsec in iter_messages(bag_path, TOPIC_GPS, typestore):
        secs, nsecs = header_to_parts(msg)

        line = (
            f"{count:06d} {secs} {nsecs:09d} "
            f"{msg.yaw:.8f} {msg.pitch:.8f} {msg.roll:.8f} "
            f"{msg.lat:.8f} {msg.lon:.8f} {msg.alt:.8f} "
            f"{msg.Ve:.8f} {msg.Vn:.8f} {msg.Vu:.8f} "
            f"{msg.speed:.8f} {msg.gpstime:.6f}"
        )
        lines.append(line)
        count += 1

    with open(out_file, 'w') as f:
        # Write header comment
        f.write("# INDEX SECS NSECS yaw pitch roll lat lon alt Ve Vn Vu speed gpstime\n")
        f.write('\n'.join(lines) + '\n')
    print(f"    Done: {count} GPS/IMU entries")
    return count


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract TJU4D rosbag → UNISCP-like format"
    )
    parser.add_argument("--bag", type=str, required=True,
                        help="Path to the .bag file")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--seq_name", type=str, default="NIGHT_GAOJIAOQIAO")
    parser.add_argument("--skip_image", action="store_true")
    parser.add_argument("--skip_lidar", action="store_true")
    parser.add_argument("--skip_radar", action="store_true")
    parser.add_argument("--skip_imu", action="store_true")
    parser.add_argument("--skip_gps", action="store_true")

    args = parser.parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"ERROR: Bag not found: {bag_path}")
        return

    output_dir = Path(args.output_dir) / args.seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TJU4D ROS Bag → UNISCP-like Format")
    print("=" * 60)
    print(f"Bag:      {bag_path}")
    print(f"Output:   {output_dir}")

    typestore = get_typestore_with_custom()
    results = {}

    if not args.skip_image:
        results["images"] = extract_images(bag_path, typestore, output_dir)
    if not args.skip_lidar:
        results["lidar"] = extract_lidar(bag_path, typestore, output_dir)
    if not args.skip_radar:
        results["radar"] = extract_radar(bag_path, typestore, output_dir)
    if not args.skip_imu:
        results["imu"] = extract_imu(bag_path, typestore, output_dir)
    if not args.skip_gps:
        results["gps_imu"] = extract_gpsimu(bag_path, typestore, output_dir)

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    for key, cnt in results.items():
        print(f"  {key:>10}: {cnt} entries")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
