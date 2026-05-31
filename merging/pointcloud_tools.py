#!/usr/bin/env python3
"""Export LingBot-MAP prediction NPZ files to browser-friendly PCD point clouds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MERGING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MERGING_ROOT.parent
LINGBOT_ROOT = PROJECT_ROOT / "lingbot-map"
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

from lingbot_map.utils.geometry import depth_to_world_coords_points  # noqa: E402


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def as_4x4(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"poses must have shape (S,3,4) or (S,4,4), got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses.copy()
    out = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    out[:, :3, :4] = poses
    return out


def extrinsic_for_unprojection(extrinsic: np.ndarray, convention: str) -> np.ndarray:
    poses = as_4x4(extrinsic)
    if convention == "w2c":
        return poses[:, :3, :4].astype(np.float32)
    if convention == "c2w":
        return np.linalg.inv(poses)[:, :3, :4].astype(np.float32)
    raise ValueError("extrinsic_convention must be 'w2c' or 'c2w'")


def images_to_nhwc(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must be 4D, got {images.shape}")
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return np.clip(images.astype(np.float32), 0.0, 1.0)


def save_pcd_binary(points_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points_rgb = np.asarray(points_rgb, dtype=np.float32)
    if points_rgb.ndim != 2 or points_rgb.shape[1] != 6:
        raise ValueError(f"Expected point cloud shape (N,6), got {points_rgb.shape}")

    xyz = points_rgb[:, :3].astype("<f4", copy=False)
    rgb_u8 = np.clip(points_rgb[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
    rgb_u32 = (
        (rgb_u8[:, 0].astype(np.uint32) << 16)
        | (rgb_u8[:, 1].astype(np.uint32) << 8)
        | rgb_u8[:, 2].astype(np.uint32)
    ).astype("<u4")
    rgb_f32 = rgb_u32.view("<f4")

    data = np.empty(
        len(points_rgb),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")],
    )
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["rgb"] = rgb_f32

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(points_rgb)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points_rgb)}\n"
        "DATA binary\n"
    ).encode("ascii")

    with output_path.open("wb") as handle:
        handle.write(header)
        handle.write(data.tobytes())


def export_depth_pcd(
    input_npz: Path,
    output_pcd: Path,
    *,
    extrinsic_convention: str = "w2c",
    conf_threshold: float = 1.5,
    min_depth: float = 0.15,
    max_depth: float = 8.0,
    frame_stride: int = 2,
    pixel_stride: int = 6,
    max_points: int = 800_000,
) -> int:
    predictions = load_predictions(input_npz)
    depth = np.asarray(predictions["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    confidence = np.asarray(predictions["depth_conf"], dtype=np.float32)
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]

    images = images_to_nhwc(predictions["images"])
    intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float32)
    w2c = extrinsic_for_unprojection(predictions["extrinsic"], extrinsic_convention)

    sampled_grid = np.zeros(depth.shape[1:], dtype=bool)
    sampled_grid[:: max(1, pixel_stride), :: max(1, pixel_stride)] = True
    chunks: list[np.ndarray] = []

    for index in range(0, depth.shape[0], max(1, frame_stride)):
        world_points, _, valid_depth = depth_to_world_coords_points(
            depth[index],
            w2c[index],
            intrinsics[index],
        )
        mask = (
            valid_depth
            & sampled_grid
            & np.isfinite(world_points).all(axis=-1)
            & np.isfinite(depth[index])
            & (depth[index] >= min_depth)
            & (depth[index] <= max_depth)
            & (confidence[index] > conf_threshold)
        )
        points = world_points[mask]
        if len(points) == 0:
            continue
        colors = images[index][mask]
        chunks.append(np.concatenate([points, colors], axis=1).astype(np.float32, copy=False))

    if not chunks:
        raise RuntimeError("No valid points were found in the prediction NPZ")

    cloud = np.concatenate(chunks, axis=0)
    if max_points > 0 and len(cloud) > max_points:
        rng = np.random.default_rng(0)
        cloud = cloud[rng.choice(len(cloud), size=max_points, replace=False)]

    save_pcd_binary(cloud, output_pcd)
    return int(len(cloud))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw LingBot-MAP NPZ predictions to binary PCD.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extrinsic_convention", choices=["w2c", "c2w"], default="w2c")
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--pixel_stride", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=800_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_depth_pcd(
        args.input,
        args.output,
        extrinsic_convention=args.extrinsic_convention,
        conf_threshold=args.conf_threshold,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        max_points=args.max_points,
    )
    print(f"Saved: {args.output} ({count:,} points)")


if __name__ == "__main__":
    main()
