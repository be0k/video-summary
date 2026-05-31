"""Point cloud export helpers for demo predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from lingbot_map.utils.geometry import depth_to_world_coords_points


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _images_to_nhwc(images: np.ndarray) -> np.ndarray:
    images = np.asarray(_to_numpy(images))
    if images.ndim != 4:
        raise ValueError(f"images must have shape (S,3,H,W) or (S,H,W,3), got {images.shape}")
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    if images.shape[-1] != 3:
        raise ValueError(f"images must have 3 color channels, got {images.shape}")
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return np.clip(images.astype(np.float32), 0.0, 1.0)


def _as_4x4(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(_to_numpy(extrinsic), dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsic must have shape (S,3,4) or (S,4,4), got {extrinsic.shape}")
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    out = np.tile(np.eye(4, dtype=np.float64), (extrinsic.shape[0], 1, 1))
    out[:, :3, :4] = extrinsic
    return out


def _get_confidence(predictions: dict, source: str) -> Optional[np.ndarray]:
    if source == "pointmap":
        conf = predictions.get("world_points_conf", predictions.get("depth_conf"))
    else:
        conf = predictions.get("depth_conf", predictions.get("world_points_conf"))
    if conf is None:
        return None
    conf = np.asarray(_to_numpy(conf), dtype=np.float32)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    return conf


def _apply_optional_sky_mask(
    conf: Optional[np.ndarray],
    predictions: dict,
    mask_sky: bool,
    image_folder: Optional[str],
    sky_mask_dir: Optional[str],
    sky_mask_visualization_dir: Optional[str],
) -> Optional[np.ndarray]:
    if conf is None or not mask_sky:
        return conf

    from lingbot_map.vis.sky_segmentation import apply_sky_segmentation

    image_paths = predictions.get("image_paths")
    if image_paths is not None:
        image_paths = [str(x) for x in np.asarray(image_paths).tolist()]
    images = predictions.get("images")
    return apply_sky_segmentation(
        conf,
        image_folder=image_folder,
        image_paths=image_paths,
        images=images,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
    )


def predictions_to_pointcloud(
    predictions: dict,
    conf_threshold: float = 1.5,
    downsample_factor: int = 10,
    source: str = "depth",
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
    max_points: int = 6_000_000,
    sample_seed: int = 0,
) -> np.ndarray:
    """Convert prediction arrays to a colored point cloud.

    Args:
        predictions: Demo prediction dict containing images, extrinsic, intrinsic,
            and either depth or world_points.
        conf_threshold: Absolute confidence cutoff, matching the interactive viewer.
        downsample_factor: Keep every N-th valid point per frame.
        source: ``"depth"`` unprojects depth with camera poses; ``"pointmap"``
            uses the model's world_points tensor.
        mask_sky: If true, multiply confidence by the cached/generated sky mask.
        max_points: Randomly cap the final cloud to this many points. Use 0 for no cap.

    Returns:
        ``(N, 6)`` float32 array: xyz + rgb in [0, 1].
    """
    source = source.lower()
    if source not in {"depth", "pointmap"}:
        raise ValueError("source must be 'depth' or 'pointmap'")

    if "images" not in predictions:
        raise ValueError("predictions must contain images for colored point cloud export")
    images = _images_to_nhwc(predictions["images"])
    num_frames = images.shape[0]
    conf = _get_confidence(predictions, source)
    conf = _apply_optional_sky_mask(
        conf,
        predictions,
        mask_sky=mask_sky,
        image_folder=image_folder,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
    )

    downsample_factor = max(1, int(downsample_factor))
    all_points = []
    all_colors = []

    if source == "pointmap":
        if "world_points" not in predictions:
            raise ValueError("source='pointmap' requires predictions['world_points']")
        world_points_all = np.asarray(_to_numpy(predictions["world_points"]), dtype=np.float32)
        if world_points_all.shape[0] != num_frames:
            raise ValueError("world_points frame count does not match images")

        for i in range(num_frames):
            points = world_points_all[i].reshape(-1, 3)
            colors = images[i].reshape(-1, 3)
            mask = np.isfinite(points).all(axis=1)
            if conf is not None:
                mask &= conf[i].reshape(-1) > conf_threshold
            points = points[mask]
            colors = colors[mask]
            if downsample_factor > 1:
                points = points[::downsample_factor]
                colors = colors[::downsample_factor]
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
    else:
        if "depth" not in predictions:
            raise ValueError("source='depth' requires predictions['depth']")
        if "extrinsic" not in predictions or "intrinsic" not in predictions:
            raise ValueError("source='depth' requires extrinsic and intrinsic")
        depth_all = np.asarray(_to_numpy(predictions["depth"]), dtype=np.float32)
        if depth_all.ndim == 4 and depth_all.shape[-1] == 1:
            depth_all = depth_all[..., 0]
        intrinsics = np.asarray(_to_numpy(predictions["intrinsic"]), dtype=np.float32)
        w2c = _as_4x4(predictions["extrinsic"])[:, :3, :4].astype(np.float32)

        for i in range(num_frames):
            world_points, _, valid_depth = depth_to_world_coords_points(
                depth_all[i],
                w2c[i],
                intrinsics[i],
            )
            points = world_points.reshape(-1, 3)
            colors = images[i].reshape(-1, 3)
            mask = valid_depth.reshape(-1) & np.isfinite(points).all(axis=1)
            if conf is not None:
                mask &= conf[i].reshape(-1) > conf_threshold
            points = points[mask]
            colors = colors[mask]
            if downsample_factor > 1:
                points = points[::downsample_factor]
                colors = colors[::downsample_factor]
            if len(points) > 0:
                all_points.append(points.astype(np.float32, copy=False))
                all_colors.append(colors.astype(np.float32, copy=False))

    if not all_points:
        raise ValueError("No valid points survived filtering; lower --conf_threshold or downsampling")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    cloud = np.concatenate([points, colors], axis=1).astype(np.float32, copy=False)

    if max_points and max_points > 0 and len(cloud) > max_points:
        rng = np.random.default_rng(sample_seed)
        keep = rng.choice(len(cloud), size=int(max_points), replace=False)
        keep.sort()
        cloud = cloud[keep]

    return cloud


def save_pointcloud(points_rgb: np.ndarray, output_path: str | Path) -> None:
    """Save ``(N,6)`` xyzrgb cloud as PCD or PLY based on file extension."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".pcd":
        save_pcd_binary(points_rgb, output_path)
    elif suffix == ".ply":
        save_ply_binary(points_rgb, output_path)
    elif suffix == ".npz":
        np.savez_compressed(output_path, points=points_rgb[:, :3], colors=points_rgb[:, 3:6])
    else:
        raise ValueError("Point cloud path must end with .pcd, .ply, or .npz")


def save_pcd_binary(points_rgb: np.ndarray, output_path: str | Path) -> None:
    """Save a PCL-compatible binary PCD with packed RGB."""
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

    with Path(output_path).open("wb") as f:
        f.write(header)
        f.write(data.tobytes())


def save_ply_binary(points_rgb: np.ndarray, output_path: str | Path) -> None:
    """Save a binary little-endian PLY with RGB colors."""
    points_rgb = np.asarray(points_rgb, dtype=np.float32)
    if points_rgb.ndim != 2 or points_rgb.shape[1] != 6:
        raise ValueError(f"Expected point cloud shape (N,6), got {points_rgb.shape}")

    rgb_u8 = np.clip(points_rgb[:, 3:6] * 255.0, 0, 255).astype(np.uint8)
    data = np.empty(
        len(points_rgb),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    data["x"] = points_rgb[:, 0]
    data["y"] = points_rgb[:, 1]
    data["z"] = points_rgb[:, 2]
    data["red"] = rgb_u8[:, 0]
    data["green"] = rgb_u8[:, 1]
    data["blue"] = rgb_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points_rgb)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with Path(output_path).open("wb") as f:
        f.write(header)
        f.write(data.tobytes())
