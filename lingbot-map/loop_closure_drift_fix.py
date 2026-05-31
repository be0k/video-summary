#!/usr/bin/env python3
"""Reduce loop-drift ghosting in demo.py reconstructions.

This script does not force the first and last camera poses to be identical.
That is usually wrong for handheld loops: returning to the same place does not
mean looking in the same direction. Instead it finds or accepts a visual loop
pair, estimates a small source-to-target alignment from local depth point
clouds with ICP, and distributes that correction smoothly along the drifted
part of the trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lingbot_map.utils.geometry import depth_to_world_coords_points
from lingbot_map.utils.pointcloud_export import predictions_to_pointcloud, save_pointcloud


@dataclass
class IcpReport:
    fitness: float
    rmse: float
    points_source: int
    points_target: int
    translation: float
    rotation_deg: float


def _as_4x4(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"poses must have shape (S,3,4) or (S,4,4), got {poses.shape}")
    if poses.shape[-2:] == (4, 4):
        return poses.copy()
    out = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    out[:, :3, :4] = poses
    return out


def _resolve_index(index: int, length: int) -> int:
    if index < 0:
        index = length + index
    if index < 0 or index >= length:
        raise IndexError(f"frame index {index} out of range for {length} poses")
    return index


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    rotvec = Rotation.from_matrix(rotation_matrix).as_rotvec()
    return float(np.degrees(np.linalg.norm(rotvec)))


def _poses_to_c2w(extrinsic: np.ndarray, convention: str) -> np.ndarray:
    poses = _as_4x4(extrinsic)
    if convention == "w2c":
        return np.linalg.inv(poses)
    if convention == "c2w":
        return poses
    raise ValueError("extrinsic_convention must be 'w2c' or 'c2w'")


def _c2w_to_extrinsic(c2w: np.ndarray, convention: str) -> np.ndarray:
    if convention == "w2c":
        return np.linalg.inv(c2w)[:, :3, :4].astype(np.float32)
    if convention == "c2w":
        return c2w[:, :3, :4].astype(np.float32)
    raise ValueError("extrinsic_convention must be 'w2c' or 'c2w'")


def _extrinsic_for_unprojection(extrinsic: np.ndarray, convention: str) -> np.ndarray:
    poses = _as_4x4(extrinsic)
    if convention == "w2c":
        return poses[:, :3, :4].astype(np.float32)
    return np.linalg.inv(poses)[:, :3, :4].astype(np.float32)


def _ease_smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _correction_alphas(
    num_frames: int,
    closure_start: int,
    closure_end: int,
    ramp_start: int | None,
) -> np.ndarray:
    if closure_end <= closure_start:
        raise ValueError("--closure_end must be greater than --closure_start")
    if ramp_start is None:
        ramp_start = closure_start
    ramp_start = _resolve_index(ramp_start, num_frames)
    if ramp_start > closure_end:
        raise ValueError("--ramp_start must be <= --closure_end")

    alphas = np.zeros(num_frames, dtype=np.float64)
    if closure_end == ramp_start:
        alphas[closure_end:] = 1.0
        return alphas
    span = np.linspace(0.0, 1.0, closure_end - ramp_start + 1)
    alphas[ramp_start : closure_end + 1] = _ease_smoothstep(span)
    if closure_end + 1 < num_frames:
        alphas[closure_end + 1 :] = 1.0
    return alphas


def _interpolate_transform(
    transform: np.ndarray,
    alpha: float,
    translation_weight: float,
    rotation_weight: float,
) -> np.ndarray:
    alpha_t = float(np.clip(alpha * translation_weight, 0.0, 1.0))
    alpha_r = float(np.clip(alpha * rotation_weight, 0.0, 1.0))

    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = transform[:3, 3] * alpha_t
    rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    out[:3, :3] = Rotation.from_rotvec(rotvec * alpha_r).as_matrix()
    return out


def _clamp_transform(
    transform: np.ndarray,
    max_translation: float | None,
    max_rotation_deg: float | None,
) -> np.ndarray:
    out = transform.copy()
    if max_translation is not None and max_translation > 0:
        t_norm = float(np.linalg.norm(out[:3, 3]))
        if t_norm > max_translation:
            out[:3, 3] *= max_translation / t_norm

    if max_rotation_deg is not None and max_rotation_deg > 0:
        rotvec = Rotation.from_matrix(out[:3, :3]).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        max_angle = np.radians(max_rotation_deg)
        if angle > max_angle:
            out[:3, :3] = Rotation.from_rotvec(rotvec * (max_angle / angle)).as_matrix()
    return out


def _frame_range(center: int, radius: int, length: int, step: int) -> Iterable[int]:
    lo = max(0, center - radius)
    hi = min(length, center + radius + 1)
    return range(lo, hi, max(1, step))


def _local_depth_cloud(
    predictions: dict,
    center: int,
    extrinsic_convention: str,
    radius: int,
    frame_step: int,
    pixel_stride: int,
    conf_threshold: float,
    min_depth: float,
    max_depth: float,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if "depth" not in predictions:
        raise ValueError("ICP correction requires predictions['depth']")
    if "intrinsic" not in predictions or "extrinsic" not in predictions:
        raise ValueError("ICP correction requires intrinsic and extrinsic")

    depth = np.asarray(predictions["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    confidence = predictions.get("depth_conf")
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]

    intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float32)
    w2c = _extrinsic_for_unprojection(predictions["extrinsic"], extrinsic_convention)
    num_frames = depth.shape[0]
    center = _resolve_index(center, num_frames)

    sampled_grid = np.zeros(depth.shape[1:], dtype=bool)
    sampled_grid[:: max(1, pixel_stride), :: max(1, pixel_stride)] = True

    chunks = []
    for idx in _frame_range(center, radius, num_frames, frame_step):
        world_points, _, valid_depth = depth_to_world_coords_points(
            depth[idx],
            w2c[idx],
            intrinsics[idx],
        )
        mask = (
            valid_depth
            & sampled_grid
            & np.isfinite(world_points).all(axis=-1)
            & (depth[idx] >= min_depth)
            & (depth[idx] <= max_depth)
        )
        if confidence is not None:
            mask &= confidence[idx] > conf_threshold
        points = world_points[mask]
        if len(points) > 0:
            chunks.append(points.astype(np.float32, copy=False))

    if not chunks:
        raise ValueError(
            f"No points survived filtering around frame {center}; lower --conf_threshold "
            "or increase --icp_radius."
        )

    points = np.concatenate(chunks, axis=0)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
    return points


def estimate_icp_transform(
    predictions: dict,
    closure_start: int,
    closure_end: int,
    extrinsic_convention: str,
    radius: int = 12,
    frame_step: int = 2,
    pixel_stride: int = 8,
    conf_threshold: float = 1.8,
    min_depth: float = 0.15,
    max_depth: float = 8.0,
    max_local_points: int = 250_000,
    voxel_size: float = 0.06,
    icp_distances: tuple[float, ...] = (0.30, 0.15, 0.07),
    max_translation: float | None = 2.0,
    max_rotation_deg: float | None = 8.0,
    seed: int = 0,
) -> tuple[np.ndarray, IcpReport]:
    """Estimate a transform that maps the closure_end local cloud to closure_start."""
    import open3d as o3d

    target_np = _local_depth_cloud(
        predictions,
        closure_start,
        extrinsic_convention=extrinsic_convention,
        radius=radius,
        frame_step=frame_step,
        pixel_stride=pixel_stride,
        conf_threshold=conf_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
        max_points=max_local_points,
        seed=seed,
    )
    source_np = _local_depth_cloud(
        predictions,
        closure_end,
        extrinsic_convention=extrinsic_convention,
        radius=radius,
        frame_step=frame_step,
        pixel_stride=pixel_stride,
        conf_threshold=conf_threshold,
        min_depth=min_depth,
        max_depth=max_depth,
        max_points=max_local_points,
        seed=seed + 1,
    )

    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_np.astype(np.float64)))
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_np.astype(np.float64)))
    if voxel_size > 0:
        target = target.voxel_down_sample(voxel_size)
        source = source.voxel_down_sample(voxel_size)

    if len(target.points) < 100 or len(source.points) < 100:
        raise ValueError(
            f"Not enough ICP points after filtering: target={len(target.points)}, "
            f"source={len(source.points)}"
        )

    normal_radius = max(voxel_size * 3.0, 0.12)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    transform = np.eye(4, dtype=np.float64)
    result = None
    for distance in icp_distances:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            float(distance),
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )
        transform = result.transformation

    transform = _clamp_transform(transform, max_translation, max_rotation_deg)
    report = IcpReport(
        fitness=float(result.fitness if result is not None else 0.0),
        rmse=float(result.inlier_rmse if result is not None else np.inf),
        points_source=len(source.points),
        points_target=len(target.points),
        translation=float(np.linalg.norm(transform[:3, 3])),
        rotation_deg=_rotation_angle_deg(transform[:3, :3]),
    )
    return transform, report


def _image_from_predictions(predictions: dict, index: int) -> np.ndarray | None:
    paths = predictions.get("image_paths")
    if paths is not None:
        path = str(np.asarray(paths)[index])
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image

    images = predictions.get("images")
    if images is None:
        return None
    image = np.asarray(images[index])
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.ndim == 3:
        image = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return image


def find_visual_loop_closure(
    predictions: dict,
    min_separation: int = 90,
    sample_stride: int = 4,
    resize_width: int = 640,
    min_inliers: int = 25,
) -> tuple[int, int, dict]:
    """Find a repeated visual place using ORB + homography inliers."""
    num_frames = int(np.asarray(predictions["extrinsic"]).shape[0])
    indices = list(range(0, num_frames, max(1, sample_stride)))
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    items = []
    for idx in indices:
        image = _image_from_predictions(predictions, idx)
        if image is None:
            continue
        scale = resize_width / max(image.shape[1], 1)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (resize_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None and len(keypoints) >= 40:
            items.append((idx, keypoints, descriptors))

    best = None
    for left_pos, (i, kp1, des1) in enumerate(items):
        for j, kp2, des2 in items[left_pos + 1 :]:
            if j - i < min_separation:
                continue
            raw_matches = matcher.knnMatch(des1, des2, k=2)
            good = [
                match[0]
                for match in raw_matches
                if len(match) == 2 and match[0].distance < 0.72 * match[1].distance
            ]
            if len(good) < min_inliers:
                continue
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if inliers < min_inliers:
                continue
            score = (inliers, len(good), j - i)
            if best is None or score > best[0]:
                best = (score, i, j)

    if best is None:
        raise ValueError(
            "Could not find a visual loop closure. Pass --closure_start and "
            "--closure_end manually, or lower --auto_min_inliers."
        )
    score, start, end = best
    return start, end, {
        "visual_inliers": int(score[0]),
        "visual_matches": int(score[1]),
        "visual_separation": int(score[2]),
    }


def apply_loop_correction(
    predictions: dict,
    correction_transform: np.ndarray,
    closure_start: int,
    closure_end: int,
    extrinsic_convention: str,
    ramp_start: int | None = None,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    warp_world_points: bool = True,
) -> tuple[dict, dict]:
    if "extrinsic" not in predictions:
        raise ValueError("predictions must contain an 'extrinsic' array")

    extrinsic_convention = extrinsic_convention.lower()
    c2w = _poses_to_c2w(predictions["extrinsic"], extrinsic_convention)
    num_frames = c2w.shape[0]
    closure_start = _resolve_index(closure_start, num_frames)
    closure_end = _resolve_index(closure_end, num_frames)
    alphas = _correction_alphas(num_frames, closure_start, closure_end, ramp_start)

    correction_mats = np.stack(
        [
            _interpolate_transform(
                correction_transform,
                alpha,
                translation_weight=translation_weight,
                rotation_weight=rotation_weight,
            )
            for alpha in alphas
        ],
        axis=0,
    )
    c2w_corrected = np.einsum("sij,sjk->sik", correction_mats, c2w)

    corrected = {key: np.array(value) for key, value in predictions.items()}
    corrected["extrinsic"] = _c2w_to_extrinsic(c2w_corrected, extrinsic_convention)

    if warp_world_points and "world_points" in corrected:
        world_points = np.asarray(corrected["world_points"], dtype=np.float32)
        if world_points.shape[0] != num_frames:
            raise ValueError("world_points frame count does not match extrinsic")
        warped_frames = []
        for idx in range(num_frames):
            shape = world_points[idx].shape
            flat = world_points[idx].reshape(-1, 3).astype(np.float64)
            warped = flat @ correction_mats[idx, :3, :3].T + correction_mats[idx, :3, 3]
            warped_frames.append(warped.reshape(shape).astype(np.float32))
        corrected["world_points"] = np.stack(warped_frames, axis=0)

    before = c2w[closure_start] @ np.linalg.inv(c2w[closure_end])
    after = c2w_corrected[closure_start] @ np.linalg.inv(c2w_corrected[closure_end])
    corrected["loop_closure_correction"] = correction_mats.astype(np.float32)
    corrected["loop_closure_alphas"] = alphas.astype(np.float32)
    corrected["loop_closure_delta"] = correction_transform.astype(np.float32)

    report = {
        "frames": num_frames,
        "closure_start": closure_start,
        "closure_end": closure_end,
        "translation_pose_delta_before": float(np.linalg.norm(before[:3, 3])),
        "translation_pose_delta_after": float(np.linalg.norm(after[:3, 3])),
        "rotation_pose_delta_deg_before": _rotation_angle_deg(before[:3, :3]),
        "rotation_pose_delta_deg_after": _rotation_angle_deg(after[:3, :3]),
        "correction_translation": float(np.linalg.norm(correction_transform[:3, 3])),
        "correction_rotation_deg": _rotation_angle_deg(correction_transform[:3, :3]),
    }
    return corrected, report


def voxel_fuse_cloud(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return cloud
    cloud = np.asarray(cloud, dtype=np.float32)
    if len(cloud) == 0:
        return cloud
    voxels = np.floor(cloud[:, :3] / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    fused = np.zeros((len(counts), 6), dtype=np.float64)
    np.add.at(fused, inverse, cloud)
    fused /= counts[:, None]
    return fused.astype(np.float32)


def load_predictions(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_predictions(path: str | Path, predictions: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **predictions)


def _parse_distances(value: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated distances")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply visual/ICP loop-drift correction to demo.py predictions."
    )
    parser.add_argument("--input", required=True, help="Raw predictions .npz saved by demo.py")
    parser.add_argument("--output", default=None, help="Corrected predictions .npz")
    parser.add_argument("--save_pointcloud", "--save_pcl", dest="save_pointcloud", default=None,
                        help="Optional corrected map path: .pcd, .ply, or .npz")
    parser.add_argument("--closure_start", type=int, default=None,
                        help="Frame index on the already-built side of the loop")
    parser.add_argument("--closure_end", type=int, default=None,
                        help="Later frame/index on the drifted side of the loop")
    parser.add_argument("--auto_closure", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-detect closure_start/end with ORB when either index is omitted.")
    parser.add_argument("--auto_min_separation", type=int, default=90)
    parser.add_argument("--auto_sample_stride", type=int, default=4)
    parser.add_argument("--auto_min_inliers", type=int, default=25)
    parser.add_argument("--extrinsic_convention", choices=["w2c", "c2w"], default="w2c",
                        help="Convention stored in --input. demo.py/viewer export uses w2c.")
    parser.add_argument("--method", choices=["icp", "pose_translation"], default="icp",
                        help="icp aligns local geometry; pose_translation only closes position drift.")
    parser.add_argument("--ramp_start", type=int, default=None,
                        help="Frame where correction starts ramping. Defaults to closure_start.")
    parser.add_argument("--translation_weight", type=float, default=1.0,
                        help="Scale applied correction translation. Use 0.5-0.8 if the map bends too much.")
    parser.add_argument("--rotation_weight", type=float, default=1.0,
                        help="Scale applied correction rotation. ICP usually estimates only a small rotation.")
    parser.add_argument("--max_translation", type=float, default=2.0,
                        help="Clamp estimated correction translation. Use 0 to disable.")
    parser.add_argument("--max_rotation_deg", type=float, default=8.0,
                        help="Clamp estimated correction rotation. Use 0 to disable.")
    parser.add_argument("--min_icp_fitness", type=float, default=0.25,
                        help="Abort if ICP fitness is below this value.")
    parser.add_argument("--icp_radius", type=int, default=12,
                        help="Frames on each side used to build local ICP clouds.")
    parser.add_argument("--icp_frame_step", type=int, default=2)
    parser.add_argument("--icp_pixel_stride", type=int, default=8)
    parser.add_argument("--icp_voxel_size", type=float, default=0.06)
    parser.add_argument("--icp_distances", type=_parse_distances, default=(0.30, 0.15, 0.07),
                        help="Comma-separated ICP distance schedule, e.g. 0.30,0.15,0.07")
    parser.add_argument("--conf_threshold", type=float, default=1.8,
                        help="Confidence cutoff for ICP and point cloud export.")
    parser.add_argument("--min_depth", type=float, default=0.15)
    parser.add_argument("--max_depth", type=float, default=8.0)
    parser.add_argument("--max_local_points", type=int, default=250_000)
    parser.add_argument("--no_warp_world_points", action="store_true",
                        help="Only update camera poses; leave existing world_points unchanged.")
    parser.add_argument("--pointcloud_source", choices=["depth", "pointmap"], default="depth",
                        help="Use corrected depth+poses or corrected world_points for export.")
    parser.add_argument("--downsample_factor", type=int, default=10,
                        help="Keep every N-th valid point per frame when exporting.")
    parser.add_argument("--max_points", type=int, default=6_000_000,
                        help="Random cap for exported points. Use 0 for no cap.")
    parser.add_argument("--fuse_voxel_size", type=float, default=0.0,
                        help="Voxel-average exported xyz/rgb to reduce residual ghosting.")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Apply sky segmentation before point cloud export.")
    parser.add_argument("--image_folder", default=None,
                        help="Image folder for sky-mask caching if image_paths are unavailable.")
    parser.add_argument("--sky_mask_dir", default=None)
    parser.add_argument("--sky_mask_visualization_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None and args.save_pointcloud is None:
        raise SystemExit("Provide --output, --save_pointcloud, or both.")

    predictions = load_predictions(args.input)
    num_frames = int(np.asarray(predictions["extrinsic"]).shape[0])

    visual_report = {}
    if args.closure_start is None or args.closure_end is None:
        if not args.auto_closure:
            raise SystemExit("Pass both --closure_start/--closure_end or enable --auto_closure.")
        closure_start, closure_end, visual_report = find_visual_loop_closure(
            predictions,
            min_separation=args.auto_min_separation,
            sample_stride=args.auto_sample_stride,
            min_inliers=args.auto_min_inliers,
        )
        print(
            f"Auto closure: {closure_start} -> {closure_end} "
            f"({visual_report['visual_inliers']} homography inliers)"
        )
    else:
        closure_start = _resolve_index(args.closure_start, num_frames)
        closure_end = _resolve_index(args.closure_end, num_frames)

    if args.method == "icp":
        transform, icp_report = estimate_icp_transform(
            predictions,
            closure_start=closure_start,
            closure_end=closure_end,
            extrinsic_convention=args.extrinsic_convention,
            radius=args.icp_radius,
            frame_step=args.icp_frame_step,
            pixel_stride=args.icp_pixel_stride,
            conf_threshold=args.conf_threshold,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_local_points=args.max_local_points,
            voxel_size=args.icp_voxel_size,
            icp_distances=args.icp_distances,
            max_translation=args.max_translation if args.max_translation > 0 else None,
            max_rotation_deg=args.max_rotation_deg if args.max_rotation_deg > 0 else None,
        )
        print(
            "ICP correction: "
            f"fitness={icp_report.fitness:.3f}, rmse={icp_report.rmse:.4f}, "
            f"translation={icp_report.translation:.3f}m, rotation={icp_report.rotation_deg:.2f}deg, "
            f"points={icp_report.points_source}/{icp_report.points_target}"
        )
        if icp_report.fitness < args.min_icp_fitness:
            raise SystemExit(
                f"ICP fitness {icp_report.fitness:.3f} is below --min_icp_fitness "
                f"{args.min_icp_fitness:.3f}; choose a better closure pair."
            )
    else:
        c2w = _poses_to_c2w(predictions["extrinsic"], args.extrinsic_convention)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = c2w[closure_start, :3, 3] - c2w[closure_end, :3, 3]
        print(f"Pose-translation correction: {np.linalg.norm(transform[:3, 3]):.3f}m")

    corrected, report = apply_loop_correction(
        predictions,
        transform,
        closure_start=closure_start,
        closure_end=closure_end,
        extrinsic_convention=args.extrinsic_convention,
        ramp_start=args.ramp_start,
        translation_weight=args.translation_weight,
        rotation_weight=args.rotation_weight,
        warp_world_points=not args.no_warp_world_points,
    )
    report.update(visual_report)

    print(
        "Visual-loop camera delta kept free: "
        f"translation {report['translation_pose_delta_before']:.4f} -> "
        f"{report['translation_pose_delta_after']:.4f}, "
        f"rotation {report['rotation_pose_delta_deg_before']:.2f}deg -> "
        f"{report['rotation_pose_delta_deg_after']:.2f}deg "
        "(not forced to zero)"
    )

    if args.output is not None:
        save_predictions(args.output, corrected)
        print(f"Saved corrected predictions: {args.output}")

    if args.save_pointcloud is not None:
        export_predictions = corrected
        if args.extrinsic_convention == "c2w" and args.pointcloud_source == "depth":
            c2w = _as_4x4(corrected["extrinsic"])
            export_predictions = dict(corrected)
            export_predictions["extrinsic"] = np.linalg.inv(c2w)[:, :3, :4].astype(np.float32)

        cloud = predictions_to_pointcloud(
            export_predictions,
            conf_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            source=args.pointcloud_source,
            mask_sky=args.mask_sky,
            image_folder=args.image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
            max_points=args.max_points,
        )
        before_fuse = len(cloud)
        cloud = voxel_fuse_cloud(cloud, args.fuse_voxel_size)
        save_pointcloud(cloud, args.save_pointcloud)
        fuse_note = f", fused {before_fuse:,}->{len(cloud):,}" if args.fuse_voxel_size > 0 else ""
        print(f"Saved corrected point cloud: {args.save_pointcloud} ({len(cloud):,} points{fuse_note})")


if __name__ == "__main__":
    main()