#!/usr/bin/env python3
"""Generate and compare multiple post-processing variants for LingBot-MAP outputs.

The goal is to make post-processing empirical: create several plausible maps
from the same ``demo.py --save_predictions`` NPZ, export each result, and write
a small report with overlap metrics around detected loop-closure candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lingbot_map.utils.geometry import depth_to_world_coords_points
from lingbot_map.utils.pointcloud_export import save_pointcloud
from loop_closure_drift_fix import (
    _as_4x4,
    _c2w_to_extrinsic,
    _correction_alphas,
    _extrinsic_for_unprojection,
    _image_from_predictions,
    _interpolate_transform,
    _local_depth_cloud,
    _poses_to_c2w,
    _rotation_angle_deg,
    estimate_icp_transform,
    voxel_fuse_cloud,
)


@dataclass
class LoopCandidate:
    start: int
    end: int
    inliers: int
    matches: int
    separation: int


@dataclass
class Variant:
    name: str
    description: str
    correction: str
    conf_threshold: float
    min_depth: float = 0.15
    max_depth: float = 8.0
    downsample_factor: int = 10
    frame_stride: int = 1
    max_points: int = 3_000_000
    fuse_voxel_size: float = 0.0
    sor_neighbors: int = 0
    sor_std_ratio: float = 2.0
    translation_weight: float = 1.0
    rotation_weight: float = 1.0
    max_translation: float = 2.0
    max_rotation_deg: float = 8.0
    icp_radius: int = 12
    max_loops: int = 1
    tsdf: bool = False
    tsdf_frame_stride: int = 3
    tsdf_voxel_length: float = 0.035
    tsdf_sdf_trunc: float = 0.12


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _images_to_nhwc(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"images must be 4D, got {images.shape}")
    if images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return np.clip(images.astype(np.float32), 0.0, 1.0)


def find_visual_loop_candidates(
    predictions: dict[str, np.ndarray],
    min_separation: int,
    sample_stride: int,
    min_inliers: int,
    max_candidates: int,
    nms_radius: int,
    resize_width: int = 640,
) -> list[LoopCandidate]:
    """Return visual loop candidates sorted by homography inlier count."""
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

    raw: list[LoopCandidate] = []
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
            if inliers >= min_inliers:
                raw.append(LoopCandidate(i, j, inliers, len(good), j - i))

    raw.sort(key=lambda c: (c.inliers, c.matches, c.separation), reverse=True)
    selected: list[LoopCandidate] = []
    for cand in raw:
        if any(abs(cand.start - old.start) <= nms_radius and abs(cand.end - old.end) <= nms_radius for old in selected):
            continue
        selected.append(cand)
        if len(selected) >= max_candidates:
            break
    return selected


def apply_correction_light(
    predictions: dict[str, np.ndarray],
    transform: np.ndarray,
    start: int,
    end: int,
    extrinsic_convention: str,
    translation_weight: float,
    rotation_weight: float,
    ramp_start: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Apply a smooth transform to poses without copying heavy arrays."""
    c2w = _poses_to_c2w(predictions["extrinsic"], extrinsic_convention)
    alphas = _correction_alphas(c2w.shape[0], start, end, ramp_start)
    correction_mats = np.stack(
        [
            _interpolate_transform(
                transform,
                alpha,
                translation_weight=translation_weight,
                rotation_weight=rotation_weight,
            )
            for alpha in alphas
        ],
        axis=0,
    )
    corrected_c2w = np.einsum("sij,sjk->sik", correction_mats, c2w)
    out = dict(predictions)
    out["extrinsic"] = _c2w_to_extrinsic(corrected_c2w, extrinsic_convention)

    before = c2w[start] @ np.linalg.inv(c2w[end])
    after = corrected_c2w[start] @ np.linalg.inv(corrected_c2w[end])
    return out, {
        "pose_delta_t_before": float(np.linalg.norm(before[:3, 3])),
        "pose_delta_t_after": float(np.linalg.norm(after[:3, 3])),
        "pose_delta_r_before": _rotation_angle_deg(before[:3, :3]),
        "pose_delta_r_after": _rotation_angle_deg(after[:3, :3]),
        "applied_t": float(np.linalg.norm(transform[:3, 3])),
        "applied_r": _rotation_angle_deg(transform[:3, :3]),
    }


def corrected_predictions_for_variant(
    base: dict[str, np.ndarray],
    variant: Variant,
    candidates: list[LoopCandidate],
    extrinsic_convention: str,
    min_icp_fitness: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    report: dict[str, Any] = {"correction": variant.correction, "loops_used": 0}
    if variant.correction == "none":
        return base, report

    if not candidates:
        report["fallback_reason"] = "no visual loop candidates; exported without pose correction"
        return base, report

    current = dict(base)
    used = []
    for cand in candidates[: max(1, variant.max_loops)]:
        if variant.correction in {"single_icp", "multi_icp"}:
            try:
                transform, icp_report = estimate_icp_transform(
                    current,
                    closure_start=cand.start,
                    closure_end=cand.end,
                    extrinsic_convention=extrinsic_convention,
                    radius=variant.icp_radius,
                    frame_step=2,
                    pixel_stride=8,
                    conf_threshold=max(variant.conf_threshold, 1.8),
                    min_depth=variant.min_depth,
                    max_depth=variant.max_depth,
                    max_local_points=250_000,
                    voxel_size=0.06,
                    icp_distances=(0.30, 0.15, 0.07),
                    max_translation=variant.max_translation,
                    max_rotation_deg=variant.max_rotation_deg,
                )
            except Exception as exc:  # noqa: BLE001 - report and continue with other variants
                used.append({"start": cand.start, "end": cand.end, "error": str(exc)})
                continue
            if icp_report.fitness < min_icp_fitness:
                used.append({
                    "start": cand.start,
                    "end": cand.end,
                    "skipped": f"fitness {icp_report.fitness:.3f} < {min_icp_fitness:.3f}",
                })
                continue
            current, apply_report = apply_correction_light(
                current,
                transform,
                cand.start,
                cand.end,
                extrinsic_convention=extrinsic_convention,
                translation_weight=variant.translation_weight,
                rotation_weight=variant.rotation_weight,
            )
            used.append({
                "start": cand.start,
                "end": cand.end,
                "inliers": cand.inliers,
                "icp": asdict(icp_report),
                "apply": apply_report,
            })
        elif variant.correction == "translation_only":
            c2w = _poses_to_c2w(current["extrinsic"], extrinsic_convention)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = c2w[cand.start, :3, 3] - c2w[cand.end, :3, 3]
            current, apply_report = apply_correction_light(
                current,
                transform,
                cand.start,
                cand.end,
                extrinsic_convention=extrinsic_convention,
                translation_weight=variant.translation_weight,
                rotation_weight=0.0,
            )
            used.append({
                "start": cand.start,
                "end": cand.end,
                "inliers": cand.inliers,
                "apply": apply_report,
            })
        else:
            raise ValueError(f"Unknown correction mode: {variant.correction}")

        if variant.correction == "single_icp":
            break

    report["loops_used"] = sum("apply" in item for item in used)
    report["loop_reports"] = used
    return current, report


def export_depth_cloud(
    predictions: dict[str, np.ndarray],
    output_path: Path,
    variant: Variant,
    extrinsic_convention: str,
    seed: int,
) -> dict[str, Any]:
    depth = np.asarray(predictions["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = np.asarray(predictions["depth_conf"], dtype=np.float32)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    images = _images_to_nhwc(predictions["images"])
    intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float32)
    w2c = _extrinsic_for_unprojection(predictions["extrinsic"], extrinsic_convention)

    pixel_stride = max(1, int(variant.downsample_factor))
    sample_grid = np.zeros(depth.shape[1:], dtype=bool)
    sample_grid[::pixel_stride, ::pixel_stride] = True

    all_points = []
    all_colors = []
    selected_frames = range(0, depth.shape[0], max(1, int(variant.frame_stride)))
    for idx in selected_frames:
        world_points, _, valid_depth = depth_to_world_coords_points(
            depth[idx],
            w2c[idx],
            intrinsics[idx],
        )
        mask = (
            valid_depth
            & sample_grid
            & np.isfinite(world_points).all(axis=-1)
            & (depth[idx] >= variant.min_depth)
            & (depth[idx] <= variant.max_depth)
            & (conf[idx] > variant.conf_threshold)
        )
        points = world_points[mask]
        if len(points) == 0:
            continue
        all_points.append(points.astype(np.float32, copy=False))
        all_colors.append(images[idx][mask].astype(np.float32, copy=False))

    if not all_points:
        raise ValueError(f"{variant.name}: no valid points survived filtering")

    cloud = np.concatenate([np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)], axis=1)
    before_cap = len(cloud)
    if variant.max_points > 0 and len(cloud) > variant.max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(cloud), size=variant.max_points, replace=False)
        keep.sort()
        cloud = cloud[keep]
    after_cap = len(cloud)

    if variant.fuse_voxel_size > 0:
        cloud = voxel_fuse_cloud(cloud, variant.fuse_voxel_size)

    if variant.sor_neighbors > 0 and len(cloud) > variant.sor_neighbors:
        cloud = statistical_outlier_filter(cloud, variant.sor_neighbors, variant.sor_std_ratio)

    save_pointcloud(cloud, output_path)
    return cloud_stats(cloud) | {
        "points_before_cap": int(before_cap),
        "points_after_cap": int(after_cap),
        "points_written": int(len(cloud)),
    }


def statistical_outlier_filter(cloud: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(cloud[:, :3].astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(np.clip(cloud[:, 3:6], 0.0, 1.0).astype(np.float64))
    filtered, indices = point_cloud.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    keep = np.asarray(indices, dtype=np.int64)
    if len(keep) == 0:
        return cloud
    return cloud[keep]


def export_tsdf_cloud(
    predictions: dict[str, np.ndarray],
    output_path: Path,
    variant: Variant,
    extrinsic_convention: str,
    seed: int,
) -> dict[str, Any]:
    import open3d as o3d

    depth = np.asarray(predictions["depth"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = np.asarray(predictions["depth_conf"], dtype=np.float32)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    images = _images_to_nhwc(predictions["images"])
    intrinsics = np.asarray(predictions["intrinsic"], dtype=np.float32)
    w2c_4x4 = _as_4x4(_extrinsic_for_unprojection(predictions["extrinsic"], extrinsic_convention))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(variant.tsdf_voxel_length),
        sdf_trunc=float(variant.tsdf_sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    for idx in range(0, depth.shape[0], max(1, int(variant.tsdf_frame_stride))):
        depth_i = depth[idx].copy()
        valid = (
            np.isfinite(depth_i)
            & (depth_i >= variant.min_depth)
            & (depth_i <= variant.max_depth)
            & (conf[idx] > variant.conf_threshold)
        )
        depth_i[~valid] = 0.0
        if np.count_nonzero(valid) < 100:
            continue

        color_u8 = np.ascontiguousarray(
            np.clip(images[idx], 0.0, 1.0) * 255.0,
            dtype=np.uint8,
        )
        depth_o3d = np.ascontiguousarray(depth_i, dtype=np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_u8),
            o3d.geometry.Image(depth_o3d),
            depth_scale=1.0,
            depth_trunc=float(variant.max_depth),
            convert_rgb_to_intensity=False,
        )
        h, w = depth_i.shape
        k = intrinsics[idx]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(w),
            int(h),
            float(k[0, 0]),
            float(k[1, 1]),
            float(k[0, 2]),
            float(k[1, 2]),
        )
        volume.integrate(rgbd, intrinsic, w2c_4x4[idx])
        integrated += 1

    point_cloud = volume.extract_point_cloud()
    points = np.asarray(point_cloud.points, dtype=np.float32)
    colors = np.asarray(point_cloud.colors, dtype=np.float32)
    if len(points) == 0:
        raise ValueError(f"{variant.name}: TSDF produced no points")
    cloud = np.concatenate([points, np.clip(colors, 0.0, 1.0)], axis=1).astype(np.float32)
    if variant.max_points > 0 and len(cloud) > variant.max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(cloud), size=variant.max_points, replace=False)
        keep.sort()
        cloud = cloud[keep]
    if variant.sor_neighbors > 0 and len(cloud) > variant.sor_neighbors:
        cloud = statistical_outlier_filter(cloud, variant.sor_neighbors, variant.sor_std_ratio)
    save_pointcloud(cloud, output_path)
    return cloud_stats(cloud) | {
        "tsdf_integrated_frames": integrated,
        "points_written": int(len(cloud)),
    }


def cloud_stats(cloud: np.ndarray) -> dict[str, Any]:
    points = cloud[:, :3]
    bbox = np.ptp(points, axis=0) if len(points) else np.zeros(3)
    volume = float(np.prod(np.maximum(bbox, 1e-6)))
    return {
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_volume": volume,
        "points_per_m3": float(len(points) / volume) if volume > 0 else 0.0,
    }


def closure_overlap_metrics(
    predictions: dict[str, np.ndarray],
    candidates: list[LoopCandidate],
    extrinsic_convention: str,
    max_pairs: int,
) -> dict[str, Any]:
    import open3d as o3d

    medians = []
    means = []
    p90s = []
    pair_reports = []
    for cand in candidates[:max_pairs]:
        try:
            target = _local_depth_cloud(
                predictions,
                cand.start,
                extrinsic_convention=extrinsic_convention,
                radius=12,
                frame_step=2,
                pixel_stride=8,
                conf_threshold=1.8,
                min_depth=0.15,
                max_depth=8.0,
                max_points=80_000,
                seed=10,
            )
            source = _local_depth_cloud(
                predictions,
                cand.end,
                extrinsic_convention=extrinsic_convention,
                radius=12,
                frame_step=2,
                pixel_stride=8,
                conf_threshold=1.8,
                min_depth=0.15,
                max_depth=8.0,
                max_points=80_000,
                seed=11,
            )
            target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target.astype(np.float64)))
            source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source.astype(np.float64)))
            target_pcd = target_pcd.voxel_down_sample(0.06)
            source_pcd = source_pcd.voxel_down_sample(0.06)
            distances = np.asarray(source_pcd.compute_point_cloud_distance(target_pcd), dtype=np.float64)
            if len(distances) == 0:
                continue
            mean = float(np.mean(distances))
            median = float(np.median(distances))
            p90 = float(np.percentile(distances, 90))
            medians.append(median)
            means.append(mean)
            p90s.append(p90)
            pair_reports.append({
                "start": cand.start,
                "end": cand.end,
                "mean": mean,
                "median": median,
                "p90": p90,
            })
        except Exception as exc:  # noqa: BLE001 - keep comparing other variants
            pair_reports.append({"start": cand.start, "end": cand.end, "error": str(exc)})

    return {
        "closure_mean_avg": float(np.mean(means)) if means else np.nan,
        "closure_median_avg": float(np.mean(medians)) if medians else np.nan,
        "closure_p90_avg": float(np.mean(p90s)) if p90s else np.nan,
        "closure_pairs": pair_reports,
    }


def default_variants(max_points: int, include_tsdf: bool) -> list[Variant]:
    variants = [
        Variant(
            name="raw_default",
            description="Default depth export, useful as a direct baseline.",
            correction="none",
            conf_threshold=1.5,
            max_depth=10.0,
            downsample_factor=10,
            max_points=max_points,
        ),
        Variant(
            name="strict_fuse",
            description="Higher confidence, shorter depth range, voxel fusion, SOR cleanup.",
            correction="none",
            conf_threshold=2.2,
            max_depth=6.0,
            downsample_factor=6,
            max_points=max_points,
            fuse_voxel_size=0.025,
            sor_neighbors=20,
            sor_std_ratio=1.8,
        ),
        Variant(
            name="frame_thin_fuse",
            description="Use every second frame to reduce duplicated surfaces, then fuse.",
            correction="none",
            conf_threshold=1.9,
            max_depth=7.0,
            downsample_factor=6,
            frame_stride=2,
            max_points=max_points,
            fuse_voxel_size=0.03,
            sor_neighbors=20,
            sor_std_ratio=1.8,
        ),
        Variant(
            name="single_icp_full",
            description="Best visual loop + local ICP correction, full correction weight.",
            correction="single_icp",
            conf_threshold=1.8,
            max_depth=8.0,
            downsample_factor=10,
            max_points=max_points,
            fuse_voxel_size=0.02,
            translation_weight=1.0,
            rotation_weight=1.0,
        ),
        Variant(
            name="single_icp_damped",
            description="Best visual loop + local ICP, damped correction to avoid bending the map.",
            correction="single_icp",
            conf_threshold=2.0,
            max_depth=7.0,
            downsample_factor=8,
            max_points=max_points,
            fuse_voxel_size=0.025,
            translation_weight=0.65,
            rotation_weight=0.35,
            max_rotation_deg=4.0,
        ),
        Variant(
            name="translation_only_damped",
            description="Visual loop position correction only; no rotation closure.",
            correction="translation_only",
            conf_threshold=2.0,
            max_depth=7.0,
            downsample_factor=8,
            max_points=max_points,
            fuse_voxel_size=0.025,
            translation_weight=0.45,
            rotation_weight=0.0,
        ),
        Variant(
            name="multi_icp_damped",
            description="Top non-overlapping visual loops, each with damped local ICP correction.",
            correction="multi_icp",
            conf_threshold=2.0,
            max_depth=7.0,
            downsample_factor=8,
            max_points=max_points,
            fuse_voxel_size=0.025,
            translation_weight=0.45,
            rotation_weight=0.25,
            max_rotation_deg=3.0,
            max_loops=3,
        ),
    ]
    if include_tsdf:
        variants.extend([
            Variant(
                name="tsdf_strict",
                description="TSDF fusion without pose correction; usually reduces noisy duplicate points.",
                correction="none",
                conf_threshold=2.1,
                max_depth=6.0,
                downsample_factor=1,
                max_points=max_points,
                sor_neighbors=20,
                sor_std_ratio=1.8,
                tsdf=True,
                tsdf_frame_stride=3,
                tsdf_voxel_length=0.035,
                tsdf_sdf_trunc=0.12,
            ),
            Variant(
                name="tsdf_single_icp",
                description="Best visual loop ICP correction followed by TSDF fusion.",
                correction="single_icp",
                conf_threshold=2.0,
                max_depth=6.5,
                downsample_factor=1,
                max_points=max_points,
                translation_weight=0.65,
                rotation_weight=0.35,
                max_rotation_deg=4.0,
                sor_neighbors=20,
                sor_std_ratio=1.8,
                tsdf=True,
                tsdf_frame_stride=3,
                tsdf_voxel_length=0.035,
                tsdf_sdf_trunc=0.12,
            ),
        ])
    return variants


def write_reports(out_dir: Path, rows: list[dict[str, Any]], candidates: list[LoopCandidate]) -> None:
    serializable = {
        "candidates": [asdict(c) for c in candidates],
        "variants": rows,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list, tuple))
        }
        flat_rows.append(flat)
    keys = sorted({key for row in flat_rows for key in row})
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate several post-processed maps and compare loop-overlap metrics."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw demo.py predictions .npz")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/postprocess_compare"))
    parser.add_argument("--extrinsic_convention", choices=["w2c", "c2w"], default="w2c")
    parser.add_argument("--max_points", type=int, default=3_000_000)
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names, or 'all'. Use --list_variants to inspect.")
    parser.add_argument("--include_tsdf", action="store_true",
                        help="Also run TSDF fusion variants. Slower, but often cleaner.")
    parser.add_argument("--list_variants", action="store_true")
    parser.add_argument("--closure_min_separation", type=int, default=90)
    parser.add_argument("--closure_sample_stride", type=int, default=4)
    parser.add_argument("--closure_min_inliers", type=int, default=25)
    parser.add_argument("--closure_candidates", type=int, default=8)
    parser.add_argument("--closure_nms_radius", type=int, default=24)
    parser.add_argument("--min_icp_fitness", type=float, default=0.25)
    parser.add_argument("--metric_pairs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = default_variants(args.max_points, args.include_tsdf)
    if args.list_variants:
        for variant in variants:
            print(f"{variant.name}: {variant.description}")
        return

    selected_names = {name.strip() for name in args.variants.split(",") if name.strip()}
    if selected_names != {"all"}:
        variants = [variant for variant in variants if variant.name in selected_names]
        missing = selected_names - {variant.name for variant in variants}
        if missing:
            raise SystemExit(f"Unknown variants: {', '.join(sorted(missing))}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading predictions: {args.input}")
    predictions = load_predictions(args.input)

    print("Finding visual loop candidates...")
    candidates = find_visual_loop_candidates(
        predictions,
        min_separation=args.closure_min_separation,
        sample_stride=args.closure_sample_stride,
        min_inliers=args.closure_min_inliers,
        max_candidates=args.closure_candidates,
        nms_radius=args.closure_nms_radius,
    )
    if not candidates:
        print(
            "No visual loop candidates found. "
            "Continuing with no-closure fallback exports."
        )
    else:
        print("Candidates:")
        for cand in candidates:
            print(
                f"  {cand.start:04d}->{cand.end:04d} "
                f"inliers={cand.inliers} matches={cand.matches} separation={cand.separation}"
            )

    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        print(f"\n[{idx + 1}/{len(variants)}] {variant.name}: {variant.description}")
        t0 = time.time()
        row: dict[str, Any] = asdict(variant)
        row["status"] = "ok"
        output_path = args.out_dir / f"{variant.name}.pcd"
        try:
            corrected, correction_report = corrected_predictions_for_variant(
                predictions,
                variant,
                candidates,
                extrinsic_convention=args.extrinsic_convention,
                min_icp_fitness=args.min_icp_fitness,
            )
            if variant.tsdf:
                export_report = export_tsdf_cloud(
                    corrected,
                    output_path,
                    variant,
                    extrinsic_convention=args.extrinsic_convention,
                    seed=idx,
                )
            else:
                export_report = export_depth_cloud(
                    corrected,
                    output_path,
                    variant,
                    extrinsic_convention=args.extrinsic_convention,
                    seed=idx,
                )
            metric_report = closure_overlap_metrics(
                corrected,
                candidates,
                extrinsic_convention=args.extrinsic_convention,
                max_pairs=args.metric_pairs,
            )
            row.update(correction_report)
            row.update(export_report)
            row.update(metric_report)
            row["output"] = str(output_path)
        except Exception as exc:  # noqa: BLE001 - keep generating other variants
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"  ERROR: {exc}")
        row["elapsed_sec"] = round(time.time() - t0, 3)
        rows.append(row)
        print(
            f"  status={row['status']} points={row.get('points_written')} "
            f"closure_median={row.get('closure_median_avg')} "
            f"elapsed={row['elapsed_sec']}s"
        )

    write_reports(args.out_dir, rows, candidates)
    print(f"\nWrote: {args.out_dir / 'report.csv'}")
    print(f"Wrote: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
