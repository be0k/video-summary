#!/usr/bin/env python3
"""Run LingBot-MAP inference without launching the interactive Viser viewer."""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

MERGING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MERGING_ROOT.parent
LINGBOT_ROOT = PROJECT_ROOT / "lingbot-map"
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

import demo as lingbot_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LingBot-MAP and save raw predictions as NPZ.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--start_seconds", type=float, default=0.0)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument(
        "--mode",
        choices=["streaming", "windowed"],
        default="streaming",
        help="Accepted for compatibility; the web pipeline forces streaming.",
    )
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=None)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--overlap_size", type=int, default=16)
    parser.add_argument("--overlap_keyframes", type=int, default=None)
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no_offload_to_cpu", action="store_true", default=False)
    return parser.parse_args()


def demo_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        image_folder=None,
        video_path=str(args.video_path),
        fps=args.fps,
        first_k=args.first_k,
        stride=args.stride,
        rotate_clockwise_90=False,
        model_path=str(args.model_path),
        image_size=args.image_size,
        patch_size=args.patch_size,
        mode=args.mode,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=args.camera_num_iterations,
        use_sdpa=args.use_sdpa,
        compile=args.compile,
        offload_to_cpu=not args.no_offload_to_cpu,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        overlap_keyframes=args.overlap_keyframes,
    )


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def extract_frames_from_original(video_path: Path, fps: int, start_seconds: float) -> Path:
    out_dir = video_path.with_name(f"{video_path.stem}_frames_start_{int(start_seconds * 1000):09d}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / fps))
    start_frame = max(0, int(round(start_seconds * src_fps)))

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= start_frame and (idx - start_frame) % interval == 0:
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    if saved <= 0:
        raise RuntimeError(
            f"Extracted 0 frames from {video_path} "
            f"(total_frames={total_frames}, start_seconds={start_seconds}, interval={interval})"
        )
    print(
        f"Extracted {saved} LingBot frames from original video "
        f"(start={start_seconds:.3f}s, total={total_frames}, interval={interval})"
    )
    return out_dir


def main() -> None:
    cli_args = parse_args()
    args = demo_namespace(cli_args)
    if args.mode != "streaming":
        print(f"Windowed mode is disabled in this pipeline; forcing streaming (requested {args.mode}).")
        args.mode = "streaming"
    cli_args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    if cli_args.start_seconds > 1e-3:
        source_frame_dir = extract_frames_from_original(cli_args.video_path, args.fps, cli_args.start_seconds)
        images, paths, _ = lingbot_demo.load_images(
            image_folder=str(source_frame_dir),
            video_path=None,
            fps=args.fps,
            first_k=args.first_k,
            stride=args.stride,
            image_size=args.image_size,
            patch_size=args.patch_size,
            rotate_clockwise_90=False,
        )
    else:
        source_frame_dir = None
        images, paths, _ = lingbot_demo.load_images(
            image_folder=None,
            video_path=str(cli_args.video_path),
            fps=args.fps,
            first_k=args.first_k,
            stride=args.stride,
            image_size=args.image_size,
            patch_size=args.patch_size,
            rotate_clockwise_90=False,
        )
    num_frames = int(images.shape[0])

    model = lingbot_demo.load_model(args, device)
    print(f"Loaded frames/model in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype}")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    print(f"Input: {num_frames} frames, shape={tuple(images.shape)}, mode={args.mode}")

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
        else:
            args.keyframe_interval = 1
    print(f"Keyframe interval: {args.keyframe_interval}")

    if args.compile and args.mode == "streaming" and device.type == "cuda":
        scale_for_warm = min(args.num_scale_frames, num_frames)
        if scale_for_warm >= num_frames:
            scale_for_warm = max(1, num_frames - 1)
        warm_stream_n = min(10, max(1, num_frames - scale_for_warm))
        lingbot_demo._warm_streaming(
            model,
            images,
            scale_for_warm,
            warm_stream_n,
            dtype,
            passes=1,
            keyframe_interval=args.keyframe_interval,
        )
        lingbot_demo.compile_model(model)
        lingbot_demo._warm_streaming(
            model,
            images,
            scale_for_warm,
            warm_stream_n,
            dtype,
            passes=3,
            keyframe_interval=args.keyframe_interval,
        )

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    print(f"Running inference with dtype={dtype}, offload_to_cpu={args.offload_to_cpu}")
    t0 = time.time()
    with torch.no_grad(), autocast_context(device, dtype):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=output_device,
        )
    print(f"Inference done in {time.time() - t0:.1f}s")

    if args.offload_to_cpu and "images" in predictions:
        images_for_post = predictions["images"]
    else:
        images_for_post = images

    predictions, images_cpu = lingbot_demo.postprocess(predictions, images_for_post)
    vis_predictions = lingbot_demo.prepare_for_visualization(predictions, images_cpu)
    vis_predictions["image_paths"] = np.asarray(paths, dtype=str)
    lingbot_demo.save_predictions_npz(str(cli_args.output), vis_predictions)
    print(f"Saved predictions NPZ: {cli_args.output}")


if __name__ == "__main__":
    main()
