#!/usr/bin/env python3
"""Dependency-light video summary fallback for the local web app.

This is not TriMamba. It exists so the website can still show a compact summary
video when the TriMamba encoder stack is unavailable in the local environment.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def video_duration(path: Path) -> tuple[float, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    cap.release()
    if duration <= 0:
        raise RuntimeError(f"Could not determine duration: {path}")
    return duration, fps, frames


def frame_at(cap: cv2.VideoCapture, second: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def score_video(path: Path, sample_fps: float, window_seconds: float) -> tuple[float, list[tuple[float, float, float]]]:
    duration, _, _ = video_duration(path)
    cap = cv2.VideoCapture(str(path))
    step = 1.0 / sample_fps
    times = np.arange(0.0, duration, step, dtype=np.float32)
    prev_gray = None
    prev_hist = None
    samples: list[tuple[float, float]] = []

    for second in times:
        frame = frame_at(cap, float(second))
        if frame is None:
            continue
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        contrast = float(gray.std()) / 64.0
        saturation = float(hsv[:, :, 1].mean()) / 128.0
        motion = 0.0 if prev_gray is None else float(cv2.absdiff(gray, prev_gray).mean()) / 32.0
        cut = 0.0 if prev_hist is None else float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        score = 0.50 * motion + 0.25 * cut + 0.15 * contrast + 0.10 * saturation
        samples.append((float(second), score))
        prev_gray = gray
        prev_hist = hist

    cap.release()
    if not samples:
        return duration, [(0.0, min(window_seconds, duration), 1.0)]

    window_count = max(1, int(math.ceil(duration / window_seconds)))
    windows = []
    for index in range(window_count):
        start = index * window_seconds
        end = min(duration, start + window_seconds)
        values = [score for second, score in samples if start <= second < end]
        if values:
            windows.append((start, end, float(np.mean(values))))
    return duration, windows


def select_segments(
    windows: list[tuple[float, float, float]],
    duration: float,
    summary_ratio: float,
    min_total_seconds: float,
) -> list[tuple[float, float]]:
    target = min(duration, max(min_total_seconds, duration * summary_ratio))
    ranked = sorted(windows, key=lambda item: item[2], reverse=True)
    selected = []
    total = 0.0
    for start, end, _ in ranked:
        if total >= target:
            break
        selected.append((start, end))
        total += end - start
    if not selected:
        selected = [(0.0, min(duration, target))]
    return sorted(selected)


def ffmpeg_bin() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is required")
    return binary


def export_segments(input_path: Path, output_path: Path, segments: list[tuple[float, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_bin()
    temp_dir = Path(tempfile.mkdtemp(prefix="simple_summary_", dir=str(output_path.parent)))
    try:
        parts = []
        for index, (start, end) in enumerate(segments):
            part = temp_dir / f"part_{index:04d}.mp4"
            cmd = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(input_path),
                "-t",
                f"{max(0.05, end - start):.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(part),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            parts.append(part)

        concat_file = temp_dir / "concat.txt"
        concat_file.write_text("".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8")
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a simple visual-summary mp4.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-ratio", type=float, default=0.15)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--min-total-seconds", type=float, default=3.0)
    parser.add_argument("--feature-npz", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    duration, windows = score_video(args.input, args.sample_fps, args.window_seconds)
    segments = select_segments(windows, duration, args.summary_ratio, args.min_total_seconds)
    export_segments(args.input, args.output, segments)
    if args.feature_npz:
        times = np.array([(start + end) / 2.0 for start, end, _ in windows], dtype=np.float32)
        scores = np.array([score for _, _, score in windows], dtype=np.float32)
        mask = np.zeros(len(windows), dtype=np.int8)
        for index, (start, end, _) in enumerate(windows):
            mid = (start + end) / 2.0
            mask[index] = int(any(seg_start <= mid < seg_end for seg_start, seg_end in segments))
        if scores.size:
            denom = float(scores.max() - scores.min())
            if denom > 1e-6:
                scores = (scores - scores.min()) / denom
        np.savez_compressed(
            args.feature_npz,
            pred_score=scores,
            summary_mask=mask,
            segments=np.asarray(segments, dtype=np.float32),
            times=times,
            fallback=np.asarray(True),
        )
    print(
        f"saved fallback summary: {args.output} "
        f"({len(segments)} segments, {sum(e - s for s, e in segments):.2f}s / {duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
