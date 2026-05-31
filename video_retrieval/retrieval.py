#!/usr/bin/env python3
"""Lighthouse QD-DETR video retrieval for the local web app.

This is the web-app friendly version of ``lighthouse/gradio_demo/demo.py``:
index once with clip_slowfast, then answer text queries with qd_detr moments,
saliency scores, and highlight frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import torch


RETRIEVAL_ROOT = Path(__file__).resolve().parent
LIGHTHOUSE_ROOT = RETRIEVAL_ROOT / "lighthouse"
WEIGHTS_ROOT = RETRIEVAL_ROOT / "weights"
if str(LIGHTHOUSE_ROOT) not in sys.path:
    sys.path.insert(0, str(LIGHTHOUSE_ROOT))

from lighthouse.models import QDDETRPredictor  # noqa: E402


FEATURE_NAME = "clip_slowfast"
MODEL_NAME = "qd_detr"
DEFAULT_CKPT = WEIGHTS_ROOT / "clip_slowfast_qd_detr_qvhighlight.ckpt"
DEFAULT_SLOWFAST = WEIGHTS_ROOT / "SLOWFAST_8x8_R50.pkl"


def device_from_arg(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def video_duration(path: Path) -> tuple[float, float, int]:
    probe = ffmpeg.probe(str(path))
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if not streams:
        raise RuntimeError(f"Could not find a video stream: {path}")
    stream = streams[0]
    duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    numerator, denominator = rate.split("/", 1)
    fps = float(numerator) / max(float(denominator), 1e-6)
    frames = int(stream.get("nb_frames") or round(duration * fps))
    if duration <= 0 or fps <= 0:
        raise RuntimeError(f"Could not determine video duration: {path}")
    return duration, fps, frames


def make_predictor(args: argparse.Namespace) -> QDDETRPredictor:
    ckpt = Path(args.ckpt)
    slowfast = Path(args.slowfast)
    if not ckpt.exists():
        raise FileNotFoundError(f"QD-DETR checkpoint not found: {ckpt}")
    if not slowfast.exists():
        raise FileNotFoundError(f"SlowFast checkpoint not found: {slowfast}")
    return QDDETRPredictor(
        str(ckpt),
        device=device_from_arg(args.device),
        feature_name=FEATURE_NAME,
        slowfast_path=str(slowfast),
        pann_path=None,
    )


def tensors_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tensors_to_cpu(item) for key, item in value.items()}
    return value


def tensors_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tensors_to_device(item, device) for key, item in value.items()}
    return value


def safe_query_dir(query: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", query).strip("._")[:48] or "query"
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def extract_frame(video_path: Path, second: float, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (
            ffmpeg.input(str(video_path), ss=max(0.0, float(second)))
            .filter("scale", 360, -2)
            .output(str(output_path), vframes=1, qscale=2)
            .global_args("-loglevel", "quiet", "-y")
            .run()
        )
    except ffmpeg.Error:
        return False
    return output_path.exists()


def build_index(args: argparse.Namespace) -> None:
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    predictor = make_predictor(args)
    duration, fps, frames = video_duration(args.video)
    encoded = predictor.encode_video(str(args.video))
    payload = {
        "format": "lighthouse_qd_detr_index_v1",
        "created_at": time.time(),
        "video": str(args.video),
        "duration": float(duration),
        "fps": float(fps),
        "frames": int(frames),
        "feature": FEATURE_NAME,
        "model": MODEL_NAME,
        "clip_len": float(predictor._clip_len),
        "encoded_video": tensors_to_cpu(encoded),
    }
    torch.save(payload, output)
    print(
        f"Saved Lighthouse index: {output} "
        f"({FEATURE_NAME} + {MODEL_NAME}, duration={duration:.2f}s, clip_len={predictor._clip_len:.2f}s)"
    )


def load_index(path: Path, device: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "lighthouse_qd_detr_index_v1":
        raise RuntimeError(f"Unsupported retrieval index format: {payload.get('format')}")
    payload["encoded_video"] = tensors_to_device(payload["encoded_video"], device)
    return payload


def moment_rows(prediction: dict[str, Any], top_k: int, duration: float) -> list[dict[str, float]]:
    rows = []
    for rank, row in enumerate(prediction.get("pred_relevant_windows", [])[:top_k], start=1):
        start = max(0.0, min(float(row[0]), duration))
        end = max(start, min(float(row[1]), duration))
        rows.append(
            {
                "rank": rank,
                "start": start,
                "end": end,
                "time": start,
                "score": float(row[2]),
            }
        )
    return rows


def saliency_rows(prediction: dict[str, Any], clip_len: float, duration: float) -> list[dict[str, float]]:
    rows = []
    for index, score in enumerate(prediction.get("pred_saliency_scores", [])):
        start = min(float(index) * clip_len, duration)
        end = min(start + clip_len, duration)
        rows.append(
            {
                "index": index,
                "time": start,
                "start": start,
                "end": end,
                "score": float(score),
            }
        )
    return rows


def highlight_rows(
    saliency: list[dict[str, float]],
    video_path: Path,
    output_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(saliency, key=lambda row: row["score"], reverse=True)[:top_k]
    rows = []
    for rank, row in enumerate(ranked, start=1):
        second = float(row["time"])
        image_path = output_dir / f"highlight_{rank:02d}_{second:08.3f}.jpg"
        extract_frame(video_path, second, image_path)
        rows.append(
            {
                "rank": rank,
                "time": second,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "score": float(row["score"]),
                "thumbnail_path": str(image_path),
            }
        )
    return rows


def query_index(args: argparse.Namespace) -> None:
    device = device_from_arg(args.device)
    payload = load_index(args.index, device)
    predictor = make_predictor(args)
    prediction = predictor.predict(args.query, payload["encoded_video"])
    if prediction is None:
        raise RuntimeError("QD-DETR returned no prediction")

    duration = float(payload["duration"])
    clip_len = float(payload["clip_len"])
    video_path = Path(payload["video"])
    out_dir = args.frames_dir or args.index.parent / "retrieval_highlights" / safe_query_dir(args.query)
    saliency = saliency_rows(prediction, clip_len=clip_len, duration=duration)
    highlights = highlight_rows(
        saliency,
        video_path=video_path,
        output_dir=out_dir,
        top_k=max(1, args.highlight_top_k),
    )
    moments = moment_rows(prediction, top_k=max(1, args.top_k), duration=duration)
    result = {
        "query": args.query,
        "feature": FEATURE_NAME,
        "model": MODEL_NAME,
        "duration": duration,
        "clip_len": clip_len,
        "moments": moments,
        "saliency": saliency,
        "highlights": highlights,
        "hits": highlights,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/query Lighthouse clip_slowfast + qd_detr retrieval.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--slowfast", type=Path, default=DEFAULT_SLOWFAST)
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index")
    index.add_argument("--video", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--frames-dir", type=Path, default=None, help="Kept for API compatibility; highlights are made at query time.")
    index.add_argument("--sample-fps", type=float, default=1.0, help="Ignored; Lighthouse uses its trained clip length.")
    index.add_argument("--device", default=None)
    index.add_argument("--ckpt", type=Path, default=None)
    index.add_argument("--slowfast", type=Path, default=None)

    query = sub.add_parser("query")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--highlight-top-k", type=int, default=5)
    query.add_argument("--frames-dir", type=Path, default=None)
    query.add_argument("--event-window", type=float, default=3.0, help="Ignored; QD-DETR predicts temporal windows directly.")
    query.add_argument("--device", default=None)
    query.add_argument("--ckpt", type=Path, default=None)
    query.add_argument("--slowfast", type=Path, default=None)
    args = parser.parse_args()

    args.device = args.device or parser.get_default("device")
    args.ckpt = args.ckpt or parser.get_default("ckpt")
    args.slowfast = args.slowfast or parser.get_default("slowfast")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "index":
        build_index(args)
    elif args.command == "query":
        query_index(args)


if __name__ == "__main__":
    main()
