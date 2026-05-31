#!/usr/bin/env python3
"""End-to-end mp4 summarization with TriMamba.

The paper evaluates the summarization model on timestep-level pre-extracted
features. This script recreates that interface for a single mp4:

    mp4 -> CLIP/RoBERTa/AST 1 Hz features -> TriMamba scores -> summary mp4

Text can come from a timestamped transcript JSON, Whisper, or zero vectors.
Zero-vector text/audio is useful because the released model was trained with
modality dropout and the paper evaluates missing-modality deployment this way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from models import build_model
from utils.generate_summary import solve_knapsack


DEFAULT_CLIP = "openai/clip-vit-large-patch14"
DEFAULT_ROBERTA = "roberta-base"
DEFAULT_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a single mp4 with TriMamba")
    parser.add_argument("--input", required=True, help="Input mp4 path")
    parser.add_argument("--output", required=True, help="Output summary mp4 path")
    parser.add_argument("--config", default="configs/mosu.yaml", help="TriMamba yaml config")
    parser.add_argument("--ckpt", default="checkpoints/best_model_ckpt_mosu.pth", help="Model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--summary-ratio", type=float, default=0.15, help="Summary budget ratio")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Feature timestep rate. MoSu uses 1 Hz.")
    parser.add_argument("--batch-size", type=int, default=16, help="Encoder batch size")
    parser.add_argument("--feature-npz", default=None, help="Optional path to save extracted features and scores")

    parser.add_argument("--clip-model", default=DEFAULT_CLIP)
    parser.add_argument("--roberta-model", default=DEFAULT_ROBERTA)
    parser.add_argument("--ast-model", default=DEFAULT_AST)

    parser.add_argument(
        "--text-source",
        choices=("whisper", "transcript", "zero"),
        default="zero",
        help="How to build per-timestep text features. Default uses zero vectors.",
    )
    parser.add_argument("--transcript", default=None, help="JSON transcript: [{'start','end','text'}, ...]")
    parser.add_argument("--whisper-model", default="base", help="Whisper model name if --text-source whisper")
    parser.add_argument("--allow-missing-text", action="store_true", help="Use zero text vectors if text extraction fails")
    parser.add_argument("--allow-missing-audio", action="store_true", help="Use zero audio vectors if audio extraction fails")

    parser.add_argument(
        "--shot-method",
        choices=("hist", "fixed"),
        default="hist",
        help="Shot boundary method over 1 Hz timesteps",
    )
    parser.add_argument("--shot-threshold-std", type=float, default=1.0)
    parser.add_argument("--min-shot-seconds", type=int, default=2)
    parser.add_argument("--fixed-shot-seconds", type=int, default=5)

    parser.add_argument("--ffmpeg", default=None, help="ffmpeg binary path")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary clip files")
    return parser.parse_args()


def require_transformers():
    try:
        from transformers import (  # noqa: F401
            ASTModel,
            AutoFeatureExtractor,
            AutoModel,
            AutoTokenizer,
            CLIPModel,
            CLIPProcessor,
        )
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "Feature extraction requires transformers, pillow, and their model dependencies. "
            "Install them before running this script."
        ) from exc


def load_config(config_path: str, device: str) -> SimpleNamespace:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    defaults = {
        "model": "trimamba",
        "visual_dim": 768,
        "text_dim": 768,
        "audio_dim": 768,
        "input_dim": 128,
        "hidden_dim": 192,
        "num_heads": 4,
        "dropout": 0.1,
        "num_encoder_layers": 2,
        "num_bottleneck_layers": 2,
        "num_fusion_layers": 1,
        "stride": 4,
        "modalities": "vta",
        "modality_dropout_prob": 0.0,
        "get_attn_weights": False,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "num_epochs": 1,
    }
    defaults.update(data or {})
    defaults["device"] = torch.device(device)
    return SimpleNamespace(**defaults)


def video_duration_seconds(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frames / fps if fps > 0 and frames > 0 else 0
    cap.release()
    if duration <= 0:
        raise RuntimeError(f"Could not determine video duration: {path}")
    return duration


def timestep_centers(duration: float, sample_fps: float) -> np.ndarray:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be positive")
    step = 1.0 / sample_fps
    centers = np.arange(0.0, duration, step, dtype=np.float32) + step / 2.0
    return np.minimum(centers, max(duration - 1e-3, 0.0))


def read_frames_at_times(path: str, times: np.ndarray) -> list[Image.Image]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames: list[Image.Image] = []
    last_frame = None
    for second in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000.0)
        ok, frame_bgr = cap.read()
        if not ok:
            if last_frame is None:
                raise RuntimeError(f"Could not read frame around {second:.2f}s")
            frame_bgr = last_frame
        last_frame = frame_bgr
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def batched(items, batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def clip_output_to_tensor(model: torch.nn.Module, output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    image_embeds = getattr(output, "image_embeds", None)
    if torch.is_tensor(image_embeds):
        return image_embeds

    pooler_output = getattr(output, "pooler_output", None)
    if torch.is_tensor(pooler_output):
        projection = getattr(model, "visual_projection", None)
        if projection is not None and pooler_output.shape[-1] == projection.in_features:
            return projection(pooler_output)
        return pooler_output

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.dim() == 2:
                return value

    raise TypeError(f"Unsupported CLIP image feature output type: {type(output).__name__}")


def extract_visual_features(
    video_path: str,
    times: np.ndarray,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    frames = read_frames_at_times(video_path, times)
    outputs = []
    with torch.no_grad():
        for images in batched(frames, batch_size):
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = clip_output_to_tensor(model, model.get_image_features(**inputs))
            outputs.append(feats.detach().cpu().float().numpy())

    features = np.concatenate(outputs, axis=0).astype(np.float32)
    if features.shape[1] != 768:
        raise RuntimeError(
            f"Visual feature dimension is {features.shape[1]}, expected 768. "
            "Use a CLIP checkpoint with 768-d projected image features, e.g. openai/clip-vit-large-patch14."
        )
    return features


def load_transcript_json(path: str) -> list[Segment]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    segments = []
    for item in raw:
        start = float(item["start"])
        end = float(item["end"])
        text = str(item.get("text", "")).strip()
        if end > start and text:
            segments.append(Segment(start=start, end=end, text=text))
    return segments


def transcribe_with_whisper(video_path: str, model_name: str) -> list[Segment]:
    try:
        import whisper
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Whisper text extraction requires the openai-whisper package. "
            "Use --text-source transcript with a timestamped JSON file, or --text-source zero."
        ) from exc

    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)
    segments = []
    for item in result.get("segments", []):
        text = str(item.get("text", "")).strip()
        if text:
            segments.append(Segment(float(item["start"]), float(item["end"]), text))
    return segments


def text_for_timestep(center: float, step: float, segments: list[Segment]) -> str:
    start = center - step / 2.0
    end = center + step / 2.0
    parts = [seg.text for seg in segments if seg.end > start and seg.start < end]
    return " ".join(parts).strip()


def extract_text_features(
    video_path: str,
    times: np.ndarray,
    sample_fps: float,
    source: str,
    transcript: str | None,
    whisper_model: str,
    roberta_model: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    if source == "zero":
        return np.zeros((len(times), 768), dtype=np.float32)
    if source == "transcript":
        if not transcript:
            raise ValueError("--transcript is required when --text-source transcript")
        segments = load_transcript_json(transcript)
    else:
        segments = transcribe_with_whisper(video_path, whisper_model)

    step = 1.0 / sample_fps
    texts = [text_for_timestep(float(center), step, segments) for center in times]

    tokenizer = AutoTokenizer.from_pretrained(roberta_model)
    model = AutoModel.from_pretrained(roberta_model).to(device).eval()
    outputs = []
    with torch.no_grad():
        for chunk in batched(texts, batch_size):
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            model_out = model(**encoded)
            feats = getattr(model_out, "pooler_output", None)
            if feats is None:
                feats = model_out.last_hidden_state[:, 0, :]
            outputs.append(feats.detach().cpu().float().numpy())

    features = np.concatenate(outputs, axis=0).astype(np.float32)
    if features.shape[1] != 768:
        raise RuntimeError(f"Text feature dimension is {features.shape[1]}, expected 768.")
    return features


def ffmpeg_binary(user_path: str | None) -> str:
    candidate = user_path or shutil.which("ffmpeg")
    if candidate is None:
        snap_ffmpeg = Path("/snap/bin/ffmpeg")
        if snap_ffmpeg.exists():
            candidate = str(snap_ffmpeg)
    if candidate is None:
        raise RuntimeError("ffmpeg is required for audio decoding and mp4 export")
    return candidate


def decode_audio_mono(video_path: str, ffmpeg: str, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        video_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty")
    return audio


def audio_window(audio: np.ndarray, center: float, duration: float, sample_rate: int, radius: float = 5.0) -> np.ndarray:
    start = max(0.0, center - radius)
    end = min(duration, center + radius)
    start_idx = int(round(start * sample_rate))
    end_idx = int(round(end * sample_rate))
    chunk = audio[start_idx:end_idx]
    if chunk.size == 0:
        chunk = np.zeros(sample_rate, dtype=np.float32)
    return chunk.astype(np.float32)


def extract_audio_features(
    video_path: str,
    times: np.ndarray,
    duration: float,
    ast_model: str,
    batch_size: int,
    device: torch.device,
    ffmpeg: str,
) -> np.ndarray:
    from transformers import ASTModel, AutoFeatureExtractor

    sample_rate = 16000
    audio = decode_audio_mono(video_path, ffmpeg, sample_rate=sample_rate)
    windows = [audio_window(audio, float(center), duration, sample_rate) for center in times]

    extractor = AutoFeatureExtractor.from_pretrained(ast_model)
    model = ASTModel.from_pretrained(ast_model).to(device).eval()

    outputs = []
    with torch.no_grad():
        for chunk in batched(windows, batch_size):
            inputs = extractor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
            model_out = model(**inputs)
            feats = getattr(model_out, "pooler_output", None)
            if feats is None:
                feats = model_out.last_hidden_state[:, 0, :]
            outputs.append(feats.detach().cpu().float().numpy())

    features = np.concatenate(outputs, axis=0).astype(np.float32)
    if features.shape[1] != 768:
        raise RuntimeError(f"Audio feature dimension is {features.shape[1]}, expected 768.")
    return features


def detect_shots_from_histograms(video_path: str, times: np.ndarray, min_len: int, threshold_std: float) -> np.ndarray:
    frames = read_frames_at_times(video_path, times)
    hists = []
    for image in frames:
        arr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([arr], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)

    diffs = np.array(
        [cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA) for i in range(1, len(hists))],
        dtype=np.float32,
    )
    if diffs.size == 0:
        return np.array([[0, len(times)]], dtype=np.int32)

    threshold = float(diffs.mean() + threshold_std * diffs.std())
    boundaries = [0]
    for index, diff in enumerate(diffs, start=1):
        if diff >= threshold and index - boundaries[-1] >= min_len:
            boundaries.append(index)
    if len(times) - boundaries[-1] < min_len and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(len(times))
    return np.array([[boundaries[i], boundaries[i + 1]] for i in range(len(boundaries) - 1)], dtype=np.int32)


def fixed_shots(num_steps: int, shot_seconds: int, sample_fps: float) -> np.ndarray:
    width = max(1, int(round(shot_seconds * sample_fps)))
    return np.array(
        [[start, min(start + width, num_steps)] for start in range(0, num_steps, width)],
        dtype=np.int32,
    )


def select_summary_shots(scores: np.ndarray, change_points: np.ndarray, summary_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    num_steps = len(scores)
    capacity = max(1, int(math.floor(num_steps * summary_ratio)))

    weights = []
    values = []
    valid_shots = []
    for start, end in change_points:
        start_i = int(max(0, start))
        end_i = int(min(num_steps, end))
        if end_i <= start_i:
            continue
        weights.append(end_i - start_i)
        values.append(float(scores[start_i:end_i].mean()))
        valid_shots.append((start_i, end_i))

    selected = solve_knapsack(capacity, weights, values, len(weights))
    mask = np.zeros(num_steps, dtype=np.int8)
    for idx in selected:
        start, end = valid_shots[idx]
        mask[start:end] = 1
    selected_shots = np.array([valid_shots[idx] for idx in selected], dtype=np.int32)
    return mask, selected_shots


def mask_to_time_segments(mask: np.ndarray, sample_fps: float, duration: float) -> list[tuple[float, float]]:
    segments = []
    start = None
    for idx, value in enumerate(mask.tolist() + [0]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            end = idx
            segments.append((start / sample_fps, min(end / sample_fps, duration)))
            start = None
    return [(s, e) for s, e in segments if e - s > 0.05]


def export_summary_mp4(
    input_path: str,
    output_path: str,
    segments: list[tuple[float, float]],
    ffmpeg: str,
    keep_temp: bool = False,
) -> None:
    if not segments:
        raise RuntimeError("No summary segments were selected")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="trimamba_summary_parts_", dir=str(output.parent)))

    try:
        part_paths = []
        for idx, (start, end) in enumerate(segments):
            part = temp_dir / f"part_{idx:04d}.mp4"
            duration = max(0.05, end - start)
            cmd = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                input_path,
                "-t",
                f"{duration:.3f}",
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
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "ffmpeg failed while creating a summary segment:\n"
                    f"  segment: {start:.3f}s - {end:.3f}s\n"
                    f"  output: {part}\n"
                    f"  stderr: {proc.stderr.strip()}"
                )
            part_paths.append(part)

        concat_file = temp_dir / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in part_paths),
            encoding="utf-8",
        )
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
            str(output),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg failed while concatenating summary segments:\n"
                f"  concat list: {concat_file}\n"
                f"  output: {output}\n"
                f"  stderr: {proc.stderr.strip()}"
            )
    finally:
        if keep_temp:
            print(f"Temporary clips kept at {temp_dir}", file=sys.stderr)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_model(cfg: SimpleNamespace, ckpt_path: str) -> torch.nn.Module:
    model = build_model(cfg).to(cfg.device).eval()
    try:
        state = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state, strict=True)
    return model


def predict_scores(
    model: torch.nn.Module,
    visual: np.ndarray,
    text: np.ndarray,
    audio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    mask = torch.ones((1, visual.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        output, _ = model(
            torch.from_numpy(visual).unsqueeze(0).to(device),
            torch.from_numpy(text).unsqueeze(0).to(device),
            torch.from_numpy(audio).unsqueeze(0).to(device),
            mask=mask,
        )
    return output.squeeze(0).detach().cpu().float().numpy()


def main() -> None:
    args = parse_args()
    require_transformers()

    input_path = str(Path(args.input).expanduser().resolve())
    output_path = str(Path(args.output).expanduser().resolve())
    device = torch.device(args.device)
    ffmpeg = ffmpeg_binary(args.ffmpeg)

    duration = video_duration_seconds(input_path)
    times = timestep_centers(duration, args.sample_fps)
    cfg = load_config(args.config, args.device)

    print(f"[1/7] duration={duration:.2f}s, timesteps={len(times)} at {args.sample_fps:g} Hz")
    print("[2/7] extracting CLIP visual features")
    visual = extract_visual_features(input_path, times, args.clip_model, args.batch_size, device)

    print(f"[3/7] extracting text features via {args.text_source}")
    try:
        text = extract_text_features(
            input_path,
            times,
            args.sample_fps,
            args.text_source,
            args.transcript,
            args.whisper_model,
            args.roberta_model,
            args.batch_size,
            device,
        )
    except Exception:
        if not args.allow_missing_text:
            raise
        print("      text extraction failed; using zero text vectors")
        text = np.zeros((len(times), cfg.text_dim), dtype=np.float32)

    print("[4/7] extracting AST audio features")
    try:
        audio = extract_audio_features(input_path, times, duration, args.ast_model, args.batch_size, device, ffmpeg)
    except Exception:
        if not args.allow_missing_audio:
            raise
        print("      audio extraction failed; using zero audio vectors")
        audio = np.zeros((len(times), cfg.audio_dim), dtype=np.float32)

    print("[5/7] loading TriMamba and predicting timestep scores")
    model = load_model(cfg, args.ckpt)
    scores = predict_scores(model, visual, text, audio, device)

    print(f"[6/7] selecting shots with {args.shot_method} boundaries")
    if args.shot_method == "hist":
        change_points = detect_shots_from_histograms(
            input_path,
            times,
            min_len=max(1, int(round(args.min_shot_seconds * args.sample_fps))),
            threshold_std=args.shot_threshold_std,
        )
        if len(change_points) <= 1:
            change_points = fixed_shots(len(times), args.fixed_shot_seconds, args.sample_fps)
    else:
        change_points = fixed_shots(len(times), args.fixed_shot_seconds, args.sample_fps)

    summary_mask, selected_shots = select_summary_shots(scores, change_points, args.summary_ratio)
    segments = mask_to_time_segments(summary_mask, args.sample_fps, duration)
    if not segments:
        top_idx = int(np.argmax(scores))
        segments = [(top_idx / args.sample_fps, min((top_idx + 1) / args.sample_fps, duration))]

    if args.feature_npz:
        np.savez_compressed(
            args.feature_npz,
            visual=visual,
            text=text,
            audio=audio,
            pred_score=scores,
            change_points=change_points,
            selected_shots=selected_shots,
            summary_mask=summary_mask,
            segments=np.array(segments, dtype=np.float32),
        )

    print(f"[7/7] exporting {len(segments)} segments to {output_path}")
    export_summary_mp4(input_path, output_path, segments, ffmpeg, keep_temp=args.keep_temp)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
