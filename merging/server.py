#!/usr/bin/env python3
"""Local web app for video upload, LingBot-MAP reconstruction, and TriMamba summary."""

from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

MERGING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("DRONE_PROJECT_ROOT", MERGING_ROOT.parent)).expanduser().resolve()
LINGBOT_ROOT = PROJECT_ROOT / "lingbot-map"
TRIMAMBA_ROOT = PROJECT_ROOT / "Trimamba"
RETRIEVAL_ROOT = PROJECT_ROOT / "video_retrieval"
DATA_ROOT = Path(os.environ.get("MERGING_DATA_ROOT", MERGING_ROOT / "data" / "jobs")).expanduser()
STATIC_ROOT = MERGING_ROOT / "static"
MODEL_PATH = Path(os.environ.get("LINGBOT_MODEL_PATH", LINGBOT_ROOT / "lingbot-map-long.pt")).expanduser()
TRIMAMBA_CKPT = Path(os.environ.get("TRIMAMBA_CKPT", TRIMAMBA_ROOT / "checkpoints" / "best_model_ckpt.pth")).expanduser()
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".qt", ".webm", ".mkv", ".avi"}
WEB_PREVIEW_SUFFIXES = {".mov", ".m4v", ".qt", ".mkv", ".avi"}

JOBS: dict[str, dict] = {}
VIEWERS: dict[tuple[str, str], dict] = {}
RUNNING_JOBS: dict[str, dict] = {}
STATUS_LOCK = threading.Lock()
VIEWER_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()


def default_conda_root() -> Path:
    explicit = os.environ.get("CONDA_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return Path(conda_exe).expanduser().resolve().parents[1]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix).expanduser().resolve().parent.parent
    return Path.home() / "miniconda3"


def conda_python(env_var: str, env_name: str) -> Path:
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit).expanduser()
    return default_conda_root() / "envs" / env_name / "bin" / "python"


LINGBOT_PYTHON = conda_python("LINGBOT_PYTHON", "lingbot-map")
TRIMAMBA_PYTHON = conda_python("TRIMAMBA_PYTHON", "triple")
RETRIEVAL_PYTHON = conda_python("RETRIEVAL_PYTHON", "retrieval")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name).strip("._")
    return stem or "video.mp4"


def media_url(job_id: str, path: Path) -> str:
    return f"/media/{job_id}/{path.name}"


def media_url_relative(job_id: str, path: Path) -> str:
    relative = path.resolve().relative_to(job_dir(job_id).resolve())
    return f"/media/{job_id}/{relative.as_posix()}"


def job_dir(job_id: str) -> Path:
    return DATA_ROOT / job_id


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.log"


def append_log(job_id: str, message: str) -> None:
    log_file = log_path(job_id)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        for line in str(message).splitlines() or [""]:
            handle.write(f"[{timestamp()}] {line}\n")


def log_tail(job_id: str, max_chars: int = 16_000) -> str:
    path = log_path(job_id)
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_chars))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def status_file(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def write_status(status: dict) -> None:
    status["updated_at"] = time.time()
    status["updated_at_text"] = timestamp()
    path = status_file(status["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def set_status(job_id: str, **fields) -> dict:
    with STATUS_LOCK:
        status = JOBS[job_id]
        for key, value in fields.items():
            if key == "artifacts":
                status["artifacts"] = dict(value)
            elif key == "params":
                status.setdefault("params", {}).update(value)
            else:
                status[key] = value
        write_status(status)
        return dict(status)


def record_error(job_id: str, message: str) -> None:
    append_log(job_id, f"ERROR: {message}")
    with STATUS_LOCK:
        status = JOBS[job_id]
        status.setdefault("errors", []).append(message)
        write_status(status)


def collect_artifacts(job_id: str) -> dict[str, str]:
    directory = job_dir(job_id)
    uploaded_video = next(directory.glob("input.*"), None)
    browser_preview = directory / "browser_preview.mp4"
    candidates = {
        "input_video_url": browser_preview if browser_preview.exists() else uploaded_video,
        "trimmed_video_url": directory / "trimmed.mp4",
        "raw_npz_url": directory / "lingbot_raw.npz",
        "raw_pcd_url": directory / "lingbot_raw.pcd",
        "post_pcd_url": directory / "translation_only_damped.pcd",
        "summary_video_url": directory / "trimamba_summary.mp4",
        "summary_features_url": directory / "trimamba_features.npz",
        "retrieval_index_url": retrieval_index_path(job_id),
    }
    artifacts = {
        key: media_url(job_id, value)
        for key, value in candidates.items()
        if value is not None and value.exists()
    }
    return artifacts


def refresh_artifacts(job_id: str) -> None:
    artifacts = collect_artifacts(job_id)
    set_status(job_id, artifacts=artifacts)


def read_status(job_id: str) -> dict | None:
    path = status_file(job_id)
    if not path.exists():
        with STATUS_LOCK:
            JOBS.pop(job_id, None)
        return None

    with STATUS_LOCK:
        status = JOBS.get(job_id)
        if status is None:
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                JOBS[status["id"]] = status
            except Exception:
                return None
        result = dict(status)
        result["params"] = dict(status.get("params", {}))
        result["errors"] = list(status.get("errors", []))
    result["artifacts"] = collect_artifacts(job_id)
    with RUN_LOCK:
        result["is_running_local"] = job_id in RUNNING_JOBS
    result["log_tail"] = log_tail(job_id)
    return result


def list_statuses() -> list[dict]:
    load_existing_jobs()
    rows = []
    for job_id in sorted(JOBS, key=lambda item: JOBS[item].get("created_at", 0), reverse=True):
        status = read_status(job_id)
        if status is None:
            continue
        status.pop("log_tail", None)
        rows.append(status)
    return rows


def load_existing_jobs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for path in DATA_ROOT.glob("*/status.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            JOBS[status["id"]] = status
        except Exception:
            continue


def probe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(proc.stdout.strip())
    except Exception:
        return None


def create_browser_preview(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
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
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "failed to create browser preview")


def run_command(job_id: str, cmd: list[str], cwd: Path, label: str) -> None:
    append_log(job_id, f"$ {shlex.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with RUN_LOCK:
        if job_id in RUNNING_JOBS:
            RUNNING_JOBS[job_id]["process"] = proc
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            append_log(job_id, line.rstrip())
            if should_stop(job_id) and proc.poll() is None:
                proc.terminate()
        code = proc.wait()
    finally:
        with RUN_LOCK:
            if job_id in RUNNING_JOBS and RUNNING_JOBS[job_id].get("process") is proc:
                RUNNING_JOBS[job_id]["process"] = None
    if should_stop(job_id):
        raise RuntimeError(f"{label} stopped by user")
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}")


def mark_running(job_id: str) -> None:
    with RUN_LOCK:
        RUNNING_JOBS[job_id] = {"process": None, "stop_requested": False, "started_at": time.time()}


def mark_finished(job_id: str) -> None:
    with RUN_LOCK:
        RUNNING_JOBS.pop(job_id, None)


def should_stop(job_id: str) -> bool:
    with RUN_LOCK:
        return bool(RUNNING_JOBS.get(job_id, {}).get("stop_requested"))


def request_stop(job_id: str) -> bool:
    with RUN_LOCK:
        info = RUNNING_JOBS.get(job_id)
        if info is None:
            return False
        info["stop_requested"] = True
        proc = info.get("process")
    if proc is not None and proc.poll() is None:
        proc.terminate()
    append_log(job_id, "Stop requested.")
    set_status(job_id, state="stopping", stage="stopping")
    return True


def stop_viewers(job_id: str) -> None:
    with VIEWER_LOCK:
        items = [(key, value) for key, value in VIEWERS.items() if key[0] == job_id]
    for key, info in items:
        proc = info.get("process")
        if proc is not None and proc.poll() is None:
            proc.terminate()
        with VIEWER_LOCK:
            VIEWERS.pop(key, None)


def delete_job(job_id: str, force: bool = False) -> None:
    with RUN_LOCK:
        is_running = job_id in RUNNING_JOBS
    if is_running and not force:
        raise RuntimeError("job is running")
    if is_running:
        request_stop(job_id)
        time.sleep(0.2)
    stop_viewers(job_id)
    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    with STATUS_LOCK:
        JOBS.pop(job_id, None)


def retrieval_index_path(job_id: str) -> Path:
    return job_dir(job_id) / "lighthouse_qd_detr_index.pt"


def start_retrieval_index(job_id: str, sample_fps: float = 1.0) -> None:
    trimmed = job_dir(job_id) / "trimmed.mp4"
    if not trimmed.exists():
        raise FileNotFoundError("trimmed video is not ready yet")

    def worker() -> None:
        mark_running(job_id)
        set_status(job_id, retrieval_state="indexing")
        try:
            run_command(
                job_id,
                [
                    str(RETRIEVAL_PYTHON),
                    str(RETRIEVAL_ROOT / "retrieval.py"),
                    "index",
                    "--video",
                    str(trimmed),
                    "--output",
                    str(retrieval_index_path(job_id)),
                    "--frames-dir",
                    str(job_dir(job_id) / "retrieval_frames"),
                    "--sample-fps",
                    f"{sample_fps:.3f}",
                    "--device",
                    "auto",
                ],
                RETRIEVAL_ROOT,
                "video retrieval indexing",
            )
            set_status(job_id, retrieval_state="ready")
            refresh_artifacts(job_id)
        except Exception as exc:
            record_error(job_id, str(exc))
            append_log(job_id, traceback.format_exc())
            if should_stop(job_id):
                set_status(job_id, retrieval_state="stopped")
            else:
                set_status(job_id, retrieval_state="failed")
        finally:
            mark_finished(job_id)

    threading.Thread(target=worker, daemon=True).start()


def query_retrieval(job_id: str, query: str, top_k: int = 8) -> dict:
    index_path = retrieval_index_path(job_id)
    if not index_path.exists():
        raise FileNotFoundError("retrieval index is not ready yet")
    proc = subprocess.run(
        [
                    str(RETRIEVAL_PYTHON),
            str(RETRIEVAL_ROOT / "retrieval.py"),
            "query",
            "--index",
            str(index_path),
            "--query",
            query,
            "--top-k",
            str(top_k),
            "--highlight-top-k",
            "5",
            "--frames-dir",
            str(job_dir(job_id) / "retrieval_highlights"),
            "--device",
            "auto",
        ],
        cwd=str(RETRIEVAL_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "retrieval query failed")
    json_start = proc.stdout.find("{")
    if json_start < 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "retrieval query returned no JSON")
    result = json.loads(proc.stdout[json_start:])
    for row in result.get("hits", []):
        thumb = Path(row.pop("thumbnail_path"))
        row["thumbnail_url"] = media_url_relative(job_id, thumb)
    for row in result.get("highlights", []):
        if "thumbnail_path" not in row:
            continue
        thumb = Path(row.pop("thumbnail_path"))
        row["thumbnail_url"] = media_url_relative(job_id, thumb)
    for event in result.get("events", []):
        thumb_value = event.get("thumbnail_url") or event.get("thumbnail_path")
        if thumb_value:
            event["thumbnail_url"] = media_url_relative(job_id, Path(thumb_value))
    return result


def summary_analysis(job_id: str) -> dict:
    feature_npz = job_dir(job_id) / "trimamba_features.npz"
    if not feature_npz.exists():
        raise FileNotFoundError("summary feature NPZ is not ready yet")
    duration = probe_duration(job_dir(job_id) / "trimmed.mp4") or 0.0
    with np.load(feature_npz, allow_pickle=False) as data:
        scores = np.asarray(data["pred_score"], dtype=np.float32).reshape(-1)
        segments = np.asarray(
            data["segments"] if "segments" in data.files else np.zeros((0, 2)),
            dtype=np.float32,
        ).reshape(-1, 2)
        summary_mask = np.asarray(
            data["summary_mask"] if "summary_mask" in data.files else np.zeros(len(scores)),
            dtype=np.int8,
        ).reshape(-1)
        change_points = np.asarray(
            data["change_points"] if "change_points" in data.files else np.zeros((0, 2)),
            dtype=np.int32,
        ).reshape(-1, 2)
        selected_shots = np.asarray(
            data["selected_shots"] if "selected_shots" in data.files else np.zeros((0, 2)),
            dtype=np.int32,
        ).reshape(-1, 2)
        fallback = bool(np.asarray(data["fallback"]).item()) if "fallback" in data.files else False
        if "times" in data.files:
            times = np.asarray(data["times"], dtype=np.float32).reshape(-1)
        else:
            times = np.arange(len(scores), dtype=np.float32)
    if summary_mask.size != scores.size:
        summary_mask = np.zeros(len(scores), dtype=np.int8)
        for i, t in enumerate(times):
            summary_mask[i] = int(any(float(start) <= float(t) < float(end) for start, end in segments))
    score_rows = [
        {"time": float(t), "score": float(s), "selected": bool(summary_mask[i])}
        for i, (t, s) in enumerate(zip(times.tolist(), scores.tolist()))
    ]
    if len(times) > 1:
        step = float(np.median(np.diff(times)))
        if not np.isfinite(step) or step <= 0:
            step = 1.0
    else:
        step = 1.0
    selected_pairs = {(int(start), int(end)) for start, end in selected_shots.tolist()}
    shot_rows = []
    for start, end in change_points.tolist():
        start_i = int(min(max(start, 0), len(scores)))
        end_i = int(min(max(end, 0), len(scores)))
        if end_i <= start_i:
            continue
        if len(times):
            time_start = float(times[min(start_i, len(times) - 1)])
            time_end = float(times[min(end_i - 1, len(times) - 1)] + step)
        else:
            time_start = float(start_i)
            time_end = float(end_i)
        shot_scores = scores[start_i:end_i]
        shot_mask = summary_mask[start_i:end_i]
        shot_rows.append(
            {
                "start": start_i,
                "end": end_i,
                "time_start": time_start,
                "time_end": min(time_end, duration) if duration else time_end,
                "length": end_i - start_i,
                "mean_score": float(shot_scores.mean()),
                "max_score": float(shot_scores.max()),
                "selected": (start_i, end_i) in selected_pairs or bool(shot_mask.any()),
            }
        )
    return {
        "duration": duration,
        "fallback": fallback,
        "segments": [{"start": float(start), "end": float(end)} for start, end in segments.tolist()],
        "scores": score_rows,
        "shots": shot_rows,
        "score_min": float(scores.min()) if scores.size else 0.0,
        "score_max": float(scores.max()) if scores.size else 0.0,
    }


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def wait_for_local_port(port: int, proc: subprocess.Popen, timeout: float = 12.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.25)
    return proc.poll() is None


def viewer_process_is_running(info: dict | None) -> bool:
    proc = info.get("process") if info else None
    return proc is not None and proc.poll() is None


def launch_viewer(job_id: str, kind: str, options: dict | None = None) -> dict:
    options = options or {}
    directory = job_dir(job_id)
    if kind == "raw":
        input_path = directory / "lingbot_raw.npz"
        if not input_path.exists():
            raise FileNotFoundError("LingBot raw NPZ is not ready yet")
        port = find_free_port()
        cmd = [
            str(LINGBOT_PYTHON),
            str(MERGING_ROOT / "lingbot_npz_viewer.py"),
            "--input",
            str(input_path),
            "--port",
            str(port),
            "--conf_threshold",
            "1.5",
            "--downsample_factor",
            "10",
            "--depth_stride",
            "1",
        ]
        cwd = MERGING_ROOT
        log_file = directory / "viewer_raw.log"
        label = "LingBot demo viewer"
    elif kind == "post":
        input_path = directory / "translation_only_damped.pcd"
        if not input_path.exists():
            raise FileNotFoundError("postprocessed PCD is not ready yet")
        port = find_free_port()
        point_size = min(max(float(options.get("point_size", 0.03) or 0.03), 0.005), 0.2)
        max_view_points = min(max(int(options.get("max_points", 1_000_000) or 1_000_000), 10_000), 3_000_000)
        cmd = [
            str(LINGBOT_PYTHON),
            str(LINGBOT_ROOT / "visualize_predictions.py"),
            str(input_path),
            "--port",
            str(port),
            "--point_size",
            f"{point_size:.4f}",
            "--max_points",
            str(max_view_points),
        ]
        cwd = LINGBOT_ROOT
        log_file = directory / "viewer_post.log"
        label = "postprocess point cloud viewer"
    else:
        raise ValueError("viewer kind must be 'raw' or 'post'")

    with VIEWER_LOCK:
        key = (job_id, kind)
        current = VIEWERS.get(key)
        if viewer_process_is_running(current):
            if kind == "post" and options.get("restart"):
                current["process"].terminate()
                VIEWERS.pop(key, None)
                time.sleep(0.2)
            else:
                return {
                    "kind": kind,
                    "url": current["url"],
                    "port": current["port"],
                    "state": "running",
                }

        append_log(job_id, f"Starting {label}: {shlex.join(cmd)}")
        log_handle = log_file.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_handle.close()
        info = {
            "process": proc,
            "port": port,
            "url": f"http://127.0.0.1:{port}",
            "kind": kind,
            "started_at": time.time(),
        }
        VIEWERS[key] = info
        ready = wait_for_local_port(port, proc)
        info["ready_at"] = time.time() if ready else None
        return {"kind": kind, "url": info["url"], "port": port, "state": "ready" if ready else "starting"}


def trim_video(
    job_id: str,
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float | None,
) -> None:
    duration = None
    if end_seconds is not None and np.isfinite(end_seconds) and end_seconds > start_seconds:
        duration = max(0.05, end_seconds - start_seconds)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, start_seconds):.3f}",
        "-i",
        str(input_path),
    ]
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend([
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ])
    run_command(job_id, cmd, MERGING_ROOT, "video trimming")


def run_job_impl(job_id: str, params: dict) -> None:
    directory = job_dir(job_id)
    input_path = next(directory.glob("input.*"), None)
    if input_path is None:
        record_error(job_id, "uploaded video is missing")
        set_status(job_id, state="failed", stage="failed")
        return

    start_seconds = float(params.get("start_seconds", 0.0))
    raw_end_seconds = params.get("end_seconds")
    end_seconds = float(raw_end_seconds) if raw_end_seconds not in (None, "") else None
    if end_seconds is not None and (not np.isfinite(end_seconds) or end_seconds <= start_seconds + 0.05):
        end_seconds = None
    lingbot_fps = int(params.get("lingbot_fps", 5))
    summary_ratio = float(params.get("summary_ratio", 0.15))
    max_points = int(params.get("max_points", 800_000))

    trimmed = directory / "trimmed.mp4"
    raw_npz = directory / "lingbot_raw.npz"
    raw_pcd = directory / "lingbot_raw.pcd"
    post_pcd = directory / "translation_only_damped.pcd"
    summary_mp4 = directory / "trimamba_summary.mp4"
    feature_npz = directory / "trimamba_features.npz"

    set_status(
        job_id,
        state="running",
        stage="trimming",
        params={
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "lingbot_fps": lingbot_fps,
            "summary_ratio": summary_ratio,
            "max_points": max_points,
        },
    )
    end_label = f"{end_seconds:.3f}s" if end_seconds is not None else "end"
    append_log(job_id, f"Starting job with clip={start_seconds:.3f}s->{end_label}, lingbot_fps={lingbot_fps}")
    append_log(job_id, f"LingBot Python: {LINGBOT_PYTHON}")
    append_log(job_id, f"TriMamba Python: {TRIMAMBA_PYTHON}")
    append_log(job_id, "LingBot input: trimmed clip")

    try:
        trim_video(job_id, input_path, trimmed, start_seconds, end_seconds)
        refresh_artifacts(job_id)
    except Exception as exc:
        record_error(job_id, str(exc))
        append_log(job_id, traceback.format_exc())
        if should_stop(job_id):
            return
        set_status(job_id, state="failed", stage="failed")
        return

    lingbot_ok = False
    raw_ok = False
    post_ok = False
    if should_stop(job_id):
        return
    try:
        set_status(job_id, stage="lingbot_inference")
        run_command(
            job_id,
            [
                str(LINGBOT_PYTHON),
                str(MERGING_ROOT / "lingbot_runner.py"),
                "--model_path",
                str(MODEL_PATH),
                "--video_path",
                str(trimmed),
                "--output",
                str(raw_npz),
                "--fps",
                str(lingbot_fps),
            ],
            MERGING_ROOT,
            "LingBot-MAP inference",
        )
        lingbot_ok = raw_npz.exists()
        refresh_artifacts(job_id)
    except Exception as exc:
        record_error(job_id, str(exc))
        append_log(job_id, traceback.format_exc())
        if should_stop(job_id):
            return

    if lingbot_ok:
        if should_stop(job_id):
            return
        try:
            set_status(job_id, stage="raw_pointcloud")
            run_command(
                job_id,
                [
                    str(LINGBOT_PYTHON),
                    str(MERGING_ROOT / "pointcloud_tools.py"),
                    "--input",
                    str(raw_npz),
                    "--output",
                    str(raw_pcd),
                    "--conf_threshold",
                    "1.5",
                    "--frame_stride",
                    "2",
                    "--pixel_stride",
                    "6",
                    "--max_points",
                    str(max_points),
                ],
                MERGING_ROOT,
                "raw point cloud export",
            )
            raw_ok = raw_pcd.exists()
            refresh_artifacts(job_id)
        except Exception as exc:
            record_error(job_id, str(exc))
            append_log(job_id, traceback.format_exc())
            if should_stop(job_id):
                return

        if should_stop(job_id):
            return
        try:
            set_status(job_id, stage="postprocess_pointcloud")
            run_command(
                job_id,
                [
                    str(LINGBOT_PYTHON),
                    str(LINGBOT_ROOT / "postprocess_compare.py"),
                    "--input",
                    str(raw_npz),
                    "--out_dir",
                    str(directory),
                    "--variants",
                    "translation_only_damped",
                    "--max_points",
                    str(max_points),
                ],
                LINGBOT_ROOT,
                "translation-only damped postprocess",
            )
            post_ok = post_pcd.exists()
            refresh_artifacts(job_id)
        except Exception as exc:
            record_error(job_id, str(exc))
            append_log(job_id, traceback.format_exc())
            if should_stop(job_id):
                return

    summary_ok = False
    if should_stop(job_id):
        return
    try:
        set_status(job_id, stage="trimamba_summary")
        run_command(
            job_id,
            [
                str(TRIMAMBA_PYTHON),
                "scripts/summarize_mp4.py",
                "--input",
                str(trimmed),
                "--output",
                str(summary_mp4),
                "--config",
                "configs/mosu.yaml",
                "--ckpt",
                str(TRIMAMBA_CKPT),
                "--feature-npz",
                str(feature_npz),
                "--text-source",
                "zero",
                "--allow-missing-text",
                "--allow-missing-audio",
                "--summary-ratio",
                f"{summary_ratio:.4f}",
            ],
            TRIMAMBA_ROOT,
            "TriMamba summarization",
        )
        summary_ok = summary_mp4.exists()
        refresh_artifacts(job_id)
    except Exception as exc:
        record_error(job_id, str(exc))
        append_log(job_id, traceback.format_exc())
        if should_stop(job_id):
            return
        try:
            set_status(job_id, stage="fallback_summary")
            append_log(job_id, "TriMamba failed. Running OpenCV fallback summary.")
            run_command(
                job_id,
                [
                    str(LINGBOT_PYTHON),
                    str(MERGING_ROOT / "simple_summary.py"),
                    "--input",
                    str(trimmed),
                    "--output",
                    str(summary_mp4),
                    "--summary-ratio",
                    f"{summary_ratio:.4f}",
                    "--feature-npz",
                    str(feature_npz),
                ],
                MERGING_ROOT,
                "fallback video summarization",
            )
            summary_ok = summary_mp4.exists()
            refresh_artifacts(job_id)
        except Exception as fallback_exc:
            record_error(job_id, str(fallback_exc))
            append_log(job_id, traceback.format_exc())

    refresh_artifacts(job_id)
    has_errors = bool(JOBS.get(job_id, {}).get("errors"))
    if lingbot_ok and raw_ok and post_ok and summary_ok and not has_errors:
        state = "complete"
        stage = "complete"
    elif lingbot_ok or raw_ok or post_ok or summary_ok:
        state = "partial"
        stage = "partial"
    else:
        state = "failed"
        stage = "failed"
    set_status(job_id, state=state, stage=stage)
    append_log(job_id, f"Job finished with state={state}")


def run_job(job_id: str, params: dict) -> None:
    try:
        run_job_impl(job_id, params)
    finally:
        if should_stop(job_id):
            set_status(job_id, state="stopped", stage="stopped")
            append_log(job_id, "Job stopped.")
        mark_finished(job_id)


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MergingLocal/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{timestamp()}] {self.address_string()} {fmt % args}")

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(str(path))
        if path.suffix.lower() == ".pcd":
            mime = "application/octet-stream"
        data_len = path.stat().st_size
        start = 0
        end = max(0, data_len - 1)
        status = 200
        range_header = self.headers.get("Range", "")

        if range_header and data_len > 0:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                start_text, end_text = match.groups()
                if start_text == "" and end_text:
                    suffix_len = int(end_text)
                    start = max(0, data_len - suffix_len)
                else:
                    start = int(start_text or "0")
                    if end_text:
                        end = min(int(end_text), data_len - 1)
                if start >= data_len or start > end:
                    try:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{data_len}")
                        self.send_header("Accept-Ranges", "bytes")
                        self.end_headers()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                status = 206

        content_len = 0 if data_len == 0 else end - start + 1
        try:
            self.send_response(status)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_len))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{data_len}")
            self.end_headers()

            with path.open("rb") as handle:
                handle.seek(start)
                remaining = content_len
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            self.send_file(STATIC_ROOT / "index.html")
            return

        if path.startswith("/static/"):
            target = (STATIC_ROOT / path[len("/static/") :]).resolve()
            if STATIC_ROOT.resolve() not in target.parents and target != STATIC_ROOT.resolve():
                self.send_error(403)
                return
            self.send_file(target)
            return

        if path.startswith("/media/"):
            relative = Path(path[len("/media/") :])
            target = (DATA_ROOT / relative).resolve()
            if DATA_ROOT.resolve() not in target.parents:
                self.send_error(403)
                return
            self.send_file(target)
            return

        if path == "/api/jobs":
            self.send_json({"jobs": list_statuses()})
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/summary/analysis", path)
        if match:
            try:
                self.send_json(summary_analysis(match.group(1)))
            except FileNotFoundError as exc:
                self.send_json({"error": str(exc)}, status=409)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return

        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            status = read_status(job_id)
            if status is None:
                self.send_json({"error": "job not found"}, status=404)
                return
            self.send_json(status)
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/upload":
            self.handle_upload()
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/run", path)
        if match:
            self.handle_run(match.group(1))
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/stop", path)
        if match:
            self.handle_stop(match.group(1))
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/view/(raw|post)", path)
        if match:
            self.handle_viewer(match.group(1), match.group(2))
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/retrieval/index", path)
        if match:
            self.handle_retrieval_index(match.group(1))
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/retrieval/query", path)
        if match:
            self.handle_retrieval_query(match.group(1))
            return

        self.send_error(404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)", path)
        if not match:
            self.send_error(404)
            return
        force = "force=1" in (parsed.query or "")
        job_id = match.group(1)
        if job_id not in JOBS:
            self.send_json({"error": "job not found"}, status=404)
            return
        try:
            delete_job(job_id, force=force)
            self.send_json({"deleted": job_id})
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=409)

    def handle_upload(self) -> None:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        if "video" not in form:
            self.send_json({"error": "video field is required"}, status=400)
            return
        field = form["video"]
        if isinstance(field, list):
            field = field[0]
        filename = safe_filename(field.filename or "video.mp4")
        suffix = Path(filename).suffix.lower() or ".mp4"
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
            self.send_json({"error": f"unsupported video format: {suffix}. Allowed: {allowed}"}, status=400)
            return
        job_id = uuid.uuid4().hex[:12]
        directory = job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        input_path = directory / f"input{suffix}"
        with input_path.open("wb") as handle:
            shutil.copyfileobj(field.file, handle, length=1024 * 1024)

        duration = probe_duration(input_path)
        input_video_path = input_path
        if suffix in WEB_PREVIEW_SUFFIXES:
            try:
                preview_path = directory / "browser_preview.mp4"
                create_browser_preview(input_path, preview_path)
                input_video_path = preview_path
            except Exception as exc:
                shutil.rmtree(directory, ignore_errors=True)
                self.send_json({"error": f"could not convert {suffix} for browser preview: {exc}"}, status=400)
                return
        status = {
            "id": job_id,
            "state": "uploaded",
            "stage": "uploaded",
            "created_at": time.time(),
            "created_at_text": timestamp(),
            "updated_at": time.time(),
            "updated_at_text": timestamp(),
            "filename": filename,
            "duration": duration,
            "params": {},
            "errors": [],
            "artifacts": {"input_video_url": media_url(job_id, input_video_path)},
        }
        with STATUS_LOCK:
            JOBS[job_id] = status
            write_status(status)
        append_log(job_id, f"Uploaded {filename} -> {input_path}")
        self.send_json(read_status(job_id) or status)

    def handle_run(self, job_id: str) -> None:
        if job_id not in JOBS:
            self.send_json({"error": "job not found"}, status=404)
            return
        current = read_status(job_id)
        if current and current.get("is_running_local"):
            self.send_json(current, status=409)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            params = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self.send_json({"error": "invalid json body"}, status=400)
            return

        set_status(job_id, state="running", stage="queued", errors=[])
        mark_running(job_id)
        thread = threading.Thread(target=run_job, args=(job_id, params), daemon=True)
        thread.start()
        self.send_json(read_status(job_id) or {"id": job_id, "state": "running"})

    def handle_stop(self, job_id: str) -> None:
        if job_id not in JOBS:
            self.send_json({"error": "job not found"}, status=404)
            return
        stopped = request_stop(job_id)
        self.send_json({"stopping": stopped, "job": read_status(job_id)})

    def handle_viewer(self, job_id: str, kind: str) -> None:
        if job_id not in JOBS:
            self.send_json({"error": "job not found"}, status=404)
            return
        body = self.read_json_body()
        try:
            self.send_json(launch_viewer(job_id, kind, body))
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, status=409)
        except Exception as exc:
            append_log(job_id, traceback.format_exc())
            self.send_json({"error": str(exc)}, status=500)

    def handle_retrieval_index(self, job_id: str) -> None:
        if job_id not in JOBS:
            self.send_json({"error": "job not found"}, status=404)
            return
        body = self.read_json_body()
        try:
            start_retrieval_index(job_id, sample_fps=float(body.get("sample_fps", 1.0)))
            self.send_json(read_status(job_id) or {"id": job_id})
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, status=409)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def handle_retrieval_query(self, job_id: str) -> None:
        if job_id not in JOBS:
            self.send_json({"error": "job not found"}, status=404)
            return
        body = self.read_json_body()
        query = str(body.get("query", "")).strip()
        if not query:
            self.send_json({"error": "query is required"}, status=400)
            return
        try:
            self.send_json(query_retrieval(job_id, query, top_k=int(body.get("top_k", 8))))
        except FileNotFoundError as exc:
            self.send_json({"error": str(exc)}, status=409)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local merging website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    load_existing_jobs()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Local website running at {url}")
    print(f"Data directory: {DATA_ROOT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
