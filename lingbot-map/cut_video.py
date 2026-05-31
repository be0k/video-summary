#!/usr/bin/env python3
"""Remove the first N seconds from a video and save the remaining tail as MP4.

Example:
    python cut_video.py input.mp4 12 output_without_first_12s.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def remove_first_seconds(input_path: Path, seconds: float, output_path: Path) -> None:
    if seconds <= 0:
        raise ValueError("seconds must be greater than 0")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(seconds),
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove the first N seconds of an input video and save the rest."
    )
    parser.add_argument("input_video", type=Path, help="Input video path")
    parser.add_argument("seconds", type=float, help="Number of seconds to remove from the front")
    parser.add_argument("output_video", type=Path, help="Output MP4 path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    remove_first_seconds(args.input_video, args.seconds, args.output_video)
    print(f"Saved: {args.output_video}")


if __name__ == "__main__":
    main()
