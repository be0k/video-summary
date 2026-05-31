#!/usr/bin/env python3
"""Combine two videos into one horizontal side-by-side MP4.

Example:
    python side_by_side_video.py left.mp4 right.mp4 output_side_by_side.mp4
"""

import argparse
import shutil
import subprocess
from pathlib import Path


def combine_side_by_side(
    left_video: Path,
    right_video: Path,
    output_video: Path,
    height: int,
    audio: str,
) -> None:
    if not left_video.is_file():
        raise FileNotFoundError(f"Left video not found: {left_video}")
    if not right_video.is_file():
        raise FileNotFoundError(f"Right video not found: {right_video}")
    if height <= 0:
        raise ValueError("height must be greater than 0")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    filter_graph = (
        f"[0:v]scale=-2:{height},setsar=1,setpts=PTS-STARTPTS[left];"
        f"[1:v]scale=-2:{height},setsar=1,setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2:shortest=1[v]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(left_video),
        "-i",
        str(right_video),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
    ]

    if audio == "left":
        cmd += ["-map", "0:a?"]
    elif audio == "right":
        cmd += ["-map", "1:a?"]

    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_video),
    ]

    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Put two videos next to each other horizontally and save one MP4."
    )
    parser.add_argument("left_video", type=Path, help="Video shown on the left")
    parser.add_argument("right_video", type=Path, help="Video shown on the right")
    parser.add_argument("output_video", type=Path, help="Output MP4 path")
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Output height of each video before stacking. Default: 720",
    )
    parser.add_argument(
        "--audio",
        choices=["left", "right", "none"],
        default="left",
        help="Which video's audio to keep. Default: left",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_side_by_side(
        left_video=args.left_video,
        right_video=args.right_video,
        output_video=args.output_video,
        height=args.height,
        audio=args.audio,
    )
    print(f"Saved: {args.output_video}")


if __name__ == "__main__":
    main()
