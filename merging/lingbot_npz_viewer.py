#!/usr/bin/env python3
"""Launch the LingBot-MAP demo-style Viser viewer from a saved prediction NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MERGING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MERGING_ROOT.parent
LINGBOT_ROOT = PROJECT_ROOT / "lingbot-map"
if str(LINGBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_ROOT))

from lingbot_map.vis import PointCloudViewer  # noqa: E402


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open LingBot-MAP NPZ in the demo-style viewer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--depth_stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.input)
    viewer = PointCloudViewer(
        pred_dict=predictions,
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        depth_stride=args.depth_stride,
    )
    print(f"LingBot demo-style viewer at http://127.0.0.1:{args.port}")
    viewer.run()


if __name__ == "__main__":
    main()
