#!/usr/bin/env python3
"""Visualize an exported PCD/PLY point cloud in a browser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def load_point_cloud(path: Path, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        raise SystemExit(f"No points loaded from {path}")

    points = np.asarray(point_cloud.points, dtype=np.float32)
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors, dtype=np.float32)
    else:
        colors = np.full_like(points, 0.72, dtype=np.float32)

    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(points), size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        colors = colors[keep]
    return points, np.clip(colors, 0.0, 1.0)


def visualize_point_cloud(path: Path, port: int, point_size: float, max_points: int) -> None:
    import viser

    points, colors = load_point_cloud(path, max_points=max_points)
    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    point_cloud = server.scene.add_point_cloud(
        "/point_cloud",
        points=points,
        colors=(colors * 255).astype(np.uint8),
        point_size=point_size,
        point_shape="circle",
    )
    center = np.mean(points, axis=0)
    extent = np.ptp(points, axis=0)
    scene_scale = max(float(np.linalg.norm(extent)), 0.1)
    axes = server.scene.add_frame(
        "/scene_center",
        position=center,
        axes_length=max(scene_scale * 0.12, 0.05),
        axes_radius=max(scene_scale * 0.002, 0.001),
        visible=True,
    )

    def set_client_view(client, direction: np.ndarray, up: np.ndarray, distance_scale: float = 1.8) -> None:
        direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
        client.camera.up_direction = tuple(up)
        client.camera.position = tuple(center + direction * scene_scale * distance_scale)
        client.camera.look_at = tuple(center)

    def reset_view(direction: np.ndarray, up: np.ndarray = np.array([0.0, -1.0, 0.0])) -> None:
        for client in server.get_clients().values():
            set_client_view(client, direction, up)

    @server.on_client_connect
    def _(client) -> None:
        set_client_view(
            client,
            np.array([0.5, -0.6, 0.6], dtype=np.float32),
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
        )

    with server.gui.add_folder("Point Cloud"):
        gui_visible = server.gui.add_checkbox("Show Point Cloud", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.2,
            step=0.001,
            initial_value=float(np.clip(point_size, 0.001, 0.2)),
        )
        gui_cloud_scale = server.gui.add_slider(
            "Point Cloud Scale",
            min=0.2,
            max=3.0,
            step=0.05,
            initial_value=1.0,
        )
        gui_point_shape = server.gui.add_dropdown(
            "Point Shape",
            options=("circle", "rounded", "square", "diamond", "sparkle"),
            initial_value="circle",
        )
        gui_point_shading = server.gui.add_dropdown(
            "Point Shading",
            options=("flat", "gradient"),
            initial_value="flat",
        )
        server.gui.add_text("Loaded Points", f"{len(points):,}", disabled=True)

    with server.gui.add_folder("Reset View Direction"):
        btn_center = server.gui.add_button("Look At Scene Center")
        btn_overview = server.gui.add_button("Overview")
        btn_front = server.gui.add_button("Front (+Z)")
        btn_back = server.gui.add_button("Back (-Z)")
        btn_top = server.gui.add_button("Top (-Y)")
        btn_left = server.gui.add_button("Left (-X)")
        btn_right = server.gui.add_button("Right (+X)")

    with server.gui.add_folder("Scene Helpers"):
        gui_show_axes = server.gui.add_checkbox("Show Center Axes", initial_value=True)

    @gui_visible.on_update
    def _(_) -> None:
        point_cloud.visible = gui_visible.value

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    @gui_cloud_scale.on_update
    def _(_) -> None:
        point_cloud.scale = gui_cloud_scale.value

    @gui_point_shape.on_update
    def _(_) -> None:
        point_cloud.point_shape = gui_point_shape.value

    @gui_point_shading.on_update
    def _(_) -> None:
        point_cloud.point_shading = gui_point_shading.value

    @gui_show_axes.on_update
    def _(_) -> None:
        axes.visible = gui_show_axes.value

    @btn_center.on_click
    def _(_) -> None:
        for client in server.get_clients().values():
            client.camera.look_at = tuple(center)

    @btn_overview.on_click
    def _(_) -> None:
        reset_view(np.array([0.5, -0.6, 0.6], dtype=np.float32))

    @btn_front.on_click
    def _(_) -> None:
        reset_view(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    @btn_back.on_click
    def _(_) -> None:
        reset_view(np.array([0.0, 0.0, -1.0], dtype=np.float32))

    @btn_top.on_click
    def _(_) -> None:
        reset_view(
            np.array([0.0, -1.0, 0.0], dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    @btn_left.on_click
    def _(_) -> None:
        reset_view(np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    @btn_right.on_click
    def _(_) -> None:
        reset_view(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    print(f"Loaded {len(points):,} points from {path}")
    print(f"Bounds: center={center.round(3).tolist()}, extent={extent.round(3).tolist()}")
    print(f"3D viewer at http://localhost:{port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopped viewer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize an exported PCD/PLY point cloud.")
    parser.add_argument("input", type=Path, help="Point cloud path, e.g. outputs/postprocess_compare_tsdf/tsdf_single_icp.pcd")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.03)
    parser.add_argument("--max_points", type=int, default=1_000_000, help="Random cap. Use 0 for all points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = args.input.suffix.lower()
    if suffix not in {".pcd", ".ply"}:
        raise SystemExit("Input must be .pcd or .ply")
    visualize_point_cloud(
        args.input,
        port=args.port,
        point_size=args.point_size,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
