#!/usr/bin/env python3
"""Render DLP clips with a MetaDrive-like fixed topdown view.

This is a visual-domain adaptation probe. It deliberately avoids DLP's
agent-centric rotating crop and history trails: the camera is fixed for a clip,
the ego vehicle moves in the frame, and colors mimic the MetaDrive parking
wrapper used by plan_park_3_guided.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
DLP_ROOT = REPO_ROOT / "dlp_dataset"
if str(DLP_ROOT) not in sys.path:
    sys.path.insert(0, str(DLP_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dlp.dataset import Dataset  # noqa: E402
from dlp.visualizer import Visualizer  # noqa: E402
from tools.prepare_dlp_open_loop import (  # noqa: E402
    Candidate,
    complete_scene_prefixes,
    iter_candidates,
    load_scene,
    trajectory_for_agent,
)


RENDER_RES = 672
REAL_RES = 224

COLORS = {
    "background": (255, 255, 255),
    "parking_fill": (232, 232, 232),
    "parking_outline": (175, 175, 175),
    "obstacle": (100, 100, 100),
    "vehicle": (150, 150, 150),
    # RGB equivalents of the BGR colors in draw_vehicle_on_img().
    "ego_tail": (255, 128, 0),
    "ego_mid": (100, 100, 100),
    "ego_head": (100, 200, 255),
    "ego_outline": (255, 255, 255),
    "traj": (255, 128, 0),
}


def world_to_px(point: np.ndarray, center: np.ndarray, px_per_m: float, size: int) -> tuple[int, int]:
    rel = np.asarray(point[:2], dtype=np.float32) - center[:2]
    x = rel[0] * px_per_m + size / 2
    y = size / 2 - rel[1] * px_per_m
    return int(round(x)), int(round(y))


def box_corners(center_xy, dims, heading):
    length, width = dims
    offsets = np.array(
        [
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
        ],
        dtype=np.float32,
    )
    offsets *= np.array([length, width], dtype=np.float32)
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return offsets @ rot.T + np.asarray(center_xy, dtype=np.float32)


def draw_polygon(draw: ImageDraw.ImageDraw, corners, center, px_per_m, fill, outline=None, width=1):
    pts = [world_to_px(p, center, px_per_m, RENDER_RES) for p in corners]
    draw.polygon(pts, fill=fill)
    if outline is not None and width > 0:
        draw.line(pts + [pts[0]], fill=outline, width=width)


def draw_vehicle(draw: ImageDraw.ImageDraw, pose, dims, center, px_per_m, is_ego: bool):
    x, y, heading = float(pose[0]), float(pose[1]), float(pose[2])
    length, width = float(dims[0]), float(dims[1])
    if not is_ego:
        corners = box_corners([x, y], [length, width], heading)
        draw_polygon(draw, corners, center, px_per_m, fill=COLORS["vehicle"])
        return

    sections = [
        ((-0.5, -1.0 / 6.0), COLORS["ego_tail"]),
        ((-1.0 / 6.0, 1.0 / 6.0), COLORS["ego_mid"]),
        ((1.0 / 6.0, 0.5), COLORS["ego_head"]),
    ]
    c, s = math.cos(heading), math.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    origin = np.array([x, y], dtype=np.float32)
    for (x0_frac, x1_frac), color in sections:
        local = np.array(
            [
                [x1_frac * length, width / 2],
                [x1_frac * length, -width / 2],
                [x0_frac * length, -width / 2],
                [x0_frac * length, width / 2],
            ],
            dtype=np.float32,
        )
        corners = local @ rot.T + origin
        draw_polygon(draw, corners, center, px_per_m, fill=color)

    body = box_corners([x, y], [length, width], heading)
    pts = [world_to_px(p, center, px_per_m, RENDER_RES) for p in body]
    draw.line(pts + [pts[0]], fill=COLORS["ego_outline"], width=5)


def render_frame(
    ds: Dataset,
    vis: Visualizer,
    frame_token: str,
    ego_inst_token: str,
    camera_center: np.ndarray,
    px_per_m: float,
    draw_future_traj: np.ndarray | None = None,
) -> np.ndarray:
    img = Image.new("RGB", (RENDER_RES, RENDER_RES), COLORS["background"])
    draw = ImageDraw.Draw(img)

    # Parking spots give the network stable map structure without using DLP's
    # green spot color or occupancy/history rendering.
    for _, p in vis.parking_spaces.iterrows():
        corners = p[2:10].to_numpy(dtype=np.float32).reshape((4, 2))
        draw_polygon(
            draw,
            corners,
            camera_center,
            px_per_m,
            fill=COLORS["parking_fill"],
            outline=COLORS["parking_outline"],
            width=2,
        )

    frame = ds.get("frame", frame_token)
    scene = ds.get("scene", frame["scene_token"])
    for obstacle_token in scene["obstacles"]:
        obstacle = ds.get("obstacle", obstacle_token)
        corners = vis._get_corners(obstacle["coords"], obstacle["size"], obstacle["heading"])
        draw_polygon(draw, corners, camera_center, px_per_m, fill=COLORS["obstacle"])

    if draw_future_traj is not None and len(draw_future_traj) > 1:
        pts = [world_to_px(p, camera_center, px_per_m, RENDER_RES) for p in draw_future_traj]
        draw.line(pts, fill=COLORS["traj"], width=4)

    for inst_token in frame["instances"]:
        inst = ds.get("instance", inst_token)
        agent = ds.get("agent", inst["agent_token"])
        if agent["type"] in {"Pedestrian", "Bicycle", "Undefined"}:
            continue
        pose = [inst["coords"][0], inst["coords"][1], inst["heading"]]
        draw_vehicle(
            draw,
            pose=pose,
            dims=agent["size"],
            center=camera_center,
            px_per_m=px_per_m,
            is_ego=(inst_token == ego_inst_token),
        )

    arr = np.asarray(img)
    arr = cv2.resize(arr, (REAL_RES, REAL_RES), interpolation=cv2.INTER_AREA)
    return arr


def camera_center_for(traj: np.ndarray, mode: str) -> np.ndarray:
    if mode == "start":
        return traj[0, :2].astype(np.float32)
    if mode == "midpoint":
        return ((traj[0, :2] + traj[-1, :2]) / 2.0).astype(np.float32)
    if mode == "mean":
        return traj[:, :2].mean(axis=0).astype(np.float32)
    raise ValueError(f"Unknown camera center mode: {mode}")


def export_clip(
    ds: Dataset,
    candidate: Candidate,
    out_dir: Path,
    horizon_frames: int,
    sample_stride: int,
    px_per_m: float,
    camera_center_mode: str,
    fps: int,
    overlay_traj: bool,
) -> None:
    instances, traj_full = trajectory_for_agent(ds, candidate.agent_token)
    raw_indices = np.arange(candidate.start_index, candidate.end_index + 1, sample_stride)
    if raw_indices[-1] != candidate.end_index:
        raw_indices = np.append(raw_indices, candidate.end_index)
    sampled_instances = [instances[int(i)] for i in raw_indices]
    sampled_traj = traj_full[raw_indices]
    camera_center = camera_center_for(sampled_traj, camera_center_mode)

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    vis = Visualizer(ds)

    frames = []
    for idx, inst in enumerate(sampled_instances):
        future = sampled_traj[idx:, :2] if overlay_traj else None
        frame = render_frame(
            ds,
            vis,
            frame_token=inst["frame_token"],
            ego_inst_token=inst["instance_token"],
            camera_center=camera_center,
            px_per_m=px_per_m,
            draw_future_traj=future,
        )
        frames.append(frame)
        Image.fromarray(frame).save(frame_dir / f"{idx:03d}.png")

    imageio.mimsave(out_dir / "clip.mp4", frames, fps=fps, macro_block_size=None)
    imageio.mimsave(out_dir / "clip.gif", frames, fps=fps)
    Image.fromarray(np.concatenate(frames[: min(len(frames), 8)], axis=1)).save(out_dir / "strip.png")

    metadata = asdict(candidate)
    metadata.update(
        {
            "horizon_frames": horizon_frames,
            "sample_stride": sample_stride,
            "camera_center_mode": camera_center_mode,
            "camera_center_xy": camera_center.tolist(),
            "px_per_m_at_672": px_per_m,
            "px_per_m_at_224": px_per_m / 3.0,
            "approx_crop_side_m": RENDER_RES / px_per_m,
            "draw_history": False,
            "rotating_agent_centric": False,
            "overlay_future_traj": overlay_traj,
        }
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=REPO_ROOT / "dlp_dataset/data", type=Path)
    parser.add_argument("--output-root", default=REPO_ROOT / "tools/outputs/dlp_metadrive_style", type=Path)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--horizon-frames", type=int, default=100)
    parser.add_argument("--search-stride", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--min-path-length", type=float, default=1.5)
    parser.add_argument("--max-mean-speed", type=float, default=4.0)
    parser.add_argument("--min-heading-change", type=float, default=0.05)
    parser.add_argument("--px-per-m", type=float, default=24.0)
    parser.add_argument("--camera-center", choices=["start", "midpoint", "mean"], default="midpoint")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--overlay-traj", action="store_true")
    args = parser.parse_args()

    scene_prefixes = args.scene or complete_scene_prefixes(args.data_root)
    all_candidates = []
    for prefix in scene_prefixes:
        print(f"[DLP render] loading {prefix}")
        ds = load_scene(args.data_root, prefix)
        candidates = list(
            iter_candidates(
                ds,
                scene_prefix=prefix,
                min_len=args.min_len,
                horizon_frames=args.horizon_frames,
                search_stride=args.search_stride,
                min_path_length=args.min_path_length,
                max_mean_speed=args.max_mean_speed,
                min_heading_change=args.min_heading_change,
            )
        )
        print(f"[DLP render] {prefix}: {len(candidates)} candidates")
        for candidate in candidates:
            all_candidates.append((candidate, ds))

    all_candidates.sort(key=lambda item: item[0].score, reverse=True)
    selected = all_candidates[: args.top_k]
    if not selected:
        print("[DLP render] no suitable candidates")
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    index = []
    for rank, (candidate, ds) in enumerate(selected, start=1):
        out_dir = args.output_root / f"{rank:02d}_{candidate.scene_prefix}_{candidate.agent_token[:8]}_{candidate.start_index:05d}"
        print(
            f"[DLP render] export #{rank}: {candidate.scene_prefix} "
            f"agent={candidate.agent_token[:8]} path={candidate.path_length_m:.2f}m "
            f"turn={candidate.heading_change_rad:.2f}rad speed={candidate.mean_abs_speed:.2f}m/s"
        )
        export_clip(
            ds,
            candidate,
            out_dir=out_dir,
            horizon_frames=args.horizon_frames,
            sample_stride=args.sample_stride,
            px_per_m=args.px_per_m,
            camera_center_mode=args.camera_center,
            fps=args.fps,
            overlay_traj=args.overlay_traj,
        )
        index.append(asdict(candidate) | {"output_dir": str(out_dir)})

    (args.output_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[DLP render] wrote {len(selected)} clips to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
