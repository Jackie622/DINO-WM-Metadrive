#!/usr/bin/env python3
"""Prepare small DLP open-loop prediction samples for DINO-WM inspection.

The script selects moving vehicle agents, crops an agent-centric semantic BEV
around the chosen car, overlays its future DLP trajectory, and saves a compact
NPZ with pose and pseudo-control arrays. It intentionally does not call the WM
yet; this keeps the first DLP step focused on data correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DLP_ROOT = REPO_ROOT / "dlp_dataset"
if str(DLP_ROOT) not in sys.path:
    sys.path.insert(0, str(DLP_ROOT))

from dlp.dataset import Dataset  # noqa: E402
from dlp.visualizer import SemanticVisualizer  # noqa: E402


VEHICLE_TYPES = {"Car", "Medium Vehicle", "Bus"}
REQUIRED_SUFFIXES = [
    "_agents.json",
    "_frames.json",
    "_instances.json",
    "_obstacles.json",
    "_scene.json",
]


@dataclass
class Candidate:
    scene_prefix: str
    scene_token: str
    agent_token: str
    agent_type: str
    start_index: int
    end_index: int
    num_instances: int
    score: float
    displacement_m: float
    path_length_m: float
    heading_change_rad: float
    mean_abs_speed: float
    reverse_ratio: float
    start_instance: str
    start_frame: str


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def complete_scene_prefixes(data_root: Path) -> list[str]:
    prefixes = []
    for scene_path in sorted(data_root.glob("*_scene.json")):
        prefix = scene_path.name[: -len("_scene.json")]
        if all((data_root / f"{prefix}{suffix}").exists() for suffix in REQUIRED_SUFFIXES):
            prefixes.append(prefix)
    return prefixes


def load_scene(data_root: Path, prefix: str) -> Dataset:
    ds = Dataset()
    ds.load(str(data_root / prefix))
    return ds


def trajectory_for_agent(ds: Dataset, agent_token: str) -> tuple[list[dict], np.ndarray]:
    instances = ds.get_agent_instances(agent_token)
    traj = []
    for inst in instances:
        signed_speed = ds.signed_speed(inst["instance_token"])
        traj.append(
            [
                inst["coords"][0],
                inst["coords"][1],
                inst["heading"],
                signed_speed,
            ]
        )
    return instances, np.asarray(traj, dtype=np.float32)


def pseudo_controls(traj: np.ndarray, dt: float, wheelbase: float) -> np.ndarray:
    """Approximate physical steer/throttle from a downsampled pose trajectory."""
    if len(traj) < 2:
        return np.zeros((0, 2), dtype=np.float32)

    xy = traj[:, :2]
    heading = traj[:, 2]
    speed = traj[:, 3]
    delta_xy = np.diff(xy, axis=0)
    ds = np.linalg.norm(delta_xy, axis=1)
    dtheta = angle_diff(heading[1:], heading[:-1])
    curvature = np.divide(dtheta, np.maximum(ds, 1e-3))
    steer = np.arctan(wheelbase * curvature)
    accel = np.diff(speed) / max(dt, 1e-6)
    return np.stack([steer, accel], axis=-1).astype(np.float32)


def score_window(traj_window: np.ndarray) -> tuple[float, dict[str, float]]:
    xy = traj_window[:, :2]
    heading = traj_window[:, 2]
    speed = traj_window[:, 3]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    path_length = float(np.sum(steps))
    heading_change = float(np.sum(np.abs(angle_diff(heading[1:], heading[:-1]))))
    mean_abs_speed = float(np.mean(np.abs(speed)))
    reverse_ratio = float(np.mean(speed < -0.05))

    # Prefer real parking-like motion: local displacement, nontrivial progress,
    # some curvature or reversing, and avoid high-speed drive-through snippets.
    speed_penalty = max(0.0, mean_abs_speed - 3.0)
    score = (
        1.4 * min(path_length, 25.0)
        + 8.0 * min(heading_change, math.pi)
        + 4.0 * reverse_ratio
        + 0.4 * min(displacement, 15.0)
        - 2.0 * speed_penalty
    )
    return score, {
        "displacement_m": displacement,
        "path_length_m": path_length,
        "heading_change_rad": heading_change,
        "mean_abs_speed": mean_abs_speed,
        "reverse_ratio": reverse_ratio,
    }


def iter_candidates(
    ds: Dataset,
    scene_prefix: str,
    min_len: int,
    horizon_frames: int,
    search_stride: int,
    min_path_length: float,
    max_mean_speed: float,
    min_heading_change: float,
) -> Iterable[Candidate]:
    scene_token = ds.list_scenes()[0]
    scene = ds.get("scene", scene_token)

    for agent_token in scene["agents"]:
        agent = ds.get("agent", agent_token)
        if agent["type"] not in VEHICLE_TYPES:
            continue

        instances, traj = trajectory_for_agent(ds, agent_token)
        if len(instances) < max(min_len, horizon_frames + 1):
            continue

        best: Candidate | None = None
        max_start = len(instances) - horizon_frames - 1
        for start in range(0, max_start + 1, search_stride):
            end = start + horizon_frames
            window = traj[start : end + 1]
            score, stats = score_window(window)
            if stats["path_length_m"] < min_path_length:
                continue
            if stats["mean_abs_speed"] > max_mean_speed:
                continue
            if stats["heading_change_rad"] < min_heading_change and stats["reverse_ratio"] < 0.05:
                continue
            candidate = Candidate(
                scene_prefix=scene_prefix,
                scene_token=scene_token,
                agent_token=agent_token,
                agent_type=agent["type"],
                start_index=start,
                end_index=end,
                num_instances=len(instances),
                score=float(score),
                start_instance=instances[start]["instance_token"],
                start_frame=instances[start]["frame_token"],
                **stats,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best is not None:
            yield best


def save_candidate(
    ds: Dataset,
    candidate: Candidate,
    output_dir: Path,
    horizon_frames: int,
    sample_stride: int,
    resolution: float,
    sensing_limit: float,
    resize: int,
    fps: float,
    wheelbase: float,
    save_sequence_images: bool,
) -> None:
    instances, traj_full = trajectory_for_agent(ds, candidate.agent_token)
    raw_indices = np.arange(candidate.start_index, candidate.end_index + 1, sample_stride)
    if raw_indices[-1] != candidate.end_index:
        raw_indices = np.append(raw_indices, candidate.end_index)

    sampled_instances = [instances[int(i)] for i in raw_indices]
    sampled_traj = traj_full[raw_indices]
    controls = pseudo_controls(sampled_traj, dt=sample_stride / fps, wheelbase=wheelbase)

    output_dir.mkdir(parents=True, exist_ok=True)
    vis = SemanticVisualizer(
        ds,
        spot_margin=0.3,
        resolution=resolution,
        sensing_limit=sensing_limit,
        steps=5,
        stride=max(1, sample_stride),
    )

    start_inst = sampled_instances[0]
    start_pose = sampled_traj[0, :3]
    frame_img = vis.plot_frame(start_inst["frame_token"])
    start_bev = vis.inst_centric(frame_img, start_inst["instance_token"], center_pose=start_pose)
    overlay = vis.plot_traj(
        inst_centric_view=start_bev,
        center_pose=start_pose,
        traj=sampled_traj,
        color=(255, 255, 255),
        width=max(2, int(round(0.4 / resolution))),
    )

    if resize > 0:
        size = (resize, resize)
        start_bev = start_bev.resize(size, Image.Resampling.NEAREST)
        overlay = overlay.resize(size, Image.Resampling.NEAREST)

    start_bev.save(output_dir / "start_bev.png")
    overlay.save(output_dir / "future_overlay.png")

    if save_sequence_images:
        seq_dir = output_dir / "sequence_bev"
        seq_dir.mkdir(exist_ok=True)
        for idx, (inst, pose) in enumerate(zip(sampled_instances, sampled_traj[:, :3])):
            img_frame = vis.plot_frame(inst["frame_token"])
            img = vis.inst_centric(img_frame, inst["instance_token"], center_pose=pose)
            if resize > 0:
                img = img.resize((resize, resize), Image.Resampling.NEAREST)
            img.save(seq_dir / f"{idx:03d}.png")

    np.savez_compressed(
        output_dir / "sample.npz",
        traj=sampled_traj.astype(np.float32),
        pseudo_actions=controls.astype(np.float32),
        raw_indices=raw_indices.astype(np.int32),
        instance_tokens=np.asarray([inst["instance_token"] for inst in sampled_instances]),
        frame_tokens=np.asarray([inst["frame_token"] for inst in sampled_instances]),
    )

    metadata = asdict(candidate)
    metadata.update(
        {
            "horizon_frames": horizon_frames,
            "sample_stride": sample_stride,
            "sampled_steps": int(len(sampled_traj)),
            "dt_sampled_sec": sample_stride / fps,
            "resolution_m_per_px": resolution,
            "sensing_limit_m": sensing_limit,
            "crop_side_m": 2 * sensing_limit,
            "output_image_size": resize if resize > 0 else int(2 * sensing_limit / resolution),
            "pseudo_action_columns": ["steer_rad", "accel_mps2"],
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=REPO_ROOT / "dlp_dataset/data", type=Path)
    parser.add_argument("--output-root", default=REPO_ROOT / "dlp_open_loop_samples", type=Path)
    parser.add_argument("--scene", action="append", help="Scene prefix such as DJI_0001. Repeatable.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--horizon-frames", type=int, default=75, help="Raw DLP frames, 75 frames is 3s at 25Hz.")
    parser.add_argument("--search-stride", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=5, help="Downsample raw 25Hz DLP to 5Hz by default.")
    parser.add_argument("--min-path-length", type=float, default=1.5)
    parser.add_argument("--max-mean-speed", type=float, default=4.0)
    parser.add_argument(
        "--min-heading-change",
        type=float,
        default=0.05,
        help="Radians over the raw horizon. Reversing snippets can pass even below this.",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--wheelbase", type=float, default=2.8)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--sensing-limit", type=float, default=20.0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--save-sequence-images", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root
    scene_prefixes = args.scene or complete_scene_prefixes(data_root)
    if not scene_prefixes:
        print(f"[DLP] no complete scenes found under {data_root}")
        return 1

    all_candidates: list[tuple[Candidate, Dataset]] = []
    for prefix in scene_prefixes:
        print(f"[DLP] loading {prefix}")
        ds = load_scene(data_root, prefix)
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
        print(f"[DLP] {prefix}: {len(candidates)} vehicle candidates")
        for candidate in candidates:
            all_candidates.append((candidate, ds))

    all_candidates.sort(key=lambda item: item[0].score, reverse=True)
    selected = all_candidates[: args.top_k]
    if not selected:
        print("[DLP] no suitable moving vehicle windows found")
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    index = []
    for rank, (candidate, ds) in enumerate(selected, start=1):
        out_dir = args.output_root / f"{rank:02d}_{candidate.scene_prefix}_{candidate.agent_token[:8]}_{candidate.start_index:05d}"
        print(
            "[DLP] export "
            f"#{rank}: {candidate.scene_prefix} agent={candidate.agent_token[:8]} "
            f"score={candidate.score:.2f} path={candidate.path_length_m:.2f}m "
            f"turn={candidate.heading_change_rad:.2f}rad speed={candidate.mean_abs_speed:.2f}m/s"
        )
        save_candidate(
            ds,
            candidate,
            out_dir,
            horizon_frames=args.horizon_frames,
            sample_stride=args.sample_stride,
            resolution=args.resolution,
            sensing_limit=args.sensing_limit,
            resize=args.resize,
            fps=args.fps,
            wheelbase=args.wheelbase,
            save_sequence_images=args.save_sequence_images,
        )
        index.append(asdict(candidate) | {"output_dir": str(out_dir)})

    (args.output_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[DLP] wrote {len(selected)} samples to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
