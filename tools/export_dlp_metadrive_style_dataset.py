#!/usr/bin/env python3
"""Export DLP clips as MetaDrive-style npz episodes for DINO-WM probes.

This is stage-0 data preparation only: no model training happens here.  The
script renders a fixed-camera top-down semantic view, infers pseudo controls
from DLP annotations, and writes npz files compatible with data_loader_park.py:

    image:  (T, 224, 224, 3) uint8
    action: (T, 2) float32, [steer_rad, accel_mps2] with final action padded
    state:  (T, 3) float32, [signed_speed, steer_rad, 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DLP_ROOT = REPO_ROOT / "dlp_dataset"
for path in (REPO_ROOT, DLP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dlp.visualizer import Visualizer  # noqa: E402
from tools.prepare_dlp_open_loop import (  # noqa: E402
    complete_scene_prefixes,
    iter_candidates,
    load_scene,
    pseudo_controls,
    trajectory_for_agent,
)
from tools.render_dlp_metadrive_style import camera_center_for, render_frame  # noqa: E402


def pad_actions(controls: np.ndarray, num_frames: int) -> np.ndarray:
    """Pad interval controls to one action row per frame."""
    if len(controls) == 0:
        return np.zeros((num_frames, 2), dtype=np.float32)
    if len(controls) >= num_frames:
        return controls[:num_frames].astype(np.float32)
    return np.concatenate([controls, controls[-1:]], axis=0).astype(np.float32)


def build_state(traj: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state = np.zeros((len(traj), 3), dtype=np.float32)
    state[:, 0] = traj[:, 3]
    state[:, 1] = actions[:, 0]
    return state


def save_preview(frames: np.ndarray, out_dir: Path, fps: int, every: int) -> None:
    if every <= 0:
        return
    preview_dir = out_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sampled = frames[:: max(1, every)]
    if len(sampled) == 0:
        return
    imageio.mimsave(preview_dir / "clip.mp4", list(sampled), fps=fps, macro_block_size=None)
    imageio.mimsave(preview_dir / "clip.gif", list(sampled), fps=fps)
    strip = np.concatenate(sampled[: min(len(sampled), 10)], axis=1)
    Image.fromarray(strip).save(preview_dir / "strip.png")


def export_candidate(
    ds,
    candidate,
    out_dir: Path,
    episode_name: str,
    raw_stride: int,
    fps: float,
    wheelbase: float,
    px_per_m: float,
    camera_center_mode: str,
    preview_every: int,
    preview_fps: int,
) -> dict:
    instances, traj_full = trajectory_for_agent(ds, candidate.agent_token)
    raw_indices = np.arange(candidate.start_index, candidate.end_index + 1, raw_stride)
    if raw_indices[-1] != candidate.end_index:
        raw_indices = np.append(raw_indices, candidate.end_index)

    sampled_instances = [instances[int(i)] for i in raw_indices]
    traj = traj_full[raw_indices].astype(np.float32)
    controls = pseudo_controls(traj, dt=raw_stride / fps, wheelbase=wheelbase)
    actions = pad_actions(controls, len(traj))
    states = build_state(traj, actions)

    camera_center = camera_center_for(traj, camera_center_mode)
    vis = Visualizer(ds)
    frames = []
    for inst in sampled_instances:
        frames.append(
            render_frame(
                ds,
                vis,
                frame_token=inst["frame_token"],
                ego_inst_token=inst["instance_token"],
                camera_center=camera_center,
                px_per_m=px_per_m,
                draw_future_traj=None,
            )
        )
    images = np.stack(frames, axis=0).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{episode_name}.npz"
    np.savez_compressed(
        npz_path,
        image=images,
        action=actions,
        state=states,
        traj=traj,
        raw_indices=raw_indices.astype(np.int32),
        instance_tokens=np.asarray([inst["instance_token"] for inst in sampled_instances]),
        frame_tokens=np.asarray([inst["frame_token"] for inst in sampled_instances]),
    )
    save_preview(images, out_dir / episode_name, fps=preview_fps, every=preview_every)

    metadata = asdict(candidate)
    metadata.update(
        {
            "episode": episode_name,
            "npz_path": str(npz_path),
            "num_frames": int(len(images)),
            "raw_stride": int(raw_stride),
            "dt_sec": float(raw_stride / fps),
            "fps_source_assumed": float(fps),
            "wheelbase_m": float(wheelbase),
            "camera_center_mode": camera_center_mode,
            "camera_center_xy": camera_center.tolist(),
            "px_per_m_at_672": float(px_per_m),
            "image_shape": list(images.shape),
            "action_shape": list(actions.shape),
            "state_shape": list(states.shape),
            "action_columns": ["steer_rad", "accel_mps2"],
            "state_columns": ["signed_speed_mps", "steer_rad", "reserved_zero"],
        }
    )
    episode_dir = out_dir / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def summarize_npz_files(npz_paths: list[Path]) -> dict:
    action_chunks = []
    state_chunks = []
    lengths = []
    image_shapes = []
    for path in npz_paths:
        with np.load(path) as data:
            action_chunks.append(data["action"].astype(np.float32))
            state_chunks.append(data["state"].astype(np.float32))
            lengths.append(int(data["image"].shape[0]))
            image_shapes.append(list(data["image"].shape))

    actions = np.concatenate(action_chunks, axis=0) if action_chunks else np.zeros((0, 2), dtype=np.float32)
    states = np.concatenate(state_chunks, axis=0) if state_chunks else np.zeros((0, 3), dtype=np.float32)
    return {
        "num_episodes": len(npz_paths),
        "length_min_mean_max": [
            int(np.min(lengths)) if lengths else 0,
            float(np.mean(lengths)) if lengths else 0.0,
            int(np.max(lengths)) if lengths else 0,
        ],
        "image_shapes_first": image_shapes[:5],
        "action_mean": actions.mean(axis=0).tolist() if len(actions) else [0.0, 0.0],
        "action_std": actions.std(axis=0).tolist() if len(actions) else [0.0, 0.0],
        "action_min": actions.min(axis=0).tolist() if len(actions) else [0.0, 0.0],
        "action_max": actions.max(axis=0).tolist() if len(actions) else [0.0, 0.0],
        "state_mean": states.mean(axis=0).tolist() if len(states) else [0.0, 0.0, 0.0],
        "state_std": states.std(axis=0).tolist() if len(states) else [0.0, 0.0, 0.0],
        "abs_steer_gt_0.3_ratio": float(np.mean(np.abs(actions[:, 0]) > 0.3)) if len(actions) else 0.0,
        "abs_steer_gt_0.5_ratio": float(np.mean(np.abs(actions[:, 0]) > 0.5)) if len(actions) else 0.0,
        "abs_accel_gt_1.0_ratio": float(np.mean(np.abs(actions[:, 1]) > 1.0)) if len(actions) else 0.0,
        "reverse_state_ratio": float(np.mean(states[:, 0] < -0.05)) if len(states) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=REPO_ROOT / "dlp_dataset/data", type=Path)
    parser.add_argument("--output-root", default=REPO_ROOT / "tools/outputs/dlp_metadrive_npz_stage0", type=Path)
    parser.add_argument("--scene", action="append", help="Scene prefix such as DJI_0001. Repeatable.")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--per-scene-top-k",
        type=int,
        default=0,
        help="If >0, keep at most this many candidates per scene before global top-k selection.",
    )
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--horizon-frames", type=int, default=100)
    parser.add_argument("--search-stride", type=int, default=10)
    parser.add_argument("--raw-stride", type=int, default=1)
    parser.add_argument("--min-path-length", type=float, default=1.5)
    parser.add_argument("--max-mean-speed", type=float, default=4.0)
    parser.add_argument("--min-heading-change", type=float, default=0.05)
    parser.add_argument("--px-per-m", type=float, default=24.0)
    parser.add_argument("--camera-center", choices=["start", "midpoint", "mean"], default="midpoint")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--wheelbase", type=float, default=2.8)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-fps", type=int, default=5)
    parser.add_argument("--skip-preview", action="store_true", help="Do not write gif/mp4/strip previews.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing episode npz files and export only missing ones.")
    args = parser.parse_args()

    scene_prefixes = args.scene or complete_scene_prefixes(args.data_root)
    all_candidates = []
    for prefix in scene_prefixes:
        print(f"[DLP npz] loading {prefix}")
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
        candidates.sort(key=lambda item: item.score, reverse=True)
        if args.per_scene_top_k > 0:
            candidates = candidates[: args.per_scene_top_k]
        print(f"[DLP npz] {prefix}: {len(candidates)} selected candidates")
        for candidate in candidates:
            all_candidates.append((candidate, ds))

    all_candidates.sort(key=lambda item: item[0].score, reverse=True)
    selected = all_candidates[: args.top_k]
    if not selected:
        print("[DLP npz] no suitable candidates")
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    npz_paths = []
    index = []
    for rank, (candidate, ds) in enumerate(selected, start=1):
        episode_name = f"episode_{rank:06d}_{candidate.scene_prefix}_{candidate.agent_token[:8]}_{candidate.start_index:05d}"
        print(
            f"[DLP npz] export #{rank}: {candidate.scene_prefix} "
            f"agent={candidate.agent_token[:8]} frames={candidate.start_index}-{candidate.end_index} "
            f"path={candidate.path_length_m:.2f}m turn={candidate.heading_change_rad:.2f}rad "
            f"speed={candidate.mean_abs_speed:.2f}m/s reverse={candidate.reverse_ratio:.2f}"
        )
        existing_npz = args.output_root / f"{episode_name}.npz"
        existing_meta = args.output_root / episode_name / "metadata.json"
        if args.skip_existing and existing_npz.exists() and existing_meta.exists():
            print(f"[DLP npz] skip existing #{rank}: {existing_npz.name}")
            metadata = json.loads(existing_meta.read_text(encoding="utf-8"))
        else:
            metadata = export_candidate(
                ds,
                candidate,
                out_dir=args.output_root,
                episode_name=episode_name,
                raw_stride=args.raw_stride,
                fps=args.fps,
                wheelbase=args.wheelbase,
                px_per_m=args.px_per_m,
                camera_center_mode=args.camera_center,
                preview_every=0 if args.skip_preview else args.preview_every,
                preview_fps=args.preview_fps,
            )
        npz_paths.append(Path(metadata["npz_path"]))
        index.append(metadata)

    summary = summarize_npz_files(npz_paths)
    (args.output_root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DLP npz] wrote {len(selected)} episodes to {args.output_root}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
