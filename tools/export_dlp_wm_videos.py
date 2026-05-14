#!/usr/bin/env python3
"""Export DLP target and WM open-loop prediction videos for visual comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_resolvers  # noqa: F401,E402
from tools.test_dlp_hybrid_predictor_decoder import (  # noqa: E402
    instantiate_hybrid_model,
    load_npz_clip,
    load_stats,
    preprocess_visual,
)
from tools.test_dlp_open_loop_wm import tensor_to_uint8  # noqa: E402


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=None)


def make_model_timestep_clip(images: np.ndarray, raw_actions: np.ndarray, states: np.ndarray, frameskip: int):
    """Convert raw DLP frames to the model timestep used by ParkingDataset.

    Training sees images/states every `frameskip` raw frames and one action
    token made by flattening the intervening `frameskip` low-level actions.
    """
    usable = (len(images) // frameskip) * frameskip
    if usable < frameskip:
        raise ValueError(f"Not enough frames for frameskip={frameskip}: {len(images)}")
    images = images[:usable]
    states = states[:usable]
    raw_actions = raw_actions[:usable]

    ds_images = images[::frameskip]
    ds_states = states[::frameskip]
    macro_actions = raw_actions.reshape(-1, frameskip * raw_actions.shape[-1])
    return ds_images, macro_actions.astype(np.float32), ds_states.astype(np.float32)


def normalize_macro_actions(
    macro_actions: np.ndarray,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    clip: float,
    device: torch.device,
) -> torch.Tensor:
    actions = torch.from_numpy(macro_actions).float()
    if clip > 0:
        actions = actions.clamp(-clip, clip)
    actions = (actions - action_mean.cpu()) / action_std.cpu()
    return actions.unsqueeze(0).to(device)


def run_one_clip(model, npz_path: Path, output_dir: Path, action_stats, meta_cfg, device, max_frames: int, fps: int):
    raw_max_frames = max_frames * int(meta_cfg.frameskip)
    images, raw_actions, states = load_npz_clip(npz_path, max_frames=raw_max_frames)
    images, macro_actions, states = make_model_timestep_clip(
        images,
        raw_actions,
        states,
        frameskip=int(meta_cfg.frameskip),
    )
    target_visual = preprocess_visual(images, device)
    proprio = torch.from_numpy(states.astype(np.float32)).unsqueeze(0).to(device)
    num_hist = int(meta_cfg.num_hist)

    obs_0 = {
        "visual": target_visual[:, :num_hist],
        "proprio": proprio[:, :num_hist],
    }
    actions = normalize_macro_actions(
        macro_actions,
        action_mean=action_stats.action_mean,
        action_std=action_stats.action_std,
        clip=1.0,
        device=device,
    )

    with torch.no_grad():
        z_obses, _ = model.rollout(obs_0, actions)
        pred_obs, _ = model.decode_obs(z_obses)

    compare_len = min(pred_obs["visual"].shape[1], target_visual.shape[1])
    pred_visual = pred_obs["visual"][:, :compare_len]
    target_visual = target_visual[:, :compare_len]

    target_frames = [tensor_to_uint8(target_visual[0, i]) for i in range(compare_len)]
    pred_frames = [tensor_to_uint8(pred_visual[0, i]) for i in range(compare_len)]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_video(output_dir / "target_video.mp4", target_frames, fps=fps)
    write_video(output_dir / "wm_pred_video.mp4", pred_frames, fps=fps)

    wm_mse = ((pred_visual - target_visual) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    persist_visual = target_visual[:, num_hist - 1 : num_hist].repeat(1, compare_len, 1, 1, 1)
    persist_mse = ((persist_visual - target_visual) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    future = slice(num_hist, None)
    metrics = {
        "npz_path": str(npz_path),
        "compare_len": int(compare_len),
        "num_hist": num_hist,
        "frameskip": int(meta_cfg.frameskip),
        "fps": int(fps),
        "future_wm_mse_mean": float(np.mean(wm_mse[future])),
        "future_persistence_mse_mean": float(np.mean(persist_mse[future])),
        "future_wm_over_persistence_mean": float(np.mean(wm_mse[future] / np.maximum(persist_mse[future], 1e-8))),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[DLP Video] wrote {output_dir}")
    print(json.dumps(metrics, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-cfg", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/metadrive_park_3/hydra.yaml"), type=Path)
    parser.add_argument("--meta-ckpt", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/dlp_predictor_finetune_stage2/checkpoints/model_latest.pth"), type=Path)
    parser.add_argument("--dlp-decoder-ckpt", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/dlp_predictor_finetune_stage2/checkpoints/model_latest.pth"), type=Path)
    parser.add_argument("--data-root", default=Path("/root/autodl-tmp/dlp_datasets/dlp_metadrive_npz_stage1_full_recon"), type=Path)
    parser.add_argument("--output-dir", default=REPO_ROOT / "tools/outputs/dlp_stage2_video_compare", type=Path)
    parser.add_argument("--normalization", choices=["metadrive", "dlp"], default="metadrive")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta_cfg = OmegaConf.load(args.meta_cfg)
    stats_cfg = meta_cfg
    if args.normalization == "dlp":
        raise ValueError("DLP normalization is not used for the stage2 checkpoint; use --normalization metadrive.")
    action_stats = load_stats(stats_cfg, args.normalization)
    model = instantiate_hybrid_model(meta_cfg, args.meta_ckpt, args.dlp_decoder_ckpt, device)

    files = sorted(args.data_root.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.data_root}")
    if args.start_index < 0 or args.start_index >= len(files):
        raise IndexError(f"start-index {args.start_index} outside 0..{len(files)-1}")

    npz_path = files[args.start_index]
    clip_dir = args.output_dir / f"clip_{args.start_index:04d}_{npz_path.stem}"
    run_one_clip(
        model=model,
        npz_path=npz_path,
        output_dir=clip_dir,
        action_stats=action_stats,
        meta_cfg=meta_cfg,
        device=device,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
