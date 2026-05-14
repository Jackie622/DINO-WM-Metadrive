#!/usr/bin/env python3
"""Open-loop WM probe on MetaDrive-style rendered DLP clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DLP_ROOT = REPO_ROOT / "dlp_dataset"
for path in (REPO_ROOT, DLP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import custom_resolvers  # noqa: F401,E402
from preprocessor import Preprocessor  # noqa: E402
from tools.prepare_dlp_open_loop import load_scene, pseudo_controls, trajectory_for_agent  # noqa: E402
from tools.test_dlp_open_loop_wm import load_model, save_strip, tensor_to_uint8  # noqa: E402


def load_frames(frame_dir: Path) -> np.ndarray:
    paths = sorted(frame_dir.glob("*.png"))
    if len(paths) < 4:
        raise ValueError(f"Need at least 4 frames under {frame_dir}, found {len(paths)}")
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths], axis=0)


def build_dlp_sample(metadata: dict, data_root: Path, fps: float) -> dict:
    ds = load_scene(data_root, metadata["scene_prefix"])
    instances, traj_full = trajectory_for_agent(ds, metadata["agent_token"])
    raw_indices = np.arange(metadata["start_index"], metadata["end_index"] + 1, metadata["sample_stride"])
    if raw_indices[-1] != metadata["end_index"]:
        raw_indices = np.append(raw_indices, metadata["end_index"])
    traj = traj_full[raw_indices].astype(np.float32)
    controls = pseudo_controls(
        traj,
        dt=float(metadata["sample_stride"]) / fps,
        wheelbase=2.8,
    )
    return {
        "traj": traj,
        "pseudo_actions": controls.astype(np.float32),
        "raw_indices": raw_indices.astype(np.int32),
        "instance_tokens": np.asarray([instances[int(i)]["instance_token"] for i in raw_indices]),
    }


def build_macro_actions(
    pseudo_actions: np.ndarray,
    num_frames: int,
    frameskip: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    controls = torch.from_numpy(pseudo_actions).float()
    if controls.numel() == 0:
        raise ValueError("No pseudo actions found")
    if clip > 0:
        controls = controls.clamp(-clip, clip)
    macro = controls.repeat_interleave(frameskip, dim=-1)
    macro = torch.cat([macro, macro[-1:]], dim=0)
    macro = macro[:num_frames]
    normalized = (macro - action_mean.cpu()) / action_std.cpu()
    return normalized.unsqueeze(0)


def build_proprio(traj: np.ndarray, controls: np.ndarray, num_frames: int) -> np.ndarray:
    proprio = np.zeros((num_frames, 3), dtype=np.float32)
    proprio[:, 0] = traj[:num_frames, 3]
    if len(controls):
        steer = np.concatenate([[controls[0, 0]], controls[:, 0]])[:num_frames]
        proprio[:, 1] = steer
    return proprio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--data-root", default=REPO_ROOT / "dlp_dataset/data", type=Path)
    parser.add_argument("--plan-cfg", default=REPO_ROOT / "conf/plan_park_guided.yaml", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-epoch", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-actions", type=float, default=1.0)
    parser.add_argument("--strip-every", type=int, default=1)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    clip_dir = args.clip_dir
    metadata = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
    frames = load_frames(clip_dir / "frames")
    sample = build_dlp_sample(metadata, args.data_root, fps=args.fps)

    cfg = OmegaConf.load(args.plan_cfg)
    model_path = Path(str(cfg.ckpt_base_path)) / "outputs" / str(cfg.model_name)
    model_epoch = args.model_epoch or str(cfg.model_epoch)
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")

    device = torch.device(args.device)
    print("[DLP style WM] loading dataset stats")
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dsets["valid"]
    train_dset = datasets["train"]
    for name in [
        "raw_action_mean",
        "raw_action_std",
        "action_mean",
        "action_std",
        "state_mean",
        "state_std",
        "proprio_mean",
        "proprio_std",
    ]:
        if hasattr(train_dset, name):
            setattr(dset, name, getattr(train_dset, name))

    model_ckpt = model_path / "checkpoints" / f"model_{model_epoch}.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device)
    preprocessor = Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )

    num_hist = int(model_cfg.num_hist)
    compare_len = min(len(frames), len(sample["traj"]))
    frames = frames[:compare_len]
    traj = sample["traj"][:compare_len]
    controls = sample["pseudo_actions"][: max(0, compare_len - 1)]
    actions = build_macro_actions(
        controls,
        num_frames=compare_len,
        frameskip=int(model_cfg.frameskip),
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        clip=args.clip_actions,
    ).to(device)
    proprio = build_proprio(traj, controls, compare_len)

    obs_0 = {
        "visual": frames[:num_hist][None],
        "proprio": proprio[:num_hist][None],
    }
    trans_obs_0 = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in preprocessor.transform_obs(obs_0).items()
    }
    target_visual = preprocessor.transform_obs_visual(frames[None]).to(device)

    with torch.no_grad():
        target_obs = {
            "visual": target_visual,
            "proprio": torch.from_numpy(proprio[None]).float().to(device),
        }
        recon_obs, _ = model.decode_obs(model.encode_obs(target_obs))
        z_obses, _ = model.rollout(trans_obs_0, actions.float())
        if model.decoder is None:
            raise ValueError("Checkpoint has no decoder; visual prediction cannot be evaluated.")
        pred_obs, _ = model.decode_obs(z_obses)

    compare_len = min(pred_obs["visual"].shape[1], target_visual.shape[1])
    pred_visual = pred_obs["visual"][:, :compare_len]
    target_visual = target_visual[:, :compare_len]
    recon_visual = recon_obs["visual"][:, :compare_len]
    persist_visual = target_visual[:, num_hist - 1 : num_hist].repeat(1, compare_len, 1, 1, 1)

    wm_mse = ((pred_visual - target_visual) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    recon_mse = ((recon_visual - target_visual) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    persist_mse = ((persist_visual - target_visual) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    future = slice(num_hist, None)
    future_wm = wm_mse[future]
    future_persist = persist_mse[future]
    ratio = future_wm / np.maximum(future_persist, 1e-8)

    output_dir = args.output_dir or (clip_dir / "wm_probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_strip(
        output_dir / "target_strip.png",
        [tensor_to_uint8(target_visual[0, i]) for i in range(compare_len)],
        "DLP MetaDrive-style target",
        every=args.strip_every,
    )
    save_strip(
        output_dir / "pred_strip.png",
        [tensor_to_uint8(pred_visual[0, i]) for i in range(compare_len)],
        "WM prediction",
        every=args.strip_every,
    )
    save_strip(
        output_dir / "persist_strip.png",
        [tensor_to_uint8(persist_visual[0, i]) for i in range(compare_len)],
        "Persistence baseline",
        every=args.strip_every,
    )
    save_strip(
        output_dir / "recon_strip.png",
        [tensor_to_uint8(recon_visual[0, i]) for i in range(compare_len)],
        "Encode-decode reconstruction",
        every=args.strip_every,
    )

    metrics = {
        "clip_dir": str(clip_dir),
        "compare_len": int(compare_len),
        "num_hist": num_hist,
        "frameskip": int(model_cfg.frameskip),
        "wm_mse_per_frame": wm_mse.tolist(),
        "reconstruction_mse_per_frame": recon_mse.tolist(),
        "persistence_mse_per_frame": persist_mse.tolist(),
        "future_wm_mse_mean": float(np.mean(future_wm)) if len(future_wm) else None,
        "reconstruction_mse_mean": float(np.mean(recon_mse)),
        "future_persistence_mse_mean": float(np.mean(future_persist)) if len(future_persist) else None,
        "future_wm_over_persistence_mean": float(np.mean(ratio)) if len(ratio) else None,
        "future_wm_better_than_persistence_pct": float(np.mean(future_wm < future_persist) * 100.0) if len(future_wm) else None,
        "note": "DLP actions are pseudo controls inferred from annotation; this tests visual-domain transfer, not native DLP control fidelity.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[DLP style WM] wrote {output_dir}")
    print(
        "[DLP style WM] "
        f"future_wm_mse={metrics['future_wm_mse_mean']:.4f}, "
        f"persistence={metrics['future_persistence_mse_mean']:.4f}, "
        f"ratio={metrics['future_wm_over_persistence_mean']:.4f}, "
        f"better_pct={metrics['future_wm_better_than_persistence_pct']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
