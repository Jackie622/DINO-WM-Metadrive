#!/usr/bin/env python3
"""Test MetaDrive predictor with a DLP-adapted decoder on DLP clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_resolvers  # noqa: F401,E402
from tools.test_dlp_open_loop_wm import save_strip, tensor_to_uint8  # noqa: E402


MODEL_KEYS = {"encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"}


def load_payload(path: Path, device: torch.device) -> dict:
    with path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    return payload


def load_stats(cfg, mode: str):
    datasets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    train_dset = datasets["train"]
    dset = traj_dsets["valid"]
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
    print(f"[Hybrid DLP] loaded {mode} stats: action_mean={dset.action_mean[:2].tolist()}, action_std={dset.action_std[:2].tolist()}")
    return dset


def instantiate_hybrid_model(meta_cfg, meta_ckpt: Path, dlp_decoder_ckpt: Path, device: torch.device):
    meta = load_payload(meta_ckpt, device)
    dlp = load_payload(dlp_decoder_ckpt, device)
    if "encoder" not in meta:
        meta["encoder"] = hydra.utils.instantiate(meta_cfg.encoder)
    missing = {"predictor", "proprio_encoder", "action_encoder"} - set(meta.keys())
    if missing:
        raise ValueError(f"MetaDrive checkpoint missing modules: {sorted(missing)}")
    if "decoder" not in dlp:
        raise ValueError(f"DLP decoder checkpoint has no decoder: {dlp_decoder_ckpt}")

    modules = {
        "encoder": meta["encoder"],
        "predictor": meta["predictor"],
        "proprio_encoder": meta["proprio_encoder"],
        "action_encoder": meta["action_encoder"],
        "decoder": dlp["decoder"],
    }

    model = hydra.utils.instantiate(
        meta_cfg.model,
        encoder=modules["encoder"],
        proprio_encoder=modules["proprio_encoder"],
        action_encoder=modules["action_encoder"],
        predictor=modules["predictor"],
        decoder=modules["decoder"],
        proprio_dim=meta_cfg.proprio_emb_dim,
        action_dim=meta_cfg.action_emb_dim,
        concat_dim=meta_cfg.concat_dim,
        num_action_repeat=meta_cfg.num_action_repeat,
        num_proprio_repeat=meta_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    print(f"[Hybrid DLP] MetaDrive predictor epoch={meta.get('epoch')} + DLP decoder epoch={dlp.get('epoch')}")
    return model


def load_npz_clip(path: Path, max_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        images = data["image"]
        actions = data["action"]
        states = data["state"]
    n = min(len(images), max_frames)
    if n < 4:
        raise ValueError(f"Need at least 4 frames, got {n}: {path}")
    return images[:n], actions[:n], states[:n]


def preprocess_visual(images: np.ndarray, device: torch.device) -> torch.Tensor:
    visual = torch.from_numpy(images).float().to(device)
    visual = visual.permute(0, 3, 1, 2) / 255.0
    visual = visual * 2.0 - 1.0
    return visual.unsqueeze(0)


def build_macro_actions(
    raw_actions: np.ndarray,
    num_frames: int,
    frameskip: int,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    clip: float,
    device: torch.device,
) -> torch.Tensor:
    actions = torch.from_numpy(raw_actions.astype(np.float32))
    if actions.shape[0] < num_frames:
        pad = actions[-1:].repeat(num_frames - actions.shape[0], 1)
        actions = torch.cat([actions, pad], dim=0)
    actions = actions[:num_frames]
    if clip > 0:
        actions = actions.clamp(-clip, clip)
    macro = actions.repeat_interleave(frameskip, dim=-1)
    normalized = (macro - action_mean.cpu()) / action_std.cpu()
    return normalized.unsqueeze(0).float().to(device)


def make_model_timestep_clip(images: np.ndarray, raw_actions: np.ndarray, states: np.ndarray, frameskip: int):
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
    actions = torch.from_numpy(macro_actions.astype(np.float32))
    if clip > 0:
        actions = actions.clamp(-clip, clip)
    normalized = (actions - action_mean.cpu()) / action_std.cpu()
    return normalized.unsqueeze(0).float().to(device)


def run_one_clip(
    model,
    npz_path: Path,
    output_dir: Path,
    action_stats,
    meta_cfg,
    device: torch.device,
    max_frames: int,
    clip_actions: float,
    strip_every: int,
) -> dict:
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
        clip=clip_actions,
        device=device,
    )

    with torch.no_grad():
        recon_obs, _ = model.decode_obs(model.encode_obs({"visual": target_visual, "proprio": proprio}))
        z_obses, _ = model.rollout(obs_0, actions)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    save_strip(
        output_dir / "target_strip.png",
        [tensor_to_uint8(target_visual[0, i]) for i in range(compare_len)],
        "DLP target",
        every=strip_every,
    )
    save_strip(
        output_dir / "recon_strip.png",
        [tensor_to_uint8(recon_visual[0, i]) for i in range(compare_len)],
        "DLP decoder reconstruction",
        every=strip_every,
    )
    save_strip(
        output_dir / "hybrid_pred_strip.png",
        [tensor_to_uint8(pred_visual[0, i]) for i in range(compare_len)],
        "MetaDrive predictor + DLP decoder",
        every=strip_every,
    )
    save_strip(
        output_dir / "persist_strip.png",
        [tensor_to_uint8(persist_visual[0, i]) for i in range(compare_len)],
        "Persistence baseline",
        every=strip_every,
    )

    metrics = {
        "npz_path": str(npz_path),
        "compare_len": int(compare_len),
        "num_hist": num_hist,
        "frameskip": int(meta_cfg.frameskip),
        "wm_mse_per_frame": wm_mse.tolist(),
        "reconstruction_mse_per_frame": recon_mse.tolist(),
        "persistence_mse_per_frame": persist_mse.tolist(),
        "future_wm_mse_mean": float(np.mean(future_wm)) if len(future_wm) else None,
        "reconstruction_mse_mean": float(np.mean(recon_mse)),
        "future_persistence_mse_mean": float(np.mean(future_persist)) if len(future_persist) else None,
        "future_wm_over_persistence_mean": float(np.mean(ratio)) if len(ratio) else None,
        "future_wm_better_than_persistence_pct": float(np.mean(future_wm < future_persist) * 100.0) if len(future_wm) else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        "[Hybrid DLP] "
        f"{npz_path.name}: future_wm={metrics['future_wm_mse_mean']:.4f}, "
        f"persist={metrics['future_persistence_mse_mean']:.4f}, "
        f"ratio={metrics['future_wm_over_persistence_mean']:.4f}, "
        f"better={metrics['future_wm_better_than_persistence_pct']:.1f}%"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-cfg", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/metadrive_park_3/hydra.yaml"), type=Path)
    parser.add_argument("--meta-ckpt", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/metadrive_park_3/checkpoints/model_latest.pth"), type=Path)
    parser.add_argument("--dlp-cfg", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/dlp_recon_stage1/hydra.yaml"), type=Path)
    parser.add_argument("--dlp-decoder-ckpt", default=Path("/root/autodl-tmp/checkpoints/checkpoints_parking/outputs/dlp_recon_stage1/checkpoints/model_latest.pth"), type=Path)
    parser.add_argument("--data-root", default=Path("/root/autodl-tmp/dlp_datasets/dlp_metadrive_npz_stage1_full_recon"), type=Path)
    parser.add_argument("--output-dir", default=REPO_ROOT / "tools/outputs/dlp_hybrid_predictor_decoder", type=Path)
    parser.add_argument("--normalization", choices=["metadrive", "dlp"], default="metadrive")
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--clip-actions", type=float, default=1.0)
    parser.add_argument("--strip-every", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    meta_cfg = OmegaConf.load(args.meta_cfg)
    dlp_cfg = OmegaConf.load(args.dlp_cfg)
    stats_cfg = meta_cfg if args.normalization == "metadrive" else dlp_cfg
    action_stats = load_stats(stats_cfg, args.normalization)

    model = instantiate_hybrid_model(meta_cfg, args.meta_ckpt, args.dlp_decoder_ckpt, device)
    files = sorted(args.data_root.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {args.data_root}")
    selected = files[args.start_index : args.start_index + args.num_clips]
    if not selected:
        raise ValueError(f"No files selected from index {args.start_index}")

    all_metrics = []
    run_dir = args.output_dir / f"norm_{args.normalization}"
    for i, npz_path in enumerate(selected):
        metrics = run_one_clip(
            model=model,
            npz_path=npz_path,
            output_dir=run_dir / f"clip_{args.start_index + i:04d}_{npz_path.stem}",
            action_stats=action_stats,
            meta_cfg=meta_cfg,
            device=device,
            max_frames=args.max_frames,
            clip_actions=args.clip_actions,
            strip_every=args.strip_every,
        )
        all_metrics.append(metrics)

    summary = {
        "normalization": args.normalization,
        "num_clips": len(all_metrics),
        "mean_future_wm_mse": float(np.mean([m["future_wm_mse_mean"] for m in all_metrics])),
        "mean_reconstruction_mse": float(np.mean([m["reconstruction_mse_mean"] for m in all_metrics])),
        "mean_future_persistence_mse": float(np.mean([m["future_persistence_mse_mean"] for m in all_metrics])),
        "mean_future_wm_over_persistence": float(np.mean([m["future_wm_over_persistence_mean"] for m in all_metrics])),
        "mean_future_wm_better_than_persistence_pct": float(np.mean([m["future_wm_better_than_persistence_pct"] for m in all_metrics])),
        "clips": all_metrics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Hybrid DLP] wrote {run_dir}")
    print(json.dumps({k: v for k, v in summary.items() if k != "clips"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
