#!/usr/bin/env python3
"""Run a first DLP open-loop WM rollout on prepared local BEV samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_resolvers  # noqa: F401,E402
from preprocessor import Preprocessor  # noqa: E402


ALL_MODEL_KEYS = {
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
}


def load_ckpt(snapshot_path: Path, device: torch.device) -> dict:
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {k: v.to(device) for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt: Path, train_cfg, num_action_repeat: int, device: torch.device):
    result = load_ckpt(model_ckpt, device)
    print(f"[DLP WM] loaded epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        if train_cfg.env.decoder_path is None:
            raise ValueError("Decoder path not found")
        decoder_path = REPO_ROOT / train_cfg.env.decoder_path
        ckpt = torch.load(decoder_path, map_location=device)
        result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().clamp(-1, 1)
    arr = ((arr + 1.0) * 127.5).byte().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return arr


def save_strip(path: Path, frames: list[np.ndarray], title: str, every: int = 1) -> None:
    frames = frames[::every]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    label_h = 18
    canvas = Image.new("RGB", (w * len(frames), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), title, fill=(0, 0, 0))
    for i, frame in enumerate(frames):
        canvas.paste(Image.fromarray(frame), (i * w, label_h))
        draw.text((i * w + 4, label_h + 4), str(i), fill=(0, 0, 0))
    canvas.save(path)


def load_sequence_images(sample_dir: Path) -> np.ndarray:
    seq_dir = sample_dir / "sequence_bev"
    if not seq_dir.exists():
        raise FileNotFoundError(
            f"{seq_dir} not found. Re-run tools/prepare_dlp_open_loop.py with --save-sequence-images."
        )
    paths = sorted(seq_dir.glob("*.png"))
    if len(paths) < 4:
        raise ValueError(f"Need at least 4 sequence frames, found {len(paths)} in {seq_dir}")
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths], axis=0)


def build_macro_actions(sample: dict, frameskip: int, action_mean: torch.Tensor, action_std: torch.Tensor, clip: float) -> torch.Tensor:
    controls = torch.from_numpy(sample["pseudo_actions"]).float()
    if controls.numel() == 0:
        raise ValueError("sample has no pseudo_actions")

    if clip > 0:
        controls = controls.clamp(-clip, clip)

    macro = controls.repeat_interleave(frameskip, dim=-1)
    if macro.shape[-1] != action_mean.numel():
        raise ValueError(
            f"macro action dim {macro.shape[-1]} does not match model action dim {action_mean.numel()}"
        )

    # Rollout expects one action token per encoded/imagined frame. Pad one last
    # macro action so length aligns with the visual sequence.
    macro = torch.cat([macro, macro[-1:]], dim=0)
    normalized = (macro - action_mean.cpu()) / action_std.cpu()
    return normalized.unsqueeze(0)


def build_proprio(sample: dict, num_frames: int) -> np.ndarray:
    traj = sample["traj"].astype(np.float32)
    controls = sample["pseudo_actions"].astype(np.float32)
    proprio = np.zeros((num_frames, 3), dtype=np.float32)
    proprio[:, 0] = traj[:num_frames, 3]
    if len(controls):
        steer = np.concatenate([[controls[0, 0]], controls[:, 0]])[:num_frames]
        proprio[:, 1] = steer
    return proprio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-cfg", default=REPO_ROOT / "conf/plan_park_guided.yaml", type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-epoch", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-actions", type=float, default=1.0)
    parser.add_argument("--strip-every", type=int, default=1)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.plan_cfg)
    ckpt_base_path = str(cfg.ckpt_base_path)
    model_name = str(cfg.model_name)
    model_epoch = args.model_epoch or str(cfg.model_epoch)
    model_path = Path(ckpt_base_path) / "outputs" / model_name
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")

    device = torch.device(args.device)
    print("[DLP WM] loading dataset stats")
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dsets["valid"]
    train_dset = datasets["train"]
    for name in ["raw_action_mean", "raw_action_std", "action_mean", "action_std", "state_mean", "state_std", "proprio_mean", "proprio_std"]:
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

    sample_dir = args.sample_dir
    output_dir = args.output_dir or (sample_dir / "wm_open_loop")
    output_dir.mkdir(parents=True, exist_ok=True)

    visuals = load_sequence_images(sample_dir)
    with np.load(sample_dir / "sample.npz") as data:
        sample = {k: data[k] for k in data.files}

    num_hist = int(model_cfg.num_hist)
    if visuals.shape[0] <= num_hist:
        raise ValueError(f"Need more frames than num_hist={num_hist}, got {visuals.shape[0]}")

    actions = build_macro_actions(
        sample,
        frameskip=int(model_cfg.frameskip),
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        clip=args.clip_actions,
    )
    actions = actions[:, : visuals.shape[0]].to(device)

    proprio = build_proprio(sample, num_frames=visuals.shape[0])
    obs_0 = {
        "visual": visuals[:num_hist][None],
        "proprio": proprio[:num_hist][None],
    }
    trans_obs_0 = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in preprocessor.transform_obs(obs_0).items()
    }
    actions = actions.float()

    with torch.no_grad():
        z_obses, _ = model.rollout(trans_obs_0, actions)
        if model.decoder is None:
            raise ValueError("This checkpoint has no decoder; visual open-loop image comparison is unavailable.")
        pred_obs, _ = model.decode_obs(z_obses)

    target_visual = preprocessor.transform_obs_visual(visuals[None]).to(device)
    compare_len = min(pred_obs["visual"].shape[1], target_visual.shape[1])
    pred_visual = pred_obs["visual"][:, :compare_len]
    target_visual = target_visual[:, :compare_len]
    mse_per_frame = ((pred_visual - target_visual) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    future_mse = mse_per_frame[num_hist:].tolist()

    pred_frames = [tensor_to_uint8(pred_visual[0, i]) for i in range(compare_len)]
    target_frames = [tensor_to_uint8(target_visual[0, i]) for i in range(compare_len)]
    save_strip(output_dir / "target_strip.png", target_frames, "DLP target", every=args.strip_every)
    save_strip(output_dir / "pred_strip.png", pred_frames, "WM prediction", every=args.strip_every)

    metrics = {
        "sample_dir": str(sample_dir),
        "compare_len": int(compare_len),
        "num_hist": num_hist,
        "frameskip": int(model_cfg.frameskip),
        "mse_per_frame": mse_per_frame.tolist(),
        "future_mse_mean": float(np.mean(future_mse)) if future_mse else None,
        "future_mse_final": float(future_mse[-1]) if future_mse else None,
        "note": "DLP controls are pseudo actions inferred from annotated trajectory, not real vehicle commands.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[DLP WM] wrote open-loop results to {output_dir}")
    print(f"[DLP WM] future_mse_mean={metrics['future_mse_mean']} final={metrics['future_mse_final']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
