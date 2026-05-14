#!/usr/bin/env python3
"""Train a DLP decoder to decode frozen MetaDrive predictor latents."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import utils as tv_utils
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_resolvers  # noqa: F401,E402
from data_loader_park import build_parking_dataset  # noqa: E402
from tools.test_dlp_hybrid_predictor_decoder import instantiate_hybrid_model  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_all_but_decoder(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True
    model.encoder.eval()
    model.predictor.eval()
    model.action_encoder.eval()
    model.proprio_encoder.eval()
    model.decoder.train()


def restore_raw_actions(actions: torch.Tensor, dset) -> torch.Tensor:
    mean = dset.action_mean.to(actions.device)
    std = dset.action_std.to(actions.device)
    return actions * std + mean


def normalize_actions(actions: torch.Tensor, dset) -> torch.Tensor:
    mean = dset.action_mean.to(actions.device)
    std = dset.action_std.to(actions.device)
    return (actions - mean) / std


def convert_actions(actions: torch.Tensor, source_dset, target_dset, mode: str) -> torch.Tensor:
    if mode == "dlp":
        return actions
    if mode != "metadrive":
        raise ValueError(f"Unknown action_normalization={mode}")
    raw = restore_raw_actions(actions, source_dset)
    return normalize_actions(raw, target_dset)


def make_loaders(cfg, meta_cfg):
    datasets, _ = build_parking_dataset(
        cfg.dlp_data_path,
        num_hist=meta_cfg.num_hist,
        num_pred=meta_cfg.num_pred,
        frameskip=meta_cfg.frameskip,
    )
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=int(cfg.training.batch_size),
            shuffle=(split == "train"),
            num_workers=int(cfg.training.num_workers),
            pin_memory=True,
        )
        for split in ["train", "valid"]
    }
    return datasets, loaders


def load_meta_stats(meta_cfg):
    datasets, _ = hydra.utils.call(
        meta_cfg.env.dataset,
        num_hist=meta_cfg.num_hist,
        num_pred=meta_cfg.num_pred,
        frameskip=meta_cfg.frameskip,
    )
    return datasets["train"]


def forward_losses(model, obs, actions, cfg):
    z = model.encode(obs, actions)
    z_src = z[:, : model.num_hist]
    z_tgt = z[:, model.num_pred :]

    with torch.no_grad():
        z_pred = model.predict(z_src)
        z_pred_obs, _ = model.separate_emb(z_pred)
        z_tgt_obs, _ = model.separate_emb(z_tgt.detach())

    pred_obs, pred_diff = model.decode_obs(z_pred_obs)
    recon_obs, recon_diff = model.decode_obs(z_tgt_obs)

    visual_tgt = obs["visual"][:, model.num_pred :]
    pred_loss = F.mse_loss(pred_obs["visual"], visual_tgt)
    recon_loss = F.mse_loss(recon_obs["visual"], visual_tgt)
    loss = (
        float(cfg.loss.pred_weight) * pred_loss
        + float(cfg.loss.recon_weight) * recon_loss
        + float(cfg.loss.vq_weight) * (pred_diff.mean() + recon_diff.mean())
    )
    return loss, {
        "loss": loss.detach(),
        "pred_loss": pred_loss.detach(),
        "recon_loss": recon_loss.detach(),
        "vq_loss": (pred_diff.mean() + recon_diff.mean()).detach(),
    }, pred_obs["visual"].detach(), recon_obs["visual"].detach(), visual_tgt.detach()


def save_plot(path: Path, target: torch.Tensor, pred: torch.Tensor, recon: torch.Tensor, num_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(num_samples, target.shape[0])
    target = target[:n]
    pred = pred[:n]
    recon = recon[:n]
    imgs = torch.cat(
        [
            target.reshape(-1, *target.shape[2:]),
            pred.reshape(-1, *pred.shape[2:]),
            recon.reshape(-1, *recon.shape[2:]),
        ],
        dim=0,
    )
    tv_utils.save_image(
        imgs,
        path,
        nrow=n * target.shape[1],
        normalize=True,
        value_range=(-1, 1),
    )


def run_epoch(model, loader, dlp_dset, meta_dset, optimizer, cfg, device, train: bool, epoch: int, out_dir: Path):
    model.decoder.train(train)
    logs = defaultdict(float)
    count = 0
    max_batches = cfg.training.max_train_batches if train else cfg.training.max_valid_batches
    phase = "train" if train else "valid"

    iterator = tqdm(loader, desc=f"Epoch {epoch:03d} {phase}")
    for batch_idx, (obs, actions, _state) in enumerate(iterator):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        obs = {k: v.to(device, non_blocking=True) for k, v in obs.items()}
        actions = actions.to(device, non_blocking=True)
        actions = convert_actions(actions, dlp_dset, meta_dset, str(cfg.action_normalization))

        with torch.set_grad_enabled(train):
            loss, batch_logs, pred, recon, target = forward_losses(model, obs, actions, cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bsz = target.shape[0]
        count += bsz
        for key, value in batch_logs.items():
            logs[key] += float(value.item()) * bsz
        iterator.set_postfix({k: f"{logs[k] / max(count, 1):.4f}" for k in ["loss", "pred_loss", "recon_loss"]})

        if batch_idx == 0 and (epoch % int(cfg.training.plot_every_x_epoch) == 0):
            save_plot(
                out_dir / phase / f"{phase}_e{epoch:05d}_b0.png",
                target,
                pred,
                recon,
                int(cfg.training.num_plot_samples),
            )

    return {key: value / max(count, 1) for key, value in logs.items()}


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg, meta_cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "decoder": model.decoder,
            "decoder_optimizer": optimizer.state_dict(),
            "stage15_cfg": OmegaConf.to_container(cfg, resolve=True),
            "meta_cfg": OmegaConf.to_container(meta_cfg, resolve=True),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=REPO_ROOT / "conf/train_dlp_decoder_align_stage15.yaml", type=Path)
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides, e.g. training.epochs=1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    meta_cfg = OmegaConf.load(cfg.meta_cfg)
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage15_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    seed_everything(int(cfg.training.seed))
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    print(f"[Stage1.5] device={device}")
    print("[Stage1.5] building DLP loaders")
    dlp_datasets, loaders = make_loaders(cfg, meta_cfg)
    print("[Stage1.5] loading MetaDrive action stats")
    meta_dset = load_meta_stats(meta_cfg)
    print("[Stage1.5] building hybrid model")
    model = instantiate_hybrid_model(
        meta_cfg=meta_cfg,
        meta_ckpt=Path(str(cfg.meta_ckpt)),
        dlp_decoder_ckpt=Path(str(cfg.dlp_decoder_ckpt)),
        device=device,
    )
    freeze_all_but_decoder(model)
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=float(cfg.training.decoder_lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    history = []
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_logs = run_epoch(
            model, loaders["train"], dlp_datasets["train"], meta_dset, optimizer, cfg, device, True, epoch, out_dir
        )
        with torch.no_grad():
            valid_logs = run_epoch(
                model, loaders["valid"], dlp_datasets["valid"], meta_dset, optimizer, cfg, device, False, epoch, out_dir
            )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_logs.items()},
            **{f"valid_{k}": v for k, v in valid_logs.items()},
        }
        history.append(row)
        print(
            "[Stage1.5] "
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.5f} train_pred={row['train_pred_loss']:.5f} "
            f"valid_loss={row['valid_loss']:.5f} valid_pred={row['valid_pred_loss']:.5f}"
        )
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epoch % int(cfg.training.save_every_x_epoch) == 0 or epoch == int(cfg.training.epochs):
            save_checkpoint(out_dir / "checkpoints" / "model_latest.pth", model, optimizer, epoch, cfg, meta_cfg)
            save_checkpoint(out_dir / "checkpoints" / f"model_{epoch}.pth", model, optimizer, epoch, cfg, meta_cfg)
            print(f"[Stage1.5] saved checkpoint at epoch {epoch}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
