#!/usr/bin/env python3
"""Fine-tune MetaDrive predictor on DLP open-loop clips with a DLP decoder."""

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


def configure_trainable_modules(model) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for module in [model.predictor, model.action_encoder, model.proprio_encoder, model.decoder]:
        for param in module.parameters():
            param.requires_grad = True

    model.encoder.eval()
    model.predictor.train()
    model.action_encoder.train()
    model.proprio_encoder.train()
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


def split_latent(model, z_pred: torch.Tensor, z_tgt: torch.Tensor):
    if model.concat_dim == 0:
        pred_visual = z_pred[:, :, :-2, :]
        tgt_visual = z_tgt[:, :, :-2, :].detach()
        pred_proprio = z_pred[:, :, -2, :]
        tgt_proprio = z_tgt[:, :, -2, :].detach()
    elif model.concat_dim == 1:
        pred_visual = z_pred[:, :, :, : -(model.proprio_dim + model.action_dim)]
        tgt_visual = z_tgt[:, :, :, : -(model.proprio_dim + model.action_dim)].detach()
        pred_proprio = z_pred[:, :, :, -(model.proprio_dim + model.action_dim) : -model.action_dim]
        tgt_proprio = z_tgt[:, :, :, -(model.proprio_dim + model.action_dim) : -model.action_dim].detach()
    else:
        raise ValueError(f"Unsupported concat_dim={model.concat_dim}")
    return pred_visual, tgt_visual, pred_proprio, tgt_proprio


def forward_losses(model, obs, actions, cfg):
    z = model.encode(obs, actions)
    z_src = z[:, : model.num_hist]
    z_tgt = z[:, model.num_pred :]

    z_pred = model.predict(z_src)
    pred_visual_z, tgt_visual_z, pred_proprio_z, tgt_proprio_z = split_latent(model, z_pred, z_tgt)

    visual_latent_loss = F.mse_loss(pred_visual_z, tgt_visual_z)
    proprio_latent_loss = F.mse_loss(pred_proprio_z, tgt_proprio_z)
    latent_loss = (
        float(cfg.loss.visual_latent_weight) * visual_latent_loss
        + float(cfg.loss.proprio_latent_weight) * proprio_latent_loss
    )

    z_pred_obs, _ = model.separate_emb(z_pred)
    z_tgt_obs, _ = model.separate_emb(z_tgt.detach())
    pred_obs, pred_diff = model.decode_obs(z_pred_obs)
    recon_obs, recon_diff = model.decode_obs(z_tgt_obs)

    visual_tgt = obs["visual"][:, model.num_pred :]
    pred_pixel_loss = F.mse_loss(pred_obs["visual"], visual_tgt)
    recon_loss = F.mse_loss(recon_obs["visual"], visual_tgt)
    vq_loss = pred_diff.mean() + recon_diff.mean()

    loss = (
        float(cfg.loss.latent_weight) * latent_loss
        + float(cfg.loss.pred_pixel_weight) * pred_pixel_loss
        + float(cfg.loss.recon_weight) * recon_loss
        + float(cfg.loss.vq_weight) * vq_loss
    )
    logs = {
        "loss": loss.detach(),
        "latent_loss": latent_loss.detach(),
        "visual_latent_loss": visual_latent_loss.detach(),
        "proprio_latent_loss": proprio_latent_loss.detach(),
        "pred_pixel_loss": pred_pixel_loss.detach(),
        "recon_loss": recon_loss.detach(),
        "vq_loss": vq_loss.detach(),
    }
    return loss, logs, pred_obs["visual"].detach(), recon_obs["visual"].detach(), visual_tgt.detach()


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
    tv_utils.save_image(imgs, path, nrow=n * target.shape[1], normalize=True, value_range=(-1, 1))


def set_train_mode(model, train: bool) -> None:
    model.encoder.eval()
    model.predictor.train(train)
    model.action_encoder.train(train)
    model.proprio_encoder.train(train)
    model.decoder.train(train)


def run_epoch(model, loader, dlp_dset, meta_dset, optimizer, cfg, device, train: bool, epoch: int, out_dir: Path):
    set_train_mode(model, train)
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
                clip_norm = float(cfg.training.grad_clip_norm)
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_norm=clip_norm,
                    )
                optimizer.step()

        bsz = target.shape[0]
        count += bsz
        for key, value in batch_logs.items():
            logs[key] += float(value.item()) * bsz
        iterator.set_postfix(
            {k: f"{logs[k] / max(count, 1):.4f}" for k in ["loss", "latent_loss", "pred_pixel_loss"]}
        )

        if batch_idx == 0 and (epoch % int(cfg.training.plot_every_x_epoch) == 0):
            save_plot(
                out_dir / phase / f"{phase}_e{epoch:05d}_b0.png",
                target,
                pred,
                recon,
                int(cfg.training.num_plot_samples),
            )

    return {key: value / max(count, 1) for key, value in logs.items()}


def build_optimizer(model, cfg):
    groups = [
        {"params": model.predictor.parameters(), "lr": float(cfg.training.predictor_lr), "name": "predictor"},
        {"params": model.action_encoder.parameters(), "lr": float(cfg.training.action_encoder_lr), "name": "action_encoder"},
        {"params": model.proprio_encoder.parameters(), "lr": float(cfg.training.proprio_encoder_lr), "name": "proprio_encoder"},
        {"params": model.decoder.parameters(), "lr": float(cfg.training.decoder_lr), "name": "decoder"},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(cfg.training.weight_decay))


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg, meta_cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "predictor": model.predictor,
            "decoder": model.decoder,
            "proprio_encoder": model.proprio_encoder,
            "action_encoder": model.action_encoder,
            "stage2_optimizer": optimizer.state_dict(),
            "stage2_cfg": OmegaConf.to_container(cfg, resolve=True),
            "meta_cfg": OmegaConf.to_container(meta_cfg, resolve=True),
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=REPO_ROOT / "conf/train_dlp_predictor_finetune_stage2.yaml", type=Path)
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides, e.g. training.epochs=1")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    meta_cfg = OmegaConf.load(cfg.meta_cfg)
    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage2_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    seed_everything(int(cfg.training.seed))
    device = torch.device(str(cfg.device) if torch.cuda.is_available() else "cpu")
    print(f"[Stage2] device={device}")
    print("[Stage2] building DLP loaders")
    dlp_datasets, loaders = make_loaders(cfg, meta_cfg)
    print("[Stage2] loading MetaDrive action stats")
    meta_dset = load_meta_stats(meta_cfg)
    print("[Stage2] building MetaDrive-init model with DLP decoder")
    model = instantiate_hybrid_model(
        meta_cfg=meta_cfg,
        meta_ckpt=Path(str(cfg.meta_ckpt)),
        dlp_decoder_ckpt=Path(str(cfg.dlp_decoder_ckpt)),
        device=device,
    )
    configure_trainable_modules(model)
    optimizer = build_optimizer(model, cfg)

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
            "[Stage2] "
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.5f} train_latent={row['train_latent_loss']:.5f} "
            f"train_pred_px={row['train_pred_pixel_loss']:.5f} "
            f"valid_loss={row['valid_loss']:.5f} valid_latent={row['valid_latent_loss']:.5f} "
            f"valid_pred_px={row['valid_pred_pixel_loss']:.5f}"
        )
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if epoch % int(cfg.training.save_every_x_epoch) == 0 or epoch == int(cfg.training.epochs):
            save_checkpoint(out_dir / "checkpoints" / "model_latest.pth", model, optimizer, epoch, cfg, meta_cfg)
            save_checkpoint(out_dir / "checkpoints" / f"model_{epoch}.pth", model, optimizer, epoch, cfg, meta_cfg)
            print(f"[Stage2] saved checkpoint at epoch {epoch}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
