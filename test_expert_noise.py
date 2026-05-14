import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from plan_park_3 import (
    MAP_CENTER,
    RENDER_RES,
    MultiAgentParkingLotEnv,
    ParkingDinoWrapper,
    PlanWorkspace,
    load_model,
)
from utils import cfg_to_dict, seed


class ClampedWM:
    def __init__(self, original_wm, n_min, n_max):
        self.original_wm = original_wm
        self.n_min = n_min
        self.n_max = n_max
        self.init_actions = None

    def rollout(self, obs_0, act, **kwargs):
        clipped_act = torch.max(torch.min(act, self.n_max), self.n_min)
        num_hist = obs_0["visual"].shape[1]
        batch_size, _, action_dim = clipped_act.shape

        if self.init_actions is not None:
            init_prev = self.init_actions.to(device=clipped_act.device, dtype=clipped_act.dtype)
            if init_prev.shape[0] < batch_size:
                ratio = batch_size // init_prev.shape[0]
                init_prev = init_prev.repeat(ratio, 1, 1)
                remainder = batch_size - init_prev.shape[0]
                if remainder > 0:
                    init_prev = torch.cat([init_prev, init_prev[:remainder]], dim=0)
            init_act = torch.cat([init_prev, init_prev[:, -1:, :]], dim=1)
        else:
            init_act = torch.zeros(batch_size, num_hist, action_dim, device=clipped_act.device)

        full_act = torch.cat([init_act, clipped_act], dim=1)
        z_obses, z = self.original_wm.rollout(obs_0, full_act, **kwargs)
        target_len = clipped_act.shape[1] + 1
        sliced_z_obses = {
            key: value[:, num_hist - 1 : num_hist - 1 + target_len]
            for key, value in z_obses.items()
        }
        return sliced_z_obses, z

    def __getattr__(self, name):
        return getattr(self.original_wm, name)


def _set_nested(cfg_dict, keys, value):
    cur = cfg_dict
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def build_workspace(cfg_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    model_cfg = OmegaConf.load(os.path.join(model_path, "hydra.yaml"))

    seed(cfg_dict["seed"])
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = traj_dsets["valid"]
    real_dset = datasets["train"]
    dset.action_mean = real_dset.action_mean
    dset.action_std = real_dset.action_std
    dset.state_mean = real_dset.state_mean
    dset.state_std = real_dset.state_std
    dset.proprio_mean = real_dset.proprio_mean
    dset.proprio_std = real_dset.proprio_std

    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)

    act_mean = dset.action_mean.to(device)
    act_std = dset.action_std.to(device)
    norm_min = (torch.tensor([-1.0] * dset.action_dim, device=device) - act_mean) / act_std
    norm_max = (torch.tensor([1.0] * dset.action_dim, device=device) - act_mean) / act_std
    model = ClampedWM(model, norm_min, norm_max)

    raw_action_mean = dset.raw_action_mean.numpy()
    raw_action_std = dset.raw_action_std.numpy()
    frameskip = int(model_cfg.frameskip)

    def create_wrapped_env():
        env_config = {
            "use_render": False,
            "num_agents": 1,
            "start_seed": 400,
            "allow_respawn": False,
            "window_size": RENDER_RES,
            "out_of_road_done": False,
            "crash_vehicle_done": False,
            "vehicle_config": {"lidar": {"num_lasers": 0}, "show_navi_mark": False},
        }
        env = MultiAgentParkingLotEnv(env_config)
        return ParkingDinoWrapper(
            env,
            frameskip=frameskip,
            action_mean=raw_action_mean,
            action_std=raw_action_std,
        )

    env = SubprocVectorEnv([create_wrapped_env for _ in range(cfg_dict["n_evals"])])
    return PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=frameskip,
        wandb_run=None,
    )


def collect_expert_trace_physical(workspace):
    phase_goal_obs_list = list(workspace.obs_g_sub_list) + [workspace.obs_g]
    segments = []
    for phase_idx, goal_obs in enumerate(phase_goal_obs_list):
        if "expert_action_segment" not in goal_obs:
            raise KeyError(f"phase {phase_idx} has no expert_action_segment; use expert_oracle.source=trace.")
        segment = np.asarray(goal_obs["expert_action_segment"][:, 0], dtype=np.float32)
        segments.append(segment)

    physical_steps = np.concatenate(segments, axis=1)
    horizon = physical_steps.shape[1] // int(workspace.frameskip)
    physical_macro = physical_steps.reshape(
        physical_steps.shape[0],
        horizon,
        int(workspace.frameskip) * 2,
    )
    return physical_macro


def evaluate_variant(workspace, name, physical_macro, save_video):
    actions = workspace._normalize_physical_macro_actions(
        torch.tensor(physical_macro, dtype=torch.float32)
    )
    action_len = np.full(actions.shape[0], actions.shape[1], dtype=float)

    wm = workspace.planner.sub_planner.wm if hasattr(workspace.planner, "sub_planner") else workspace.wm
    if hasattr(wm, "init_actions"):
        wm.init_actions = None

    workspace.evaluator.history_actions = None
    workspace.evaluator.assign_init_cond(obs_0=workspace.obs_0, state_0=workspace.state_0)
    workspace.evaluator.assign_goal_cond(obs_g=workspace.obs_g, state_g=workspace.state_g)
    logs, successes, _, _ = workspace.evaluator.eval_actions(
        actions.detach(),
        action_len,
        filename=f"noise_probe_{name}",
        save_video=save_video,
    )
    row = {key: float(value) for key, value in logs.items()}
    row["name"] = name
    row["success_rate"] = float(np.mean(successes.astype(float)))
    return row


def make_variants(base_physical, noise_stds, random_count, rng):
    variants = [("expert", base_physical.copy())]
    for std in noise_stds:
        if float(std) <= 0:
            continue
        noisy = base_physical + rng.normal(0.0, float(std), size=base_physical.shape).astype(np.float32)
        variants.append((f"expert_noise_{float(std):.3f}".replace(".", "p"), np.clip(noisy, -1.0, 1.0)))

    for idx in range(int(random_count)):
        random_actions = rng.uniform(-1.0, 1.0, size=base_physical.shape).astype(np.float32)
        variants.append((f"random_uniform_{idx}", random_actions))
    return variants


@hydra.main(config_path="conf", config_name="plan_park", version_base=None)
def main(cfg: OmegaConf):
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    with open_dict(cfg):
        cfg["saved_folder"] = output_dir
        cfg["wandb_logging"] = False
        cfg["n_evals"] = int(cfg.get("n_evals", 1))
        cfg.setdefault("diagnostics", {})
        cfg["diagnostics"]["save_expert_video"] = False
        cfg.setdefault("expert_oracle", {})
        cfg["expert_oracle"]["enabled"] = True
        cfg["expert_oracle"]["source"] = "trace"
        cfg["expert_oracle"]["save_video"] = False

    cfg_dict = cfg_to_dict(cfg)
    probe_cfg = cfg_dict.get("noise_probe", {})
    noise_stds = probe_cfg.get("noise_stds", [0.02, 0.05, 0.1, 0.2, 0.4])
    random_count = int(probe_cfg.get("random_count", 2))
    save_video = bool(probe_cfg.get("save_video", True))
    rng = np.random.RandomState(int(probe_cfg.get("seed", cfg_dict["seed"] + 1000)))

    workspace = build_workspace(cfg_dict)
    try:
        base_physical = collect_expert_trace_physical(workspace)
        print(
            f"[Noise Probe] expert physical macro shape={base_physical.shape}, "
            f"physical_steps={base_physical.shape[1] * workspace.frameskip}",
            flush=True,
        )

        rows = []
        for name, physical_macro in make_variants(base_physical, noise_stds, random_count, rng):
            diff = np.abs(physical_macro - base_physical)
            print(
                f"\n[Noise Probe] evaluating {name}: "
                f"mean_abs_delta={diff.mean():.4f} max_abs_delta={diff.max():.4f}",
                flush=True,
            )
            row = evaluate_variant(workspace, name, physical_macro, save_video=save_video)
            row["mean_abs_action_delta"] = float(diff.mean())
            row["max_abs_action_delta"] = float(diff.max())
            rows.append(row)

        output_path = Path("expert_noise_probe_results.json")
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        print("\n[Noise Probe] summary")
        print("name | success | env_dist | env_head | wm_env_visual_div | wm_env_proprio_div")
        for row in rows:
            print(
                f"{row['name']} | {row['success_rate']:.3f} | "
                f"{row.get('mean_distance', float('nan')):.3f} | "
                f"{row.get('mean_heading_error', float('nan')):.3f} | "
                f"{row.get('mean_div_visual_emb', float('nan')):.3f} | "
                f"{row.get('mean_div_proprio_emb', float('nan')):.3f}"
            )
        print(f"[Noise Probe] wrote {output_path.resolve()}")
    finally:
        try:
            workspace.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
