import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import repeat
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from test_expert_noise import build_workspace
from utils import cfg_to_dict, move_to_device


def _as_float(value, default):
    return default if value is None else float(value)


def _as_int(value, default):
    return default if value is None else int(value)


def _phase_prior_physical(workspace, goal_obs, horizon):
    if "expert_action_segment" not in goal_obs:
        raise KeyError("expert_action_segment is missing; use expert trace subgoals.")

    n_steps = horizon * int(workspace.frameskip)
    segment = np.asarray(goal_obs["expert_action_segment"][:, 0], dtype=np.float32)
    if segment.shape[1] != n_steps:
        fixed = np.zeros((segment.shape[0], n_steps, 2), dtype=np.float32)
        take = min(segment.shape[1], n_steps)
        fixed[:, :take] = segment[:, :take]
        segment = fixed
    return segment.reshape(segment.shape[0], horizon, int(workspace.frameskip) * 2)


def _physical_macro_to_normalized(workspace, physical_macro):
    return workspace._normalize_physical_macro_actions(
        torch.as_tensor(physical_macro, dtype=torch.float32)
    )


def _rollout_objective_losses(workspace, cur_obs, goal_obs, actions, eval_batch_size):
    device = workspace.device
    trans_obs_0 = move_to_device(workspace.data_preprocessor.transform_obs(cur_obs), device)
    trans_obs_g = move_to_device(workspace.data_preprocessor.transform_obs(goal_obs), device)

    with torch.no_grad():
        z_obs_g = workspace.wm.encode_obs(trans_obs_g)

    if actions.shape[0] != trans_obs_0["visual"].shape[0]:
        if trans_obs_0["visual"].shape[0] != 1:
            raise ValueError("guided MPPI currently expects n_evals=1 or one action per eval.")
        trans_obs_0 = {
            key: repeat(value, "1 ... -> n ...", n=actions.shape[0])
            for key, value in trans_obs_0.items()
        }
        z_obs_g = {
            key: repeat(value, "1 ... -> n ...", n=actions.shape[0])
            for key, value in z_obs_g.items()
        }

    chunks = []
    with torch.no_grad():
        for start in range(0, actions.shape[0], int(eval_batch_size)):
            end = min(start + int(eval_batch_size), actions.shape[0])
            obs_chunk = {key: value[start:end] for key, value in trans_obs_0.items()}
            goal_chunk = {key: value[start:end] for key, value in z_obs_g.items()}
            i_z_obses, _ = workspace.wm.rollout(obs_0=obs_chunk, act=actions[start:end])
            chunks.append(workspace.objective_fn(i_z_obses, goal_chunk).detach())
            del obs_chunk, goal_chunk, i_z_obses
    return torch.cat(chunks, dim=0)


def _regularization_loss(candidate_physical, prior_physical, residual_weight, smooth_weight):
    residual = candidate_physical - prior_physical
    loss = residual_weight * residual.pow(2).mean(dim=(1, 2))
    if candidate_physical.shape[1] > 1:
        diff = candidate_physical[:, 1:] - candidate_physical[:, :-1]
        loss = loss + smooth_weight * diff.pow(2).mean(dim=(1, 2))
    return loss


def guided_mppi_phase(workspace, phase_idx, cur_obs, goal_obs, prior_physical, cfg):
    num_samples = _as_int(cfg.get("num_samples"), 512)
    opt_steps = _as_int(cfg.get("opt_steps"), 5)
    temperature = _as_float(cfg.get("temperature"), 0.7)
    eval_batch_size = _as_int(cfg.get("eval_batch_size"), 128)
    residual_weight = _as_float(cfg.get("residual_weight"), 0.2)
    smooth_weight = _as_float(cfg.get("smooth_weight"), 0.05)
    sigma_steer = _as_float(cfg.get("sigma_steer"), 0.04)
    sigma_throttle = _as_float(cfg.get("sigma_throttle"), 0.06)
    clip_steer = _as_float(cfg.get("clip_steer"), 0.12)
    clip_throttle = _as_float(cfg.get("clip_throttle"), 0.18)
    return_best = bool(cfg.get("return_best", True))

    if prior_physical.shape[0] != 1:
        raise ValueError("test_guided_mppi.py currently supports n_evals=1 for clear debugging.")

    device = workspace.device
    prior = torch.as_tensor(prior_physical, dtype=torch.float32, device=device)
    mu_delta = torch.zeros_like(prior)
    best_physical = prior.detach().clone()
    best_loss = torch.tensor(float("inf"), device=device)

    noise_scale = torch.empty(prior.shape[-1], device=device)
    noise_scale[0::2] = sigma_steer
    noise_scale[1::2] = sigma_throttle
    noise_clip = torch.empty(prior.shape[-1], device=device)
    noise_clip[0::2] = clip_steer
    noise_clip[1::2] = clip_throttle

    for step in range(opt_steps):
        noise = torch.randn(num_samples, *prior.shape[1:], device=device) * noise_scale
        noise[0].zero_()
        delta = torch.clamp(mu_delta[0].unsqueeze(0) + noise, -noise_clip, noise_clip)
        candidate_physical = torch.clamp(prior[0].unsqueeze(0) + delta, -1.0, 1.0)

        actions = _physical_macro_to_normalized(workspace, candidate_physical.detach().cpu()).to(device)
        wm_loss = _rollout_objective_losses(
            workspace, cur_obs, goal_obs, actions, eval_batch_size=eval_batch_size
        )
        reg_loss = _regularization_loss(
            candidate_physical, prior[0].unsqueeze(0), residual_weight, smooth_weight
        )
        total_loss = wm_loss + reg_loss

        step_best_idx = int(torch.argmin(total_loss).item())
        if total_loss[step_best_idx] < best_loss:
            best_loss = total_loss[step_best_idx].detach()
            best_physical = candidate_physical[step_best_idx : step_best_idx + 1].detach().clone()

        beta = torch.min(total_loss)
        weights = torch.softmax(-(total_loss - beta) / max(temperature, 1e-6), dim=0)
        mu_delta[0] = torch.sum(weights.view(-1, 1, 1) * delta, dim=0)
        mu_delta[0] = torch.clamp(mu_delta[0], -noise_clip, noise_clip)

        print(
            f"[Guided MPPI] phase={phase_idx} step={step + 1}/{opt_steps} "
            f"best_loss={float(total_loss[step_best_idx]):.6f} "
            f"wm={float(wm_loss[step_best_idx]):.6f} "
            f"reg={float(reg_loss[step_best_idx]):.6f} "
            f"mean|delta|={float(delta.abs().mean()):.4f}",
            flush=True,
        )

        del actions, wm_loss, reg_loss, total_loss, noise, delta, candidate_physical
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected = best_physical if return_best else torch.clamp(prior + mu_delta, -1.0, 1.0)
    diff = (selected - prior).detach().cpu().numpy()
    print(
        f"[Guided MPPI] phase={phase_idx} selected "
        f"mean_abs_delta={np.abs(diff).mean():.4f} max_abs_delta={np.abs(diff).max():.4f}",
        flush=True,
    )
    return _physical_macro_to_normalized(workspace, selected.detach().cpu()).to(device), selected.detach().cpu().numpy()


def evaluate_phase(workspace, phase_idx, cur_obs, goal_obs, goal_state, actions, accumulated, save_video):
    workspace.evaluator.history_actions = accumulated.detach() if accumulated is not None else None
    workspace.evaluator.assign_init_cond(obs_0=cur_obs, state_0=workspace.state_0)
    workspace.evaluator.assign_goal_cond(obs_g=goal_obs, state_g=goal_state)

    action_len = np.full(actions.shape[0], actions.shape[1], dtype=float)
    logs, successes, e_obses, e_states = workspace.evaluator.eval_actions(
        actions.detach(),
        action_len=action_len,
        filename=f"guided_mppi_phase{phase_idx}",
        save_video=save_video,
    )
    e_final_obs = workspace.evaluator._get_trajdict_last_n(
        e_obses, action_len * workspace.frameskip + 1, n=3
    )
    e_final_state = workspace.evaluator._get_traj_last(
        e_states, action_len * workspace.frameskip + 1
    )[:, 0]
    return logs, successes, e_final_obs, e_final_state


def run_guided_mppi(workspace, cfg):
    phase_goal_obs_list = list(workspace.obs_g_sub_list) + [workspace.obs_g]
    phase_goal_state_list = list(workspace.state_g_sub_list) + [workspace.state_g]
    horizon = int(cfg.get("horizon", workspace.phase_H))
    save_phase_video = bool(cfg.get("save_phase_video", False))
    save_final_video = bool(cfg.get("save_final_video", True))

    cur_obs = workspace.obs_0
    cur_state = workspace.state_0
    all_actions = []
    phase_rows = []

    for phase_idx, (goal_obs, goal_state) in enumerate(zip(phase_goal_obs_list, phase_goal_state_list)):
        target_name = "final goal" if phase_idx == len(phase_goal_obs_list) - 1 else f"subgoal {phase_idx + 1}"
        print(f"\n=== Guided MPPI Phase {phase_idx}: approach {target_name}/{len(phase_goal_obs_list)} ===")

        accumulated = torch.cat(all_actions, dim=1) if all_actions else None
        if hasattr(workspace.wm, "init_actions"):
            workspace.wm.init_actions = accumulated[:, -2:, :].detach().cpu() if accumulated is not None and accumulated.shape[1] >= 2 else None

        prior_physical = _phase_prior_physical(workspace, goal_obs, horizon)
        phase_actions, selected_physical = guided_mppi_phase(
            workspace, phase_idx, cur_obs, goal_obs, prior_physical, cfg
        )
        logs, successes, cur_obs, cur_state = evaluate_phase(
            workspace,
            phase_idx,
            cur_obs,
            goal_obs,
            goal_state,
            phase_actions,
            accumulated,
            save_video=save_phase_video,
        )

        row = {key: float(value) for key, value in logs.items()}
        row["phase"] = phase_idx
        row["success_rate"] = float(np.mean(successes.astype(float)))
        row["selected_mean_abs_delta"] = float(np.abs(selected_physical - prior_physical).mean())
        row["selected_max_abs_delta"] = float(np.abs(selected_physical - prior_physical).max())
        phase_rows.append(row)
        all_actions.append(phase_actions)

    actions = torch.cat(all_actions, dim=1)
    action_len = np.full(actions.shape[0], actions.shape[1], dtype=float)
    workspace.evaluator.history_actions = None
    workspace.evaluator.assign_init_cond(obs_0=workspace.obs_0, state_0=workspace.state_0)
    workspace.evaluator.assign_goal_cond(obs_g=workspace.obs_g, state_g=workspace.state_g)
    if hasattr(workspace.wm, "init_actions"):
        workspace.wm.init_actions = None
    final_logs, final_successes, _, _ = workspace.evaluator.eval_actions(
        actions.detach(),
        action_len=action_len,
        filename="guided_mppi_final",
        save_video=save_final_video,
    )
    final_row = {key: float(value) for key, value in final_logs.items()}
    final_row["success_rate"] = float(np.mean(final_successes.astype(float)))
    return phase_rows, final_row


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
    guided_cfg = cfg_dict.get("guided_mppi", {})
    workspace = build_workspace(cfg_dict)
    try:
        phase_rows, final_row = run_guided_mppi(workspace, guided_cfg)
        output_path = Path("guided_mppi_results.json")
        output_path.write_text(
            json.dumps({"phases": phase_rows, "final": final_row}, indent=2),
            encoding="utf-8",
        )

        print("\n[Guided MPPI] final summary")
        print(
            f"success={final_row['success_rate']:.3f} "
            f"dist={final_row.get('mean_distance', float('nan')):.3f} "
            f"head={final_row.get('mean_heading_error', float('nan')):.3f} "
            f"visual_div={final_row.get('mean_div_visual_emb', float('nan')):.3f} "
            f"proprio_div={final_row.get('mean_div_proprio_emb', float('nan')):.3f}"
        )
        print(f"[Guided MPPI] wrote {output_path.resolve()}")
    finally:
        try:
            workspace.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
