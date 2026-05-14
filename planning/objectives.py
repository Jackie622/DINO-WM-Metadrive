import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(
    alpha,
    base=2,
    mode="last",
    speed_weight=50.0,
    terminal_weight=3.0,
    best_weight=1.0,
    progress_weight=2.0,
    time_weight=0.25,
):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    def _target_with_time(z_tgt, z_pred):
        """Make target broadcast explicit for (B, T, ...) prediction tensors."""
        if z_tgt.ndim == z_pred.ndim - 1:
            z_tgt = z_tgt.unsqueeze(1)
        if z_tgt.shape[1] == 1 and z_pred.shape[1] != 1:
            z_tgt = z_tgt.expand(-1, z_pred.shape[1], *([-1] * (z_pred.ndim - 2)))
        return z_tgt

    def _loss_by_time(pred, tgt):
        tgt = _target_with_time(tgt, pred)
        return metric(pred, tgt).mean(dim=tuple(range(2, pred.ndim)))

    def objective_fn_fast(z_obs_pred, z_obs_tgt):
        """
        Fast arrival objective: time-weighted loss + terminal speed penalty.
        Encourages reaching the goal state quickly with zero final speed.
        - Higher base → more weight on later timesteps (must reach goal early and stay)
        - speed_weight > 0 → penalizes non-zero speed at the last predicted frame
        """
        T = z_obs_pred["visual"].shape[1]
        coeffs = np.array([base**i for i in range(T)], dtype=np.float32)
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)

        # Terminal speed penalty: proprio[..., 0] is ego speed (m/s)
        # Goal state has speed=0, so penalize any residual speed at the last frame
        if speed_weight > 0:
            terminal_speed = z_obs_pred["proprio"][:, -1, 0]  # (B,)
            speed_penalty = speed_weight * (terminal_speed ** 2)
        else:
            speed_penalty = 0.0

        loss = loss_visual + alpha * loss_proprio + speed_penalty
        return loss

    def objective_fn_progress(z_obs_pred, z_obs_tgt):
        """
        Goal-seeking objective for short-horizon parking MPC.

        The old "all/fast" objectives make every predicted frame match the
        subgoal. With H=2/3 this can over-favor low-motion predictions,
        especially when a terminal speed penalty is large. This objective is
        more direct:
        - strongly minimize terminal target distance;
        - also reward the best target approach anywhere in the horizon;
        - penalize candidates whose target distance increases over time;
        - keep terminal speed penalty weak and configurable.
        """
        T = z_obs_pred["visual"].shape[1]
        coeffs = np.array([base**i for i in range(T)], dtype=np.float32)
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

        visual_t = _loss_by_time(z_obs_pred["visual"], z_obs_tgt["visual"])
        proprio_t = _loss_by_time(z_obs_pred["proprio"], z_obs_tgt["proprio"])
        target_t = visual_t + alpha * proprio_t

        terminal_loss = target_t[:, -1]
        best_loss = target_t.min(dim=1).values
        weighted_loss = (target_t * coeffs).sum(dim=1)

        if T > 1 and progress_weight > 0:
            progress_bad = torch.relu(target_t[:, 1:] - target_t[:, :-1]).mean(dim=1)
        else:
            progress_bad = torch.zeros_like(terminal_loss)

        if speed_weight > 0:
            terminal_speed = z_obs_pred["proprio"][:, -1, 0]
            speed_penalty = speed_weight * (terminal_speed ** 2)
        else:
            speed_penalty = 0.0

        loss = (
            terminal_weight * terminal_loss
            + best_weight * best_loss
            + time_weight * weighted_loss
            + progress_weight * progress_bad
            + speed_penalty
        )
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    elif mode == "fast":
        return objective_fn_fast
    elif mode == "progress":
        return objective_fn_progress
    else:
        raise NotImplementedError
