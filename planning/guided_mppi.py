import torch
import numpy as np
from einops import rearrange, repeat

from .base_planner import BasePlanner
from utils import move_to_device


class GuidedMPPIPlanner(BasePlanner):
    """
    Demonstration-guided residual MPPI.

    The input `actions` is treated as a normalized expert/demo prior with shape
    (B, H, action_dim). The planner samples bounded residuals in physical action
    space and scores candidates with the learned world model objective.
    """

    def __init__(
        self,
        horizon,
        num_samples,
        opt_steps,
        eval_every,
        temperature,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="guided_mppi",
        log_filename="logs.json",
        sigma_steer=0.04,
        sigma_throttle=0.06,
        clip_steer=0.12,
        clip_throttle=0.18,
        residual_weight=0.2,
        smooth_weight=0.05,
        random_sample_ratio=0.0,
        random_sigma_steer=0.35,
        random_sigma_throttle=0.45,
        random_clip_steer=0.85,
        random_clip_throttle=0.9,
        prior_rank_weight=0.08,
        random_branch_penalty=0.6,
        eval_batch_size=128,
        return_best=True,
        save_iter_images=True,
        save_iter_video=False,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.opt_steps = int(opt_steps)
        self.eval_every = int(eval_every)
        self.temperature = float(temperature)
        self.logging_prefix = logging_prefix
        self.sigma_steer = float(sigma_steer)
        self.sigma_throttle = float(sigma_throttle)
        self.clip_steer = float(clip_steer)
        self.clip_throttle = float(clip_throttle)
        self.residual_weight = float(residual_weight)
        self.smooth_weight = float(smooth_weight)
        self.random_sample_ratio = float(random_sample_ratio)
        self.random_sigma_steer = float(random_sigma_steer)
        self.random_sigma_throttle = float(random_sigma_throttle)
        self.random_clip_steer = float(random_clip_steer)
        self.random_clip_throttle = float(random_clip_throttle)
        self.prior_rank_weight = float(prior_rank_weight)
        self.random_branch_penalty = float(random_branch_penalty)
        self.eval_batch_size = int(eval_batch_size)
        self.return_best = bool(return_best)
        self.save_iter_images = bool(save_iter_images)
        self.save_iter_video = bool(save_iter_video)
        self.last_selected_physical = None

    def _zero_prior(self, obs_0):
        n_evals = obs_0["visual"].shape[0]
        return torch.zeros(n_evals, self.horizon, self.action_dim, device=self.device)

    def _fit_prior(self, obs_0, actions):
        if actions is None:
            prior = self._zero_prior(obs_0)
        else:
            prior = actions.to(self.device)
        time_dim = 2 if prior.ndim == 4 else 1
        if prior.shape[time_dim] < self.horizon:
            pad_shape = list(prior.shape)
            pad_shape[time_dim] = self.horizon - prior.shape[time_dim]
            pad = torch.zeros(
                *pad_shape,
                device=prior.device,
                dtype=prior.dtype,
            )
            prior = torch.cat([prior, pad], dim=time_dim)
        if prior.ndim == 4:
            return prior[:, :, : self.horizon]
        return prior[:, : self.horizon]

    def _normalized_macro_to_physical(self, actions):
        frameskip = int(self.evaluator.frameskip)
        per_step = rearrange(
            actions.detach().cpu(), "b t (f d) -> b (t f) d", f=frameskip, d=2
        )
        physical = self.preprocessor.denormalize_actions(per_step).to(self.device)
        physical = torch.clamp(physical, -1.0, 1.0)
        return rearrange(
            physical, "b (t f) d -> b t (f d)", t=actions.shape[1], f=frameskip
        )

    def _physical_macro_to_normalized(self, physical):
        frameskip = int(self.evaluator.frameskip)
        per_step = rearrange(
            physical.detach().cpu(), "b t (f d) -> b (t f) d", f=frameskip, d=2
        )
        normalized = self.preprocessor.normalize_actions(per_step)
        normalized = rearrange(
            normalized, "b (t f) d -> b t (f d)", t=physical.shape[1], f=frameskip
        )
        return normalized.to(self.device)

    def _regularization_loss(self, candidate_physical, reference_physical):
        residual = candidate_physical - reference_physical
        loss = self.residual_weight * residual.pow(2).mean(dim=(1, 2))
        if candidate_physical.shape[1] > 1 and self.smooth_weight > 0:
            diff = candidate_physical[:, 1:] - candidate_physical[:, :-1]
            loss = loss + self.smooth_weight * diff.pow(2).mean(dim=(1, 2))
        return loss

    def _wm_losses(self, trans_obs_0, z_obs_g, actions):
        losses = []
        with torch.no_grad():
            for start in range(0, actions.shape[0], self.eval_batch_size):
                end = min(start + self.eval_batch_size, actions.shape[0])
                obs_chunk = {key: value[start:end] for key, value in trans_obs_0.items()}
                goal_chunk = {key: value[start:end] for key, value in z_obs_g.items()}
                i_z_obses, _ = self.wm.rollout(obs_0=obs_chunk, act=actions[start:end])
                losses.append(self.objective_fn(i_z_obses, goal_chunk).detach())
                del obs_chunk, goal_chunk, i_z_obses
        return torch.cat(losses, dim=0)

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        with torch.no_grad():
            z_obs_g = {key: value.detach() for key, value in self.wm.encode_obs(trans_obs_g).items()}

        prior_norm = self._fit_prior(obs_0, actions)
        if prior_norm.ndim == 4:
            n_evals, n_priors = prior_norm.shape[:2]
            flat_prior = prior_norm.reshape(n_evals * n_priors, self.horizon, self.action_dim)
            prior_bank_physical = self._normalized_macro_to_physical(flat_prior).reshape(
                n_evals, n_priors, self.horizon, self.action_dim
            )
        else:
            n_evals = prior_norm.shape[0]
            n_priors = 1
            prior_bank_physical = self._normalized_macro_to_physical(prior_norm).unsqueeze(1)
        prior_physical = prior_bank_physical[:, 0]

        noise_scale = torch.empty(self.action_dim, device=self.device)
        noise_scale[0::2] = self.sigma_steer
        noise_scale[1::2] = self.sigma_throttle
        noise_clip = torch.empty(self.action_dim, device=self.device)
        noise_clip[0::2] = self.clip_steer
        noise_clip[1::2] = self.clip_throttle
        random_scale = torch.empty(self.action_dim, device=self.device)
        random_scale[0::2] = self.random_sigma_steer
        random_scale[1::2] = self.random_sigma_throttle
        random_clip = torch.empty(self.action_dim, device=self.device)
        random_clip[0::2] = self.random_clip_steer
        random_clip[1::2] = self.random_clip_throttle
        num_random = int(round(self.num_samples * self.random_sample_ratio))
        num_random = max(0, min(self.num_samples - 1, num_random))
        num_guided = self.num_samples - num_random

        selected_physical = prior_physical.detach().clone()
        best_physical = prior_physical.detach().clone()
        best_losses = torch.full((n_evals,), float("inf"), device=self.device)
        if num_random > 0:
            print(
                f"[Guided MPPI] {self.logging_prefix} mixture "
                f"guided={num_guided} random={num_random} priors={n_priors} "
                f"residual_weight={self.residual_weight:.4f} "
                f"rank_weight={self.prior_rank_weight:.4f} "
                f"random_penalty={self.random_branch_penalty:.4f}",
                flush=True,
            )
        elif n_priors > 1:
            print(
                f"[Guided MPPI] {self.logging_prefix} prior_bank priors={n_priors}",
                flush=True,
            )

        for opt_step in range(self.opt_steps):
            step_losses = []
            for traj in range(n_evals):
                cur_obs = {
                    key: repeat(value[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, value in trans_obs_0.items()
                }
                cur_goal = {
                    key: repeat(value[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, value in z_obs_g.items()
                }

                centers = prior_bank_physical[traj]
                center_ids = torch.arange(num_guided, device=self.device) % n_priors
                center_physical = centers[center_ids]
                noise = torch.randn(
                    num_guided,
                    self.horizon,
                    self.action_dim,
                    device=self.device,
                ) * noise_scale
                noise[: min(n_priors, num_guided)].zero_()
                guided_physical = torch.clamp(center_physical + noise, -1.0, 1.0)
                guided_physical[0] = selected_physical[traj]
                guided_reference = center_physical
                guided_reference[0] = prior_physical[traj]
                if num_random > 0:
                    random_physical = torch.randn(
                        num_random,
                        self.horizon,
                        self.action_dim,
                        device=self.device,
                    ) * random_scale
                    random_physical = torch.clamp(random_physical, -random_clip, random_clip)
                    candidate_physical = torch.cat([guided_physical, random_physical], dim=0)
                    random_reference = prior_physical[traj].unsqueeze(0).expand_as(random_physical)
                    reference_physical = torch.cat([guided_reference, random_reference], dim=0)
                    prior_penalty = torch.cat(
                        [
                            center_ids.float() * self.prior_rank_weight,
                            torch.full(
                                (num_random,),
                                self.random_branch_penalty,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    )
                else:
                    candidate_physical = guided_physical
                    reference_physical = guided_reference
                    prior_penalty = center_ids.float() * self.prior_rank_weight
                candidate_actions = self._physical_macro_to_normalized(candidate_physical)

                wm_loss = self._wm_losses(cur_obs, cur_goal, candidate_actions)
                reg_loss = self._regularization_loss(
                    candidate_physical, reference_physical
                )
                total_loss = wm_loss + reg_loss + prior_penalty

                best_idx = int(torch.argmin(total_loss).item())
                if total_loss[best_idx] < best_losses[traj]:
                    best_losses[traj] = total_loss[best_idx].detach()
                    best_physical[traj] = candidate_physical[best_idx].detach()

                beta = torch.min(total_loss)
                weights = torch.softmax(
                    -(total_loss - beta) / max(self.temperature, 1e-6), dim=0
                )
                candidate_delta = candidate_physical - prior_physical[traj].unsqueeze(0)
                weighted_delta = torch.sum(weights.view(-1, 1, 1) * candidate_delta, dim=0)
                selected_physical[traj] = torch.clamp(
                    prior_physical[traj] + torch.clamp(weighted_delta, -noise_clip, noise_clip),
                    -1.0,
                    1.0,
                )
                step_losses.append(float(total_loss[best_idx].detach().cpu().item()))

                print(
                    f"[Guided MPPI] {self.logging_prefix} traj={traj} "
                    f"step={opt_step + 1}/{self.opt_steps} "
                    f"best={float(total_loss[best_idx]):.6f} "
                    f"wm={float(wm_loss[best_idx]):.6f} "
                    f"reg={float(reg_loss[best_idx]):.6f}",
                    flush=True,
                )

                del cur_obs, cur_goal, noise, guided_physical, candidate_physical, candidate_actions
                del center_ids, center_physical, guided_reference, reference_physical, prior_penalty
                del wm_loss, reg_loss, total_loss

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(step_losses), "step": opt_step + 1}
            )
            if (
                self.evaluator is not None
                and self.save_iter_images
                and (opt_step + 1) % max(self.eval_every, 1) == 0
            ):
                eval_physical = best_physical if self.return_best else selected_physical
                eval_actions = self._physical_macro_to_normalized(eval_physical)
                logs, _, _, _ = self.evaluator.eval_actions(
                    eval_actions.detach(),
                    filename=f"{self.logging_prefix}_output_{opt_step + 1}",
                    save_video=self.save_iter_video,
                )
                logs = {f"{self.logging_prefix}/{key}": value for key, value in logs.items()}
                self.wandb_run.log(logs)
                print(
                    f"[Guided MPPI] rollout image saved: "
                    f"{self.logging_prefix}_output_{opt_step + 1}.png",
                    flush=True,
                )
                del eval_actions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        planned_physical = best_physical if self.return_best else selected_physical
        diff = (planned_physical - prior_physical).detach().cpu().numpy()
        print(
            f"[Guided MPPI] {self.logging_prefix} selected "
            f"mean_abs_delta={np.abs(diff).mean():.4f} "
            f"max_abs_delta={np.abs(diff).max():.4f}",
            flush=True,
        )
        self.last_selected_physical = planned_physical.detach().cpu()
        return self._physical_macro_to_normalized(planned_physical), np.full(n_evals, np.inf)
