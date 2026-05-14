import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        sigma_momentum=0.5,     # momentum for adaptive variance (0=no smoothing, 1=frozen)
        sigma_min=0.05,         # floor for variance to prevent premature convergence
        elite_weighted=True,    # weighted elite average (vs plain mean)
        topk_min=10,            # minimum elite count (dynamic scheduling)
        topk_max=60,            # maximum elite count (dynamic scheduling)
        action_templates=True,  # seed CEM with a few driving-prior action sequences
        template_frac=0.08,     # fraction of samples reserved for templates
        template_noise=0.05,    # normalized-space noise added around templates
        return_best=True,       # execute the best sampled sequence instead of averaged mu
        eval_batch_size=None,   # chunk WM rollouts to reduce peak CUDA memory
        action_smooth_weight=0.0,
        action_intra_macro_weight=0.0,
        action_bound_weight=0.0,
        action_mag_weight=0.0,
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
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.sigma_momentum = sigma_momentum
        self.sigma_min = sigma_min
        self.elite_weighted = elite_weighted
        self.topk_min = topk_min
        self.topk_max = topk_max
        self.action_templates = action_templates
        self.template_frac = template_frac
        self.template_noise = template_noise
        self.return_best = return_best
        self.eval_batch_size = eval_batch_size
        self.action_smooth_weight = float(action_smooth_weight)
        self.action_intra_macro_weight = float(action_intra_macro_weight)
        self.action_bound_weight = float(action_bound_weight)
        self.action_mag_weight = float(action_mag_weight)

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def _update_elite(self, action, loss, sigma_prev, opt_step=None):
        """Update mu and sigma using weighted elite selection with adaptive variance.

        Dynamic elite scheduling: when loss drops quickly, shrink elite set for faster
        convergence; when loss plateaus, expand elite set for more exploration.
        """
        # Dynamic elite count: interpolate between topk_min and topk_max
        # based on loss variance across the population.
        if opt_step is not None and self.topk_max > self.topk_min:
            # Use loss spread (max-min) relative to median as convergence signal
            with torch.no_grad():
                loss_median = loss.median()
                loss_spread = (loss.max() - loss.min()).abs()
                # spread/median ratio: large → high uncertainty → more elites
                ratio = (loss_spread / (loss_median + 1e-8)).item()
                ratio = np.clip(ratio, 0.2, 2.0)
                # Map: ratio 2.0 → topk_max, ratio 0.2 → topk_min
                alpha = (ratio - 0.2) / (2.0 - 0.2)
                current_topk = int(self.topk_min + alpha * (self.topk_max - self.topk_min))
                current_topk = max(min(current_topk, action.shape[0]), 1)
        else:
            current_topk = self.topk

        topk_idx = torch.argsort(loss)[:current_topk]
        topk_action = action[topk_idx]
        topk_loss = loss[topk_idx]

        if self.elite_weighted and current_topk > 1:
            # Weighted average: weight each elite sample inversely proportional to its loss
            weights = torch.softmax(-topk_loss, dim=0)  # (topk,)
            mu_new = (topk_action * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            mu_new = topk_action.mean(dim=0)

        sigma_new = topk_action.std(dim=0)

        # Adaptive variance: momentum smoothing + floor clipping
        if sigma_prev is not None and self.sigma_momentum > 0:
            sigma_out = (1 - self.sigma_momentum) * sigma_new + self.sigma_momentum * sigma_prev
        else:
            sigma_out = sigma_new

        sigma_out = torch.clamp(sigma_out, min=self.sigma_min)

        return mu_new, sigma_out

    def _normalize_template_actions(self, actions):
        """Normalize physical action templates while tolerating flat frameskip actions."""
        mean = self.preprocessor.action_mean
        std = self.preprocessor.action_std
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=actions.dtype)
        if not torch.is_tensor(std):
            std = torch.tensor(std, dtype=actions.dtype)
        mean = mean.to(device=actions.device, dtype=actions.dtype)
        std = std.to(device=actions.device, dtype=actions.dtype)

        if mean.shape[-1] != actions.shape[-1]:
            if actions.shape[-1] % mean.shape[-1] == 0:
                repeat_n = actions.shape[-1] // mean.shape[-1]
                mean = mean.repeat(repeat_n)
                std = std.repeat(repeat_n)
            else:
                padded_mean = torch.zeros(actions.shape[-1], device=actions.device, dtype=actions.dtype)
                padded_std = torch.ones(actions.shape[-1], device=actions.device, dtype=actions.dtype)
                n = min(mean.shape[-1], actions.shape[-1])
                padded_mean[:n] = mean[:n]
                padded_std[:n] = std[:n]
                mean, std = padded_mean, padded_std

        return (actions - mean) / torch.clamp(std, min=1e-6)

    def _build_action_templates(self, device, dtype):
        """Build constant steering/throttle templates in normalized action space."""
        if not self.action_templates or self.num_samples <= 1 or self.action_dim < 2:
            return None

        max_templates = max(0, int(round(self.num_samples * self.template_frac)))
        if max_templates == 0:
            return None

        # Physical MetaDrive actions: [steer, throttle].
        base_pairs = torch.tensor(
            [
                [-1.0, 0.8],   # forward hard left
                [1.0, 0.8],    # forward hard right
                [-1.0, 0.35],  # slow forward hard left
                [1.0, 0.35],   # slow forward hard right
                [-1.0, -0.6],  # reverse hard left
                [1.0, -0.6],   # reverse hard right
                [0.0, 0.0],    # brake / stop
                [0.0, 0.8],    # straight forward
                [0.0, -0.6],   # straight reverse
            ],
            device=device,
            dtype=dtype,
        )
        n_templates = min(max_templates, base_pairs.shape[0])
        base_pairs = base_pairs[:n_templates]

        physical = torch.zeros(
            n_templates, self.horizon, self.action_dim, device=device, dtype=dtype
        )
        if self.action_dim % 2 == 0:
            physical[:] = base_pairs[:, None, :].repeat(1, self.horizon, self.action_dim // 2)
        else:
            physical[:, :, :2] = base_pairs[:, None, :]

        return self._normalize_template_actions(physical)

    def _denormalize_macro_actions(self, actions):
        """Convert normalized macro actions to physical per-step [steer, throttle]."""
        if self.action_dim % 2 == 0:
            per_step = rearrange(actions, "b t (f d) -> b (t f) d", d=2)
        else:
            per_step = actions[..., :2]
        mean = self.preprocessor.action_mean
        std = self.preprocessor.action_std
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=actions.dtype)
        if not torch.is_tensor(std):
            std = torch.tensor(std, dtype=actions.dtype)
        mean = mean.to(device=actions.device, dtype=actions.dtype)
        std = std.to(device=actions.device, dtype=actions.dtype)
        return per_step * std + mean

    def _action_regularization_loss(self, actions):
        """Penalize implausible bang-bang controls without replacing the task loss."""
        if (
            self.action_smooth_weight <= 0
            and self.action_intra_macro_weight <= 0
            and self.action_bound_weight <= 0
            and self.action_mag_weight <= 0
        ):
            return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

        physical = self._denormalize_macro_actions(actions)
        reg = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

        if self.action_bound_weight > 0:
            over = torch.relu(torch.abs(physical[..., :2]) - 1.0)
            reg = reg + self.action_bound_weight * over.square().mean(dim=(1, 2))

        if self.action_smooth_weight > 0 and physical.shape[1] > 1:
            delta = physical[:, 1:, :2] - physical[:, :-1, :2]
            reg = reg + self.action_smooth_weight * delta.square().mean(dim=(1, 2))

        if self.action_intra_macro_weight > 0 and self.action_dim % 2 == 0:
            f = self.action_dim // 2
            macro = rearrange(physical, "b (t f) d -> b t f d", t=actions.shape[1], f=f)
            intra_var = macro[:, :, :, :2].var(dim=2, unbiased=False)
            reg = reg + self.action_intra_macro_weight * intra_var.mean(dim=(1, 2))

        if self.action_mag_weight > 0:
            reg = reg + self.action_mag_weight * physical[..., :2].square().mean(dim=(1, 2))

        return reg

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g = {key: value.detach() for key, value in z_obs_g.items()}

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        best_actions = mu.detach().clone()
        best_losses = torch.full((n_evals,), float("inf"), device=self.device)

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                templates = self._build_action_templates(action.device, action.dtype)
                if templates is not None:
                    n_templates = min(templates.shape[0], self.num_samples - 1)
                    template_action = templates[:n_templates]
                    if self.template_noise > 0:
                        template_action = template_action + torch.randn_like(template_action) * self.template_noise
                    action[1 : 1 + n_templates] = template_action

                eval_batch_size = self.eval_batch_size or self.num_samples
                loss_chunks = []
                with torch.no_grad():
                    for start in range(0, self.num_samples, eval_batch_size):
                        end = min(start + eval_batch_size, self.num_samples)
                        chunk_obs_0 = {
                            key: value[start:end]
                            for key, value in cur_trans_obs_0.items()
                        }
                        chunk_z_obs_g = {
                            key: value[start:end]
                            for key, value in cur_z_obs_g.items()
                        }
                        i_z_obses, i_zs = self.wm.rollout(
                            obs_0=chunk_obs_0,
                            act=action[start:end],
                        )
                        loss_chunks.append(self.objective_fn(i_z_obses, chunk_z_obs_g).detach())
                        del chunk_obs_0, chunk_z_obs_g, i_z_obses, i_zs
                loss = torch.cat(loss_chunks, dim=0)
                loss = loss + self._action_regularization_loss(action).detach()
                best_idx = torch.argmin(loss)
                best_loss = loss[best_idx].detach()
                if best_loss < best_losses[traj]:
                    best_losses[traj] = best_loss
                    best_actions[traj] = action[best_idx].detach()
                mu_new, sigma_new = self._update_elite(
                    action, loss, sigma[traj], opt_step=i
                )
                mu[traj] = mu_new
                sigma[traj] = sigma_new
                losses.append(loss[torch.argsort(loss)[0]].item())
                del cur_trans_obs_0, cur_z_obs_g, action, loss, loss_chunks

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        planned = best_actions if self.return_best else mu
        return planned, np.full(n_evals, np.inf)
