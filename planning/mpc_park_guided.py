import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlannerGuided(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc_phase0",
        log_filename="logs.json",
        save_video=True,
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
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.save_video = bool(save_video)
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []
        self.final_obs_0 = None   # last real obs after plan() completes
        self.final_state_0 = None  # last real state after plan() completes
        self.base_history = None  # prepended to history_actions for phase continuity
        self.prior_actions = None  # normalized expert/demo prior for the current phase
        self.save_rollout_images = bool(kwargs.get("save_rollout_images", True))

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        e_final_state = init_state_0  # default if loop never runs
        memo_actions = None

        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"{self.logging_prefix}_plan_{self.iter}"

            # 1. 告诉 Evaluator 当前已经走过的历史动作是什么
            current_planned = torch.cat(self.planned_actions, dim=1) if len(self.planned_actions) > 0 else None
            if self.base_history is not None and current_planned is not None:
                self.evaluator.history_actions = torch.cat([self.base_history, current_planned], dim=1)
            elif self.base_history is not None:
                self.evaluator.history_actions = self.base_history
            else:
                self.evaluator.history_actions = current_planned

            # 🔥 当 history_actions 非空时，必须回到 init_state_0 进行完整 rollout，
            # 否则 GD planner 内部 eval 会从 e_final_state 出发再跑一遍 history，造成"双重计数"！
            # 🟢 obs_0 必须用 cur_obs_0（当前子规划器的 WM 起点），而不是 init_obs_0，
            #    这样 evaluator 内部的 WM rollout 和子规划器的梯度计算使用同一个起点。
            if self.evaluator.history_actions is not None:
                self.evaluator.assign_init_cond(obs_0=cur_obs_0, state_0=init_state_0)
                
            subplanner_actions = self.prior_actions if self.iter == 0 else memo_actions
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=subplanner_actions,
            )
            actions = torch.clamp(actions, -2.5, 2.5)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)

            # For final eval, use base_history (phase 1 actions) if present, but not current planned
            # (action_so_far already contains all current phase actions)
            self.evaluator.history_actions = self.base_history

            # 🟢 WM obs_0 must be cur_obs_0 when base_history is set (Phase 2).
            #     If we use init_obs_0 instead, the WM would start from the original scene
            #     but only receive Phase-2 continuation actions — producing garbage images.
            #     Env rollout uses state_0 + history_actions + action_so_far, then slices
            #     history off, so env eval is correct regardless.
            self.evaluator.assign_init_cond(
                obs_0=cur_obs_0 if self.base_history is not None else init_obs_0,
                state_0=init_state_0,
            )
            rollout_filename = f"{self.logging_prefix}_plan{self.iter}"
            print(
                f"[MPC DEBUG] Starting Phase-2 (or continued) eval: "
                f"filename={rollout_filename}, save_video={self.save_video}, "
                f"save_image={self.save_rollout_images}"
            )
            original_decoder = self.evaluator.wm.decoder
            if not self.save_rollout_images:
                self.evaluator.wm.decoder = None
            try:
                logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                    action_so_far,
                    self.action_len,
                    filename=rollout_filename,
                    save_video=self.save_video,
                )
            finally:
                self.evaluator.wm.decoder = original_decoder
            if self.save_rollout_images:
                print(f"[MPC DEBUG] Rollout image saved: {rollout_filename}.png")
            print(f"[MPC DEBUG] Eval finished for {rollout_filename}")
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            # Use action_len to grab the obs/state at the moment each traj
            # reached success, so finished episodes do not drift past the goal.
            # Use _get_trajdict_last_n(n=3) so cur_obs_0 has 3 frames of history
            # for the WM predictor to produce accurate rollouts in the next MPC iteration.
            e_final_obs = self.evaluator._get_trajdict_last_n(
                e_obses, self.action_len * self.evaluator.frameskip + 1, n=3
            )
            e_final_state = self.evaluator._get_traj_last(
                e_states, self.action_len * self.evaluator.frameskip + 1
            )[:, 0]
            cur_obs_0 = e_final_obs

            # 🔥🔥🔥 核心修复：为下一轮 WM encode 设置真实的 init_actions
            #
            # cur_obs_0 从切片后的 env rollout 中提取，3 帧位于时间步
            # [step*(n-2), step*(n-1), step*n] 处（n = action_so_far 的宏动作数）。
            # 训练配对：encode(frame[i], act[i]) 中 act[i] = 从 frame[i] 出发的动作。
            #
            #   - frame[step*(n-2)] act = actions[-2]  (从该帧出发的宏动作)
            #   - frame[step*(n-1)] act = actions[-1]  (从该帧出发的宏动作)
            #   - frame[step*n]     act = clipped_act[0] (即将规划的第 1 个动作)
            #
            # ClampedWM.rollout() 内部会拼接 init_actions + clipped_act[:, 0:1, :]，
            # 所以我们只需提供最后 2 个历史动作。
            if hasattr(self.wm, "init_actions") and action_so_far.shape[1] >= 2:
                self.wm.init_actions = action_so_far[:, -2:, :].detach().cpu()

            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.sub_planner.logging_prefix = f"{self.logging_prefix}_plan_{self.iter}"

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.final_obs_0 = cur_obs_0
        self.final_state_0 = e_final_state
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len

    def reset(self):
        self.is_success = None
        self.action_len = None
        self.iter = 0
        self.planned_actions = []
        self.final_obs_0 = None
        self.final_state_0 = None
        self.base_history = None
        self.prior_actions = None
