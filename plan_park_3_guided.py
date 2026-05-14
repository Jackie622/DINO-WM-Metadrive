import os
import gym
import cv2
import imageio
import math
import json
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator_park import PlanEvaluator
from utils import cfg_to_dict, seed, move_to_device

from metadrive.envs.marl_envs import MultiAgentParkingLotEnv
from metadrive.component.vehicle.vehicle_type import SVehicle

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

# ========================================================
# 1. 绘图与环境配置参数 (来自 gen_dataset_parking.py)
# ========================================================
RENDER_RES = (672, 672)
REAL_RES = (224, 224)
BEV_SCALING = 24.0
MAP_CENTER = (29.0, 1.0)

CONTROL_CONFIG = {
    "MAX_FORWARD_SPEED": 3.0, "MAX_REVERSE_SPEED": -2.0, "APPROACH_DIST": 0.8,
    "APPROACH_ANGLE_TOLERANCE": 0.15, "STOP_THRESHOLD": 0.2,
}

EXPERT_STAGE_TO_ID = {
    "APPROACH": 0,
    "STOPPING": 1,
    "REVERSE": 2,
    "FINISH": 3,
}
EXPERT_ID_TO_STAGE = {v: k for k, v in EXPERT_STAGE_TO_ID.items()}

PARKING_SLOTS = [
    {"id": 0, "pos": [34.36, 11.00], "heading": -1.57}, {"id": 1, "pos": [30.88, 11.00], "heading": -1.57},
    {"id": 2, "pos": [23.86, -8.50], "heading": 1.57}, {"id": 3, "pos": [31.00, -8.50], "heading": 1.57},
    {"id": 4, "pos": [24.17, 11.00], "heading": -1.57}, {"id": 5, "pos": [27.72, 11.00], "heading": -1.57},
    {"id": 6, "pos": [34.73, -8.50], "heading": 1.57}, {"id": 7, "pos": [27.40, -8.50], "heading": 1.57},
]

def to_px(pos):
    x = (pos[0] - MAP_CENTER[0]) * BEV_SCALING + RENDER_RES[0] / 2
    y = (pos[1] - MAP_CENTER[1]) * BEV_SCALING + RENDER_RES[1] / 2
    return (int(x), int(RENDER_RES[1] - y))

def get_transformed_pts(center, heading, local_pts):
    c, s = math.cos(heading), math.sin(heading)
    world_pts = []
    for x, y in local_pts:
        wx = center[0] + x * c - y * s
        wy = center[1] + x * s + y * c
        world_pts.append(to_px((wx, wy)))
    return np.array(world_pts, dtype=np.int32)

def draw_vehicle_on_img(img, pos, heading, is_ego=False):
    L, W = 4.6, 2.0
    if not is_ego:
        local_box = [(L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
        pts = get_transformed_pts(pos, heading, local_box)
        cv2.fillPoly(img, [pts], (150, 150, 150), lineType=cv2.LINE_AA)
        return

    # 1. 车尾段 (后 1/3) -> 霓虹紫/品红 (BGR: 200, 50, 180)
    tail_local = [(-L / 6, W / 2), (-L / 6, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
    tail_pts = get_transformed_pts(pos, heading, tail_local)
    cv2.fillPoly(img, [tail_pts], (0, 128, 255), lineType=cv2.LINE_AA)

    # 2. 车中段 (中 1/3) -> 深石板灰 (BGR: 60, 60, 60)
    mid_local = [(L / 6, W / 2), (L / 6, -W / 2), (-L / 6, -W / 2), (-L / 6, W / 2)]
    mid_pts = get_transformed_pts(pos, heading, mid_local)
    cv2.fillPoly(img, [mid_pts], (100, 100, 100), lineType=cv2.LINE_AA)

    # 3. 车头段 (前 1/3) -> 荧光青/亮葱绿 (BGR: 150, 250, 50)
    head_local = [(L / 2, W / 2), (L / 2, -W / 2), (L / 6, -W / 2), (L / 6, W / 2)]
    head_pts = get_transformed_pts(pos, heading, head_local)
    cv2.fillPoly(img, [head_pts], (255, 200, 100), lineType=cv2.LINE_AA)

    # 4. 全车外轮廓硬描边 (亮白色，让车在任何背景下都能被抠出来)
    body_local = [(L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
    body_pts = get_transformed_pts(pos, heading, body_local)
    cv2.polylines(img, [body_pts], True, (255, 255, 255), 2, cv2.LINE_AA)

# ========================================================
# 2. 泊车专家策略
# ========================================================
class ParkingPilot:
    def __init__(self):
        self.stage = "APPROACH"
        self.target_slot = None
        self.setup_pose = None
        self.stop_timer = 0

    def set_target(self, slot, vehicle):
        self.target_slot = slot
        tx, ty = slot["pos"]
        th = slot["heading"]

        curr_h = vehicle.heading_theta
        out_vec = np.array([math.cos(th), math.sin(th)])
        approach_vec = np.array([math.cos(curr_h), math.sin(curr_h)])

        cross_prod = approach_vec[0] * out_vec[1] - approach_vec[1] * out_vec[0]

        angle_offset = math.pi / 4.0  # 45度
        D_out = 10.0  # 💡 垂直驶出距离调大，确保准备点在马路中间，不要贴车位太近
        D_side = 3.5  # 💡 向前开过车位的距离

        if cross_prod < 0:
            # 车位在【左侧】。理论上应向【右】甩头。
            # ⚠️ 如果你画出点后发现红线还是反的，把下面的减号改成加号！
            setup_h = self._normalize_angle(curr_h - angle_offset)
        else:
            # 车位在【右侧】。理论上应向【左】甩头。
            # ⚠️ 如果反了，把下面的加号改成减号！
            setup_h = self._normalize_angle(curr_h + angle_offset)

        setup_xy = np.array([tx, ty]) + out_vec * D_out + approach_vec * D_side
        self.setup_pose = np.array([setup_xy[0], setup_xy[1], setup_h])

        # print(
        #     f"🎯 目标车位 ID: {slot['id']} | 准备点: ({setup_xy[0]:.2f}, {setup_xy[1]:.2f})，期望朝向: {math.degrees(setup_h):.1f}°")
        self.stage = "APPROACH"
        self.stop_timer = 0

    def get_action(self, vehicle):
        heading_vec = np.array([math.cos(vehicle.heading_theta), math.sin(vehicle.heading_theta)])
        velocity_vec = np.array(vehicle.velocity)[:2]
        real_speed = np.dot(velocity_vec, heading_vec)
        ego_pos = vehicle.position
        ego_heading = vehicle.heading_theta

        if self.stage == "APPROACH":
            dist = np.linalg.norm(ego_pos - self.setup_pose[:2])
            if dist < CONTROL_CONFIG["APPROACH_DIST"]:
                self.stage = "STOPPING"
                return self._force_stop(real_speed)
            return self._velocity_control(ego_pos, ego_heading, real_speed, self.setup_pose[:2], self.setup_pose[2], reverse=False)

        if self.stage == "STOPPING":
            self.stop_timer += 1
            if abs(real_speed) < CONTROL_CONFIG["STOP_THRESHOLD"] or self.stop_timer > 20:
                self.stage = "REVERSE"
                return [0.0, 0.0]
            return self._force_stop(real_speed)

        if self.stage == "REVERSE":
            target_pos = np.array(self.target_slot["pos"])
            target_heading = self.target_slot["heading"]
            dist = np.linalg.norm(ego_pos - target_pos)
            head_err = abs(self._normalize_angle(ego_heading - target_heading))
            if dist < 0.15 and head_err < 0.10:
                self.stage = "FINISH"
                return [0.0, 0.0]
            return self._velocity_control(ego_pos, ego_heading, real_speed, target_pos, target_heading, reverse=True)
        return self._force_stop(real_speed)

    def _force_stop(self, real_speed):
        if real_speed > 0.05: return [0.0, -1.0]
        if real_speed < -0.05: return [0.0, 1.0]
        return [0.0, 0.0]

    def _velocity_control(self, curr_pos, curr_heading, real_speed, target_pos, target_heading, reverse=False):
        vec = target_pos - curr_pos
        dist = np.linalg.norm(vec)

        if reverse:
            # 倒车速度慢，保留原有的简单插值追踪即可
            aim_angle = self._normalize_angle(math.atan2(vec[1], vec[0]) + math.pi)
            alpha = np.clip((dist - 0.4) / 3.0, 0.0, 1.0)

            angle_diff = self._normalize_angle(aim_angle - target_heading)
            target_h = self._normalize_angle(target_heading + alpha * angle_diff)

            angle_err = self._normalize_angle(target_h - curr_heading)
            steer = np.clip(angle_err * 3.5, -1.0, 1.0)
            steer = -steer
            max_v = CONTROL_CONFIG["MAX_REVERSE_SPEED"]
            target_v = max_v if dist > 1.2 else max_v * 0.5
        else:
            # 🔥 前进：彻底抛弃插值，采用“动态牵引点 (Carrot-Chasing)”

            # pull_dist: 虚拟牵引点向后拉伸的距离。
            # 距离越远，拉得越长；当车距小于 1.5 米时，牵引点退回原位 (target_pos)
            pull_dist = np.clip(dist - 4.0, 0.0, 3.0)

            # 沿着目标朝向的反方向（-cos, -sin）计算虚拟牵引点坐标
            carrot_x = target_pos[0] - math.cos(target_heading) * pull_dist
            carrot_y = target_pos[1] - math.sin(target_heading) * pull_dist

            # 让车辆纯粹地追踪这个虚拟点！
            carrot_vec = np.array([carrot_x, carrot_y]) - curr_pos
            aim_angle = math.atan2(carrot_vec[1], carrot_vec[0])

            angle_err = self._normalize_angle(aim_angle - curr_heading)

            # 因为是纯位置追踪，不会再产生目标冲突和画龙
            steer = np.clip(angle_err * 2.5, -1.0, 1.0)
            target_v = CONTROL_CONFIG["MAX_FORWARD_SPEED"]

        v_err = target_v - real_speed
        throttle = np.clip(v_err * 2.0, -1.0, 1.0)
        return [steer, throttle]

    def _normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

# ========================================================
# 3. 环境 Wrapper适配 (多智能体车位 + 专家策略起步)
# ========================================================
class ParkingDinoWrapper(gym.Wrapper):
    def __init__(self, env, frameskip=1, action_mean=None, action_std=None): # <--- 新增
        super().__init__(env)
        self.proprio_dim = 3
        self.current_seed = None
        self.steps_to_goal = 25
        self.start_offset_range = [0, 3]
        self.frameskip = frameskip
        self.static_vehicles = []
        self.ego = None
        self.agent_id = None
        self.current_action = [0.0, 0.0] 
        
        # ⚠️ 保存动作的均值和方差，用于反归一化
        self.action_mean = action_mean if action_mean is not None else np.zeros(frameskip * 2)
        self.action_std = action_std if action_std is not None else np.ones(frameskip * 2)

    def _resolve_scene_seed(self, seed):
        """Map eval seeds to the MetaDrive scene seed space exactly once."""
        start_seed_limit = self.env.config.get("start_seed", 0)
        if seed is None:
            seed = self.current_seed if self.current_seed is not None else start_seed_limit
        seed = int(seed)
        if seed < start_seed_limit:
            seed += start_seed_limit
        return seed

    def update_env(self, env_info):
        if env_info:
            if 'seed' in env_info:
                self.current_seed = int(env_info['seed'])
            if 'steps_to_goal' in env_info:
                self.steps_to_goal = int(env_info['steps_to_goal'])
        return "OK"

    def _setup_scene(self, seed):
        """基于 seed 的确定性场景初始化"""
        seed = self._resolve_scene_seed(seed)
        rng = np.random.RandomState(seed)
        
        # ⚠️ 关键修复：必须在 env.reset() 之前清理上一局手动生成的车辆！
        if self.static_vehicles and hasattr(self.env, "engine") and self.env.engine is not None:
            for sv in self.static_vehicles:
                try:
                    self.env.engine.clear_objects([sv.id])
                except:
                    pass
        self.static_vehicles = []

        # 清理完之后，再安全地重置环境
        try:
            self.env.reset()
        except TypeError:
            self.env.reset()

        active_agents = self.env.agent_manager.active_agents
        if not active_agents:
            pass
            
        self.agent_id = list(active_agents.keys())[0]
        self.ego = active_agents[self.agent_id]

        # ==========================================
        # 🔥 1. 相对随机初始化主车 (与训练集严格对齐)
        # ==========================================
        # 使用 rng.choice 获取索引，防止直接抽取字典报错
        target_idx = rng.choice(len(PARKING_SLOTS))
        target_slot = PARKING_SLOTS[target_idx]
        target_id = target_slot["id"]
        target_x = target_slot["pos"][0]

        spawn_mode = ["right_side", "left_side"][rng.choice(2)]
        
        if spawn_mode == "left_side":
            start_x = target_x - rng.uniform(8.0, 14.0)
            start_y = rng.uniform(0.0, 2.0)
            start_h = 0.0 + rng.uniform(-0.25, 0.25)
        else:
            start_x = target_x + rng.uniform(8.0, 14.0)
            start_y = rng.uniform(0.0, 2.0)
            start_h = np.pi + rng.uniform(-0.25, 0.25)

        self.ego.set_position([start_x, start_y])
        self.ego.set_heading_theta(start_h)
        self.ego.set_velocity([0, 0])

        pilot = ParkingPilot()
        pilot.set_target(target_slot, self.ego)

        # ==========================================
        # 🔥 2. 随机背景车生成 (与训练集严格对齐)
        # ==========================================
        available_slots = [s for s in PARKING_SLOTS if s["id"] != target_id]
        
        # np.random.randint 的区间是 [low, high)，所以这就相当于抽 1 到 (len-1) 辆车
        num_static_cars = rng.randint(1, len(available_slots))
        
        # 为了避免 numpy 抽字典报错，我们抽取槽位的索引
        occupied_indices = rng.choice(len(available_slots), num_static_cars, replace=False)
        occupied_slots = [available_slots[i] for i in occupied_indices]

        for s in occupied_slots:
            noise_x = rng.uniform(-0.2, 0.2)
            noise_y = rng.uniform(-0.2, 0.2)
            noise_h = rng.uniform(-0.1, 0.1)
            spawn_pos = [s["pos"][0] + noise_x, s["pos"][1] + noise_y]
            spawn_h = s["heading"] + noise_h
            
            obs_car = self.env.engine.spawn_object(SVehicle, vehicle_config={
                "spawn_position_heading": (spawn_pos, spawn_h),
                "random_color": False
            })
            self.static_vehicles.append(obs_car)
            
        # 起步执行一定的偏移帧数，让车进入运动状态
        start_offset = rng.randint(self.start_offset_range[0], self.start_offset_range[1])
        return pilot, start_offset

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _interpolate_pose(self, pose_a, pose_b, t):
        """Linearly interpolate between two poses (x, y, heading).
        Args:
            pose_a, pose_b: (3,) arrays (x, y, heading)
            t: float in [0, 1], 0 → pose_a, 1 → pose_b
        Returns:
            (3,) interpolated pose
        """
        x = pose_a[0] + t * (pose_b[0] - pose_a[0])
        y = pose_a[1] + t * (pose_b[1] - pose_a[1])
        # SLERP for heading to handle angle wrap-around
        h_err = self._normalize_angle(pose_b[2] - pose_a[2])
        h = self._normalize_angle(pose_a[2] + t * h_err)
        return np.array([x, y, h], dtype=np.float32)

    def _pick_mid_pose_from_path(self, poses):
        poses = np.asarray(poses, dtype=np.float32)
        if len(poses) == 0:
            return None
        if len(poses) == 1:
            return poses[0]

        step_dist = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(step_dist)])
        if cum_dist[-1] < 1e-4:
            return poses[len(poses) // 2]
        mid_idx = int(np.searchsorted(cum_dist, 0.5 * cum_dist[-1]))
        return poses[np.clip(mid_idx, 0, len(poses) - 1)]

    def _collect_expert_approach_path(self, pilot, max_steps=120):
        poses = []
        for _ in range(max_steps):
            poses.append(np.array([
                self.ego.position[0], self.ego.position[1], self.ego.heading_theta
            ], dtype=np.float32))
            if pilot.stage != "APPROACH":
                break
            act = pilot.get_action(self.ego)
            self.env.step({self.agent_id: act})
            self.current_action = act
        return poses

    def _collect_expert_path(self, pilot, max_steps=420):
        """Collect the full ParkingPilot pose/stage/action trace from the current planner t0."""
        poses = []
        stages = []
        actions = []
        for _ in range(max_steps):
            poses.append(np.array([
                self.ego.position[0], self.ego.position[1], self.ego.heading_theta
            ], dtype=np.float32))
            stages.append(EXPERT_STAGE_TO_ID.get(getattr(pilot, "stage", ""), -1))
            if pilot.stage == "FINISH":
                break
            act = pilot.get_action(self.ego)
            actions.append(np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0))
            try:
                self.env.step({self.agent_id: act})
            except Exception:
                break
            self.current_action = act
        return poses, stages, actions

    def _make_expert_action_segment(self, expert_actions, start_frame, end_frame, n_steps):
        expert_actions = np.asarray(expert_actions, dtype=np.float32)
        start_frame = max(0, int(start_frame))
        end_frame = max(start_frame, int(end_frame))
        segment = expert_actions[start_frame:min(end_frame, len(expert_actions))]
        if len(segment) > int(n_steps):
            segment = segment[:int(n_steps)]
        if len(segment) < int(n_steps):
            pad = np.zeros((int(n_steps) - len(segment), 2), dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=0) if len(segment) else pad
        return segment.astype(np.float32)

    def save_expert_video(self, seed, filename, max_steps=500, fps=10, start_from_t0=False):
        """Save the closed-loop ParkingPilot trajectory in the same BEV view used by subgoal images."""
        seed = self._resolve_scene_seed(seed)
        pilot, start_offset = self._setup_scene(seed)

        if start_from_t0:
            num_hist = 3
            for _ in range(start_offset + (num_hist - 1) * self.frameskip):
                act = pilot.get_action(self.ego)
                self.env.step({self.agent_id: act})
                self.current_action = act

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        writer = imageio.get_writer(filename, format="FFMPEG", fps=int(fps), macro_block_size=None)
        final_state = None
        n_frames = 0

        try:
            for step_idx in range(int(max_steps)):
                obs = self._get_dino_obs()
                frame = np.asarray(obs["visual"], dtype=np.uint8)
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))

                stage = getattr(pilot, "stage", "UNKNOWN")
                cv2.putText(
                    frame,
                    f"seed={seed} step={step_idx} stage={stage}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                writer.append_data(frame)
                n_frames += 1

                if stage == "FINISH":
                    break

                act = np.clip(np.asarray(pilot.get_action(self.ego), dtype=np.float32), -1.0, 1.0)
                try:
                    self.env.step({self.agent_id: act})
                    self.current_action = act
                except Exception:
                    break
        finally:
            writer.close()
            final_state = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta], dtype=np.float32)

        print(f"[Expert Video] saved {filename} frames={n_frames} final_stage={pilot.stage}", flush=True)
        return {
            "filename": filename,
            "frames": n_frames,
            "final_stage": pilot.stage,
            "final_state": final_state,
        }

    def _generate_dense_subgoals_by_h(
        self,
        expert_path,
        setup_pose,
        goal_pose,
        phase_h,
        setup_policy="snap_to_expert",
        setup_snap_warn_dist=1.0,
        setup_neighbor_suppress_radius=1.0,
        expert_stages=None,
        expert_actions=None,
    ):
        """Sample short-horizon subgoals from expert path, keeping setup as a mode boundary."""
        poses = np.asarray(expert_path, dtype=np.float32)
        if len(poses) == 0:
            return [setup_pose], [self.get_obs_at_pose(setup_pose)]

        stride = max(1, int(phase_h) * int(self.frameskip))
        setup_pose = np.asarray(setup_pose, dtype=np.float32)
        goal_pose = np.asarray(goal_pose, dtype=np.float32)

        pos_dist = np.linalg.norm(poses[:, :2] - setup_pose[:2], axis=1)
        heading_dist = np.abs(np.arctan2(
            np.sin(poses[:, 2] - setup_pose[2]),
            np.cos(poses[:, 2] - setup_pose[2]),
        ))
        setup_score = pos_dist + 0.5 * heading_dist
        raw_setup_idx = int(np.argmin(setup_score))
        if setup_policy == "snap_to_expert_stride":
            stride_grid = np.arange(stride, len(poses), stride, dtype=np.int32)
            if len(stride_grid) == 0:
                setup_idx = raw_setup_idx
            else:
                setup_idx = int(stride_grid[int(np.argmin(setup_score[stride_grid]))])
        else:
            setup_idx = raw_setup_idx
        setup_anchor = poses[setup_idx] if setup_policy in ("snap_to_expert", "snap_to_expert_stride") else setup_pose
        setup_anchor_dist = float(np.linalg.norm(setup_anchor[:2] - setup_pose[:2]))
        if setup_anchor_dist > float(setup_snap_warn_dist):
            print(
                f"[Subgoals] warning: setup_pose is {setup_anchor_dist:.3f}m from expert path; "
                f"using setup_policy={setup_policy}.",
                flush=True,
            )

        selected = []
        for idx in range(stride, setup_idx, stride):
            if np.linalg.norm(poses[idx, :2] - setup_anchor[:2]) >= setup_neighbor_suppress_radius:
                selected.append((poses[idx], idx))

        selected.append((np.asarray(setup_anchor, dtype=np.float32), setup_idx))

        for idx in range(setup_idx + stride, len(poses), stride):
            if np.linalg.norm(poses[idx, :2] - goal_pose[:2]) < 0.5:
                break
            if np.linalg.norm(poses[idx, :2] - setup_anchor[:2]) >= setup_neighbor_suppress_radius:
                selected.append((poses[idx], idx))

        # Remove near-duplicates so very short approach/reverse pieces do not create zero-motion phases.
        deduped = []
        deduped_frame_idxs = []
        for pose, frame_idx in selected:
            pose = np.asarray(pose, dtype=np.float32)
            if not deduped or np.linalg.norm(pose[:2] - np.asarray(deduped[-1])[:2]) > 0.25:
                deduped.append(pose)
                deduped_frame_idxs.append(int(frame_idx))

        # Keep the "trust horizon" invariant even when setup-neighbor suppression
        # removes a visually close waypoint: every executed phase should cover at
        # most one stride of expert time.
        filled = []
        filled_frame_idxs = []
        prev_frame_idx = 0
        for pose, frame_idx in zip(deduped, deduped_frame_idxs):
            while int(frame_idx) - prev_frame_idx > stride:
                mid_idx = prev_frame_idx + stride
                filled.append(np.asarray(poses[mid_idx], dtype=np.float32))
                filled_frame_idxs.append(int(mid_idx))
                prev_frame_idx = int(mid_idx)
            filled.append(pose)
            filled_frame_idxs.append(int(frame_idx))
            prev_frame_idx = int(frame_idx)
        deduped = filled
        deduped_frame_idxs = filled_frame_idxs

        obs_list = []
        prev_frame_idx = 0
        for pose, frame_idx in zip(deduped, deduped_frame_idxs):
            obs = self.get_obs_at_pose(pose)
            obs["expert_frame_idx"] = np.array(frame_idx, dtype=np.int32)
            if expert_stages is not None and 0 <= int(frame_idx) < len(expert_stages):
                obs["expert_stage"] = np.array(int(expert_stages[int(frame_idx)]), dtype=np.int32)
            if expert_actions is not None:
                obs["expert_action_segment"] = self._make_expert_action_segment(
                    expert_actions, prev_frame_idx, frame_idx, stride
                )
            obs_list.append(obs)
            prev_frame_idx = int(frame_idx)
        print(
            f"[Subgoals] mode=expert_dense_by_h phase_H={phase_h} "
            f"stride={stride} physical_steps setup_idx={setup_idx} "
            f"setup_policy={setup_policy} setup_anchor_dist={setup_anchor_dist:.3f} "
            f"subgoals={len(deduped)}",
            flush=True,
        )
        return deduped, obs_list

    def generate_intermediate_subgoals(self, start_pose, setup_pose, goal_pose, num_phases, forward_mid_pose=None):
        """Generate intermediate subgoal observations for multi-phase planning.

        Subgoal layout for each num_phases:

        2 phases (start → setup_pose → goal):
          subgoal[0] = setup_pose

        3 phases (start → forward_mid → setup_pose → goal):
          subgoal[0] = forward midpoint from expert approach trajectory
          subgoal[1] = setup_pose

        4 phases (start → forward_mid → setup_pose → reverse_mid → goal):
          subgoal[0] = forward midpoint from expert approach trajectory
          subgoal[1] = setup_pose
          subgoal[2] = reverse midpoint (50% blend between setup_pose and goal_pose)
        """
        subgoal_obs_list = []
        subgoal_pose_list = []
        fwd_mid = forward_mid_pose
        if fwd_mid is None:
            fwd_mid = self._interpolate_pose(start_pose, setup_pose, 0.5)

        if num_phases == 2:
            # Phase 0: start → setup_pose, Phase 1: setup_pose → goal
            pose = setup_pose
            obs = self.get_obs_at_pose(pose)
            subgoal_obs_list.append(obs)
            subgoal_pose_list.append(pose)

        elif num_phases == 3:
            # Phase 0: start → forward_mid
            # Phase 1: forward_mid → setup_pose
            # Phase 2: setup_pose → goal
            mid_pose = fwd_mid
            obs_mid = self.get_obs_at_pose(mid_pose)
            subgoal_obs_list.append(obs_mid)
            subgoal_pose_list.append(mid_pose)

            obs_sub = self.get_obs_at_pose(setup_pose)
            subgoal_obs_list.append(obs_sub)
            subgoal_pose_list.append(setup_pose)

        elif num_phases == 4:
            # Phase 0: start → forward midpoint
            # Phase 1: forward midpoint → setup_pose
            # Phase 2: setup_pose → reverse midpoint
            # Phase 3: reverse midpoint → goal

            # Forward midpoint on the expert approach path.
            obs_fwd_mid = self.get_obs_at_pose(fwd_mid)
            subgoal_obs_list.append(obs_fwd_mid)
            subgoal_pose_list.append(fwd_mid)

            # setup_pose itself
            obs_sub = self.get_obs_at_pose(setup_pose)
            subgoal_obs_list.append(obs_sub)
            subgoal_pose_list.append(setup_pose)

            # Reverse midpoint between setup_pose and goal_pose
            # Keep heading pointed outward (same as setup_pose) since reversing
            rev_mid = self._interpolate_pose(setup_pose, goal_pose, 0.5)
            obs_rev_mid = self.get_obs_at_pose(rev_mid)
            subgoal_obs_list.append(obs_rev_mid)
            subgoal_pose_list.append(rev_mid)

        # Final goal is always parking slot (used as obs_g, not in subgoal list)
        obs_goal = self.get_obs_at_pose(goal_pose)
        return subgoal_obs_list, subgoal_pose_list, obs_goal

    def sample_random_init_goal_states(
        self,
        seed=None,
        num_phases=3,
        subgoal_mode="fixed",
        phase_h=None,
        subgoal_setup_policy="snap_to_expert",
        setup_snap_warn_dist=1.0,
        setup_neighbor_suppress_radius=1.0,
    ):
        seed = self._resolve_scene_seed(seed)

        pilot, start_offset = self._setup_scene(seed)
        num_hist = 3 # 强制使用 3 帧历史

        # 1. 先让车辆起步，进入真实的运动状态
        for _ in range(start_offset):
            act = pilot.get_action(self.ego)
            self.env.step({self.agent_id: act})

        # 2. 严格按 frameskip 收集 3 帧历史
        obs_hist_visual = []
        obs_hist_proprio = []
        obs_hist_poses = []
        for i in range(num_hist):
            obs = self._get_dino_obs()
            obs_hist_visual.append(obs['visual'])
            obs_hist_proprio.append(obs['proprio'])
            obs_hist_poses.append(obs['raw_pose'])

            # 跑到下一帧（如果已经是最后一帧就不需要跑了）
            if i < num_hist - 1:
                for _ in range(self.frameskip):
                    act = pilot.get_action(self.ego)
                    self.env.step({self.agent_id: act})

        obs_init = {
            "visual": np.stack(obs_hist_visual),   # 形状: (3, C, H, W)
            "proprio": np.stack(obs_hist_proprio),  # 形状: (3, 3)
            "raw_pose": np.stack(obs_hist_poses),   # 形状: (3, 3)
        }

        # Start pose is the last frame of the history
        start_pose = obs_init['raw_pose'][-1]  # (x, y, heading)

        # Setup pose (geometric offset in front of parking slot)
        setup_pose = pilot.setup_pose

        # Goal: use the actual slot pose
        target_slot = pilot.target_slot
        goal_pose = np.array([target_slot["pos"][0], target_slot["pos"][1], target_slot["heading"]], dtype=np.float32)

        if subgoal_mode == "expert_dense_by_h":
            if phase_h is None:
                raise ValueError("subgoal_mode=expert_dense_by_h requires phase_h.")
            expert_path, expert_stages, expert_actions = self._collect_expert_path(pilot)
            subgoal_pose_list, subgoal_obs_list = self._generate_dense_subgoals_by_h(
                expert_path,
                setup_pose,
                goal_pose,
                phase_h,
                setup_policy=subgoal_setup_policy,
                setup_snap_warn_dist=setup_snap_warn_dist,
                setup_neighbor_suppress_radius=setup_neighbor_suppress_radius,
                expert_stages=expert_stages,
                expert_actions=expert_actions,
            )
            obs_goal = self.get_obs_at_pose(goal_pose)
            if len(subgoal_obs_list) > 0 and len(expert_path) > 0:
                last_frame_idx = int(np.asarray(subgoal_obs_list[-1].get("expert_frame_idx", 0)).reshape(-1)[0])
                goal_frame_idx = max(last_frame_idx, len(expert_path) - 1)
                obs_goal["expert_frame_idx"] = np.array(goal_frame_idx, dtype=np.int32)
                obs_goal["expert_stage"] = np.array(EXPERT_STAGE_TO_ID["FINISH"], dtype=np.int32)
                obs_goal["expert_action_segment"] = self._make_expert_action_segment(
                    expert_actions,
                    last_frame_idx,
                    goal_frame_idx,
                    max(1, int(phase_h) * int(self.frameskip)),
                )
        else:
            # The first subgoal must be reachable by the expert/pilot dynamics.
            # A geometric midpoint can land off the actual approach arc, so sample
            # it from the expert trajectory starting at the same t0 as planning.
            expert_approach_path = self._collect_expert_approach_path(pilot)
            forward_mid_pose = self._pick_mid_pose_from_path(expert_approach_path)

            # Generate intermediate subgoal observations for multi-phase planning
            subgoal_obs_list, subgoal_pose_list, obs_goal = self.generate_intermediate_subgoals(
                start_pose, setup_pose, goal_pose, num_phases, forward_mid_pose=forward_mid_pose
            )

        # ==============================================================
        # 4. 🔥 最核心的物理对齐 🔥
        # Evaluator 在接下来评测时，会在当前环境直接进行物理推演 (env.rollout)。
        # 因此，物理环境必须精准停在刚才收集的第 3 帧（t0）的那一刻！
        # ==============================================================
        pilot, _ = self._setup_scene(seed)
        total_steps_to_t0 = start_offset + (num_hist - 1) * self.frameskip
        for _ in range(total_steps_to_t0):
            act = pilot.get_action(self.ego)
            self.env.step({self.agent_id: act})
            self.current_action = act # 保持当前动作同步

        return obs_init, subgoal_obs_list, obs_goal

    def rollout(self, seed, init_state, actions):
        seed = self._resolve_scene_seed(seed)

        pilot, start_offset = self._setup_scene(seed)
        
        # 🔥 核心修复：对齐起点！
        # World Model 的 obs_0 包含了 3 帧历史，所以真正的物理起点 t0 
        # 必须是 start_offset + 2 * frameskip 的位置！
        num_hist = 3
        total_steps_to_t0 = start_offset + (num_hist - 1) * self.frameskip
        
        for _ in range(total_steps_to_t0):
            act = pilot.get_action(self.ego)
            self.env.step({self.agent_id: act})
            self.current_action = act

        # 🔥 核心修复：将主车传送至 init_state 指定的位置/朝向
        # Phase 2 中 init_state = Phase 1 结束时的真实位置，实现两阶段物理衔接
        # 注意：仅当位置/朝向确实不同时才传送，否则保留 Pilot 已建立的运动速度
        if init_state is not None:
            current_xyh = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta])
            if np.linalg.norm(current_xyh - init_state) > 1e-4:
                self.ego.set_position(init_state[:2])
                self.ego.set_heading_theta(init_state[2])
                self.ego.set_velocity([0.0, 0.0])

        obs_visuals = []
        obs_proprios = []
        obs_poses = []

        # 此时的环境状态，才完美等同于世界模型看到的最后一帧画面！
        obs_start = self._get_dino_obs()
        obs_visuals.append(obs_start['visual'])
        obs_proprios.append(obs_start['proprio'])
        obs_poses.append(obs_start['raw_pose'])

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        for act in actions:
            obs, reward, done, info = self.step(act)
            obs_visuals.append(obs['visual'])
            obs_proprios.append(obs['proprio'])
            obs_poses.append(obs['raw_pose'])

        visual_stack = np.stack(obs_visuals)
        proprio_stack = np.stack(obs_proprios)
        pose_stack = np.stack(obs_poses)

        return {"visual": visual_stack, "proprio": proprio_stack}, pose_stack

    def rollout_controller_actions(self, seed, init_state, goal_state, horizon, reverse=False, history_actions=None):
        """Generate a closed-loop controller action sequence from this phase state.

        Returns physical macro-actions shaped (horizon, frameskip * 2), matching
        the planner action layout before normalization.
        """
        seed = self._resolve_scene_seed(seed)
        pilot, start_offset = self._setup_scene(seed)

        num_hist = 3
        total_steps_to_t0 = start_offset + (num_hist - 1) * self.frameskip
        for _ in range(total_steps_to_t0):
            act = pilot.get_action(self.ego)
            self.env.step({self.agent_id: act})
            self.current_action = act

        if history_actions is not None:
            history_actions = np.asarray(history_actions, dtype=np.float32)
            for act in history_actions:
                act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
                self.env.step({self.agent_id: act})
                self.current_action = act

        if init_state is not None:
            current_xyh = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta])
            if history_actions is None and np.linalg.norm(current_xyh - init_state) > 1e-4:
                self.ego.set_position(init_state[:2])
                self.ego.set_heading_theta(float(init_state[2]))
                self.ego.set_velocity([0.0, 0.0])

        ctrl = ParkingPilot()
        physical_steps = []
        goal_state = np.asarray(goal_state, dtype=np.float32)

        for _ in range(int(horizon) * self.frameskip):
            heading_vec = np.array([math.cos(self.ego.heading_theta), math.sin(self.ego.heading_theta)])
            velocity_vec = np.array(self.ego.velocity)[:2]
            real_speed = float(np.dot(velocity_vec, heading_vec))
            act = ctrl._velocity_control(
                curr_pos=np.asarray(self.ego.position, dtype=np.float32),
                curr_heading=float(self.ego.heading_theta),
                real_speed=real_speed,
                target_pos=goal_state[:2],
                target_heading=float(goal_state[2]),
                reverse=bool(reverse),
            )
            act = np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)
            physical_steps.append(act)
            self.env.step({self.agent_id: act})
            self.current_action = act

        physical_steps = np.stack(physical_steps, axis=0)
        macro_actions = physical_steps.reshape(int(horizon), self.frameskip * 2)
        final_state = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta], dtype=np.float32)
        return macro_actions.astype(np.float32), final_state

    def eval_state(self, state, goal):
        state = np.atleast_2d(state)  # handles both (3,) and (B, 3)
        goal = np.atleast_2d(goal)
        dist = np.linalg.norm(state[:, :2] - goal[:, :2], axis=-1)  # (B,)
        # Orientation error in [0, pi]
        heading_diff = np.abs(np.arctan2(np.sin(state[:, 2] - goal[:, 2]),
                                         np.cos(state[:, 2] - goal[:, 2])))
        is_success = (dist < 1.0) & (heading_diff < 0.50)
        return {"success": is_success.squeeze(), "distance": dist.squeeze(), "heading_error": heading_diff.squeeze()}

    def get_obs_at_pose(self, pose_xyh):
        pose_xyh = np.asarray(pose_xyh, dtype=np.float32)
        old_pos = np.array([self.ego.position[0], self.ego.position[1]], dtype=np.float32)
        old_heading = float(self.ego.heading_theta)
        old_speed = float(self.ego.speed)
        old_action = np.array(self.current_action, dtype=np.float32)

        self.ego.set_position(pose_xyh[:2].tolist())
        self.ego.set_heading_theta(float(pose_xyh[2]))
        self.ego.set_velocity([0.0, 0.0])  # 零速度
        self.current_action = [0.0, 0.0]   # 零动作

        # MetaDrive's topdown renderer can lag one render after teleporting a
        # vehicle. Discard one frame so the target visual matches pose_xyh.
        self._flush_topdown_render()
        obs = self._get_dino_obs()
        obs["raw_pose"] = pose_xyh.copy()

        self.ego.set_position(old_pos.tolist())
        self.ego.set_heading_theta(old_heading)
        self.ego.set_velocity([old_speed * np.cos(old_heading), old_speed * np.sin(old_heading)])
        self.current_action = old_action
        return obs

    def _flush_topdown_render(self):
        try:
            self.env.render(
                mode="topdown", window=False, screen_size=RENDER_RES,
                scaling=BEV_SCALING, camera_position=MAP_CENTER, draw_history=False
            )
        except Exception:
            pass

    def step(self, action):
        total_reward = 0.0
        info = {}
        
        # 1. 提取安全的 2D 动作 (Evaluator 传进来的是已经是 2D 的单帧物理动作)
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().detach().numpy().flatten()
        else:
            action_np = np.array(action).flatten()
            
        safe_act = action_np[:2] if len(action_np) >= 2 else np.array([0.0, 0.0])
        
        # 2. 裁剪到 MetaDrive 的物理极限 [-1, 1]
        clipped_act = np.clip(safe_act, -1.0, 1.0)
        self.current_action = clipped_act

        # 👇 ====== 新增这一行打印 ====== 👇
        # print(f"[Debug Env] 接收动作: {safe_act} -> 截断后执行: {clipped_act} | 当前车速: {self.ego.speed:.3f}")
        # 👆 ============================ 👆
        
        # 3. 仅执行 1 次底层物理步！(去掉了 frameskip 循环)
        try:
            next_obs, rewards, dones, infos = self.env.step({self.agent_id: clipped_act})
            total_reward += rewards.get(self.agent_id, 0.0)
            done = dones.get(self.agent_id, False) or dones.get("__all__", False)
            info = infos.get(self.agent_id, {})
        except Exception as e:
            done = True
            info["error"] = str(e)
            
        # 4. 获取下一帧观察
        dino_obs = self._get_dino_obs()
        return dino_obs, total_reward, done, info
    
    def _get_dino_obs(self):
        try:
            bev = self.env.render(
                mode="topdown", window=False, screen_size=RENDER_RES,
                scaling=BEV_SCALING, camera_position=MAP_CENTER, draw_history=False
            )
            if hasattr(bev, 'get'):
                img_raw = bev.get()
            elif hasattr(bev, 'cpu'):
                img_raw = bev.cpu().numpy()
            else:
                img_raw = bev

            # 1. 此时转成 BGR 是为了用 OpenCV 画图
            img = cv2.cvtColor(np.array(img_raw).astype(np.uint8), cv2.COLOR_RGB2BGR)

            for sv in self.static_vehicles:
                draw_vehicle_on_img(img, sv.position, sv.heading_theta, is_ego=False)
            if self.ego:
                draw_vehicle_on_img(img, self.ego.position, self.ego.heading_theta, is_ego=True)

            img = cv2.resize(img, REAL_RES, interpolation=cv2.INTER_AREA)

            # 🔥🔥🔥 新增这一行：画完图后，把 BGR 翻转回模型认识的 RGB 格式！ 🔥🔥🔥
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # 调整维度 (H, W, C) -> (C, H, W) (如果有些预处理需要的话，保持原有逻辑)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        
        try:
            # 恢复为 3 维，严格对齐模型训练时的维度
            proprio = np.array([self.ego.speed, self.current_action[0], 0.0], dtype=np.float32)
        except:
            proprio = np.zeros(3, dtype=np.float32)

        raw_pose = np.array([self.ego.position[0], self.ego.position[1], self.ego.heading_theta], dtype=np.float32)
        return {"visual": img, "proprio": proprio, "raw_pose": raw_pose}

# ========================================================
# 4. 专家策略调试：在 plan_park_3 环境中运行专家并保存视频
# ========================================================
def debug_expert_parking(cfg_dict, eval_seeds):
    """调试用函数，已禁用。"""
    pass

# ========================================================
# 5. 规划与核心基础组件 (沿用 plan_meta)
# ========================================================
def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(epoch, cfg_dicts, plan_output_dir):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(folder=subdir_path, slurm_max_num_timeout=20)
            executor.update_parameters(
                **{k: v for k, v in cfg_dict["hydra"]["launcher"].items() if k != "submitit_folder"})
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}")
        return jobs

def build_plan_cfg_dicts(plan_cfg_path="", ckpt_base_path="", model_name="", model_epoch="final", planner=["gd", "cem"],
                         goal_source=["dset"], goal_H=[1, 5, 10], alpha=[0, 0.1, 1]):
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {"planner": p, "goal_source": g_source, "goal_H": g_H, "ckpt_base_path": ckpt_base_path,
         "model_name": model_name, "model_epoch": model_epoch, "objective": {"alpha": a}}
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(os.path.join(config_path, f"planner/{planner}.yaml"))
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]
        cfg_dicts.append(cfg_dict)
    return cfg_dicts

class DummyWandbRun:
    def __init__(self): self.mode = "disabled"
    def log(self, *args, **kwargs): pass
    def watch(self, *args, **kwargs): pass
    def config(self, *args, **kwargs): pass
    def finish(self): pass

def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {k: v.to(device) for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload["epoch"]
    return result

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else torch.load(decoder_path)
        else:
            raise ValueError("Decoder path not found")
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
    return model

def diagnose_parking_dataset_actions(datasets, traj_dsets=None, max_files=None):
    """Print action/state coverage statistics for the parking dataset."""
    files = []
    for dset_group in (datasets, traj_dsets or {}):
        for split_name in ("train", "valid"):
            dset = dset_group.get(split_name) if isinstance(dset_group, dict) else None
            split_files = getattr(dset, "files", None)
            if split_files:
                files.extend(split_files)
    files = sorted(set(files))
    if max_files is not None and max_files > 0:
        files = files[: int(max_files)]

    if not files:
        print("[Dataset Diagnostics] skipped: no dataset files found.")
        return {}

    action_chunks = []
    state_chunks = []
    lengths = []
    for fpath in files:
        try:
            with np.load(fpath) as data:
                action = np.asarray(data["action"], dtype=np.float32)
                action_chunks.append(action)
                lengths.append(len(action))
                if "state" in data:
                    state_chunks.append(np.asarray(data["state"], dtype=np.float32))
        except Exception as e:
            print(f"[Dataset Diagnostics] warning: failed to read {fpath}: {e}")

    if not action_chunks:
        print("[Dataset Diagnostics] skipped: no readable action arrays.")
        return {}

    actions = np.concatenate(action_chunks, axis=0)
    steer = actions[:, 0]
    throttle = actions[:, 1]
    reverse_mask = throttle < -0.05
    forward_mask = throttle > 0.05
    abs_steer = np.abs(steer)

    def pct(mask):
        return 100.0 * float(np.mean(mask)) if len(mask) else 0.0

    def summarize(name, values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return f"{name}: empty"
        qs = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        return (
            f"{name}: mean={values.mean():.4f}, std={values.std():.4f}, "
            f"min={values.min():.4f}, p01={qs[0]:.4f}, p05={qs[1]:.4f}, "
            f"p25={qs[2]:.4f}, p50={qs[3]:.4f}, p75={qs[4]:.4f}, "
            f"p95={qs[5]:.4f}, p99={qs[6]:.4f}, max={values.max():.4f}"
        )

    big_turn = abs_steer > 0.5
    run_lengths = []
    current = 0
    for flag in big_turn:
        if flag:
            current += 1
        elif current > 0:
            run_lengths.append(current)
            current = 0
    if current > 0:
        run_lengths.append(current)

    print("\n[Dataset Diagnostics] Parking action distribution")
    print(f"[Dataset Diagnostics] files={len(files)}, steps={len(actions)}, traj_len mean/min/max={np.mean(lengths):.1f}/{np.min(lengths)}/{np.max(lengths)}")
    print(f"[Dataset Diagnostics] {summarize('steer', steer)}")
    print(f"[Dataset Diagnostics] {summarize('throttle', throttle)}")
    print(f"[Dataset Diagnostics] abs(steer)>0.3/0.5/0.8 = {pct(abs_steer > 0.3):.2f}% / {pct(abs_steer > 0.5):.2f}% / {pct(abs_steer > 0.8):.2f}%")
    print(f"[Dataset Diagnostics] throttle reverse/near_zero/forward = {pct(reverse_mask):.2f}% / {pct(np.abs(throttle) <= 0.05):.2f}% / {pct(forward_mask):.2f}%")
    print(f"[Dataset Diagnostics] reverse abs(steer)>0.3/0.5/0.8 = {pct(abs_steer[reverse_mask] > 0.3):.2f}% / {pct(abs_steer[reverse_mask] > 0.5):.2f}% / {pct(abs_steer[reverse_mask] > 0.8):.2f}%")
    print(f"[Dataset Diagnostics] forward abs(steer)>0.3/0.5/0.8 = {pct(abs_steer[forward_mask] > 0.3):.2f}% / {pct(abs_steer[forward_mask] > 0.5):.2f}% / {pct(abs_steer[forward_mask] > 0.8):.2f}%")
    if run_lengths:
        run_lengths = np.asarray(run_lengths)
        print(
            "[Dataset Diagnostics] big-turn run length abs(steer)>0.5: "
            f"count={len(run_lengths)}, mean={run_lengths.mean():.2f}, "
            f"p50={np.percentile(run_lengths, 50):.1f}, p95={np.percentile(run_lengths, 95):.1f}, max={run_lengths.max()}"
        )
    else:
        print("[Dataset Diagnostics] big-turn run length abs(steer)>0.5: none")

    if state_chunks:
        states = np.concatenate(state_chunks, axis=0)
        print(f"[Dataset Diagnostics] {summarize('state/proprio dim0', states[:, 0])}")
        if states.shape[1] > 1:
            print(f"[Dataset Diagnostics] {summarize('state/proprio dim1', states[:, 1])}")

    return {
        "num_files": len(files),
        "num_steps": int(len(actions)),
        "abs_steer_gt_03": pct(abs_steer > 0.3),
        "abs_steer_gt_05": pct(abs_steer > 0.5),
        "abs_steer_gt_08": pct(abs_steer > 0.8),
        "reverse_pct": pct(reverse_mask),
    }

class PlanWorkspace:
    def __init__(
            self,
            cfg_dict: dict,
            wm: torch.nn.Module,
            dset,
            env: SubprocVectorEnv,
            env_name: str,
            frameskip: int,
            wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        self.eval_seed = [cfg_dict["seed"] + n for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.num_phases = cfg_dict.get("num_phases", 2)  # default 2-phase for backward compat
        self.subgoal_mode = cfg_dict.get("subgoal_mode", "fixed")
        self.phase_H = cfg_dict.get("phase_H", cfg_dict["planner"].get("sub_planner", {}).get("horizon", self.goal_H))
        self.subgoal_setup_policy = cfg_dict.get("subgoal_setup_policy", "snap_to_expert")
        self.setup_snap_warn_dist = cfg_dict.get("setup_snap_warn_dist", 1.0)
        self.setup_neighbor_suppress_radius = cfg_dict.get("setup_neighbor_suppress_radius", 1.0)
        self.expert_oracle_cfg = cfg_dict.get("expert_oracle", {})
        self.action_dim = self.dset.action_dim
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(cfg_dict["objective"])
        self.objective_fn = objective_fn

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.raw_action_mean, action_std=self.dset.raw_action_std,
            state_mean=self.dset.state_mean, state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean, proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )
        diagnostics_cfg = self.cfg_dict.get("diagnostics", {})
        self.evaluator.plot_full = bool(diagnostics_cfg.get("plot_full_rollout_images", False))
        self.evaluator.print_action_stats = bool(diagnostics_cfg.get("print_action_stats", False))

        if self.wandb_run is None or isinstance(self.wandb_run, wandb.sdk.lib.disabled.RunDisabled):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        from planning.mpc_park_guided import MPCPlannerGuided as MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = int(self.phase_H)
            self.planner.n_taken_actions = cfg_dict["planner"]["n_taken_actions"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.save_subgoal_visualization()
        self.save_expert_video_if_requested()
        self.dump_targets()

    def prepare_targets(self):
        if self.goal_source == "random_state":
            target_phys_steps = self.frameskip * self.goal_H
            print(f"🚀 Planning Setup: ParkingPilot running {target_phys_steps} steps to generate goal.", flush=True)
            print(
                f"[Subgoals] request mode={self.subgoal_mode} phase_H={self.phase_H} "
                f"legacy_num_phases={self.num_phases}",
                flush=True,
            )

            observations, states, actions, env_info = self.sample_traj_segment_from_dset(traj_len=2)

            steps_update_list = [{'steps_to_goal': target_phys_steps} for _ in range(self.n_evals)]
            self.env.update_env(steps_update_list)

            init_obs_tuple, subgoal_obs_tuple_list, goal_obs_tuple = self.env.sample_random_init_goal_states(
                self.eval_seed,
                num_phases=self.num_phases,
                subgoal_mode=self.subgoal_mode,
                phase_h=self.phase_H,
                subgoal_setup_policy=self.subgoal_setup_policy,
                setup_snap_warn_dist=self.setup_snap_warn_dist,
                setup_neighbor_suppress_radius=self.setup_neighbor_suppress_radius,
            )
            print(f"[Subgoals] received {len(subgoal_obs_tuple_list)} subgoals from env", flush=True)

            for i, obs in enumerate(init_obs_tuple):
                if "debug_info" in obs: del obs["debug_info"]

            def stack_dicts(dict_iterable):
                dict_list = list(dict_iterable)
                keys = dict_list[0].keys()
                return {k: np.stack([d[k] for d in dict_list]) for k in keys}

            obs_0 = stack_dicts(init_obs_tuple)
            obs_g = stack_dicts(goal_obs_tuple)

            # Build list of subgoal observations
            self.obs_g_sub_list = []
            self.state_g_sub_list = []
            for subgoal_tuple in subgoal_obs_tuple_list:
                obs_g_sub = stack_dicts(subgoal_tuple)
                state_g_sub = obs_g_sub['raw_pose']  # (B, 3)
                # goal obs: add time dimension -> (B, 1, ...)
                for k in obs_g_sub.keys():
                    obs_g_sub[k] = np.expand_dims(obs_g_sub[k], axis=1)
                self.obs_g_sub_list.append(obs_g_sub)
                self.state_g_sub_list.append(state_g_sub)

            # state_0: last history frame pose (x, y, heading)
            state_0 = obs_0['raw_pose'][:, -1, :]
            state_g = obs_g['raw_pose']            # (B, 3)

            # goal obs: add time dimension -> (B, 1, ...)
            for k in obs_g.keys():
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = state_0
            self.state_g = state_g
            self.gt_actions = None

        else:
            # Dataset 分支同样适配 3 帧历史逻辑
            num_hist = 3
            total_len = self.frameskip * (num_hist - 1 + self.goal_H) + 1
            observations, states, actions, env_info = self.sample_traj_segment_from_dset(traj_len=total_len)
            self.env.update_env(env_info)

            # 确定 t=0 (当前时刻) 在序列中的索引
            t0_idx = (num_hist - 1) * self.frameskip

            # 真车的初始物理状态是历史的最后一帧
            init_state = [x[t0_idx] for x in states]
            init_state = np.array(init_state)
            
            actions = torch.stack(actions)
            test_actions = actions[:, t0_idx : t0_idx + self.frameskip * self.goal_H]
            if self.goal_source == "random_action":
                test_actions = torch.randn_like(test_actions)

            wm_actions = test_actions[:, ::self.frameskip, :]
            exec_actions = self.data_preprocessor.denormalize_actions(test_actions)
            rollout_obses, rollout_states = self.env.rollout(self.eval_seed, init_state, exec_actions.numpy())

            # 提取完美的 3 帧历史喂给模型
            obs_0_dict = {}
            for k in rollout_obses.keys():
                batch_obs = []
                for b in range(self.n_evals):
                    hist_frames = []
                    for i in range(num_hist):
                        idx = i * self.frameskip
                        hist_frames.append(observations[b][k][idx])
                    batch_obs.append(np.stack(hist_frames))
                obs_0_dict[k] = np.stack(batch_obs)
                
            self.obs_0 = obs_0_dict
            self.obs_g = {key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()}
            self.state_0 = init_state
            self.state_g = rollout_states[:, -1]
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []
        
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")
            
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:
                traj_id = random.randint(0, len(self.dset) - 1)
                
                # 兼容返回 3 个或 4 个值的数据集
                sample_data = self.dset[traj_id]
                if len(sample_data) == 4:
                    obs, act, state, e_info = sample_data
                else:
                    obs, act, state = sample_data
                    e_info = {}
                    
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {key: arr[offset: offset + traj_len] for key, arr in obs.items()}
            state = state[offset: offset + traj_len]
            act = act[offset: offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def save_subgoal_visualization(self):
        """Save the exact start/subgoal/goal observations used by the planner."""
        try:
            def visual_to_bgr(visual):
                img = visual.cpu().numpy() if torch.is_tensor(visual) else np.asarray(visual)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                if np.issubdtype(img.dtype, np.floating):
                    if img.min() < 0:
                        img = (np.clip(img, -1.0, 1.0) + 1.0) * 127.5
                    elif img.max() <= 1.0:
                        img = np.clip(img, 0.0, 1.0) * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # These are the exact images consumed by planning:
            # start = last history frame; subgoals/final goal = evaluator targets.
            panels = [("start", self.obs_0["visual"][0, -1])]
            if hasattr(self, 'obs_g_sub_list') and self.obs_g_sub_list:
                for i, obs_g_sub in enumerate(self.obs_g_sub_list, start=1):
                    label = str(i)
                    if "expert_frame_idx" in obs_g_sub:
                        frame_idx = int(np.asarray(obs_g_sub["expert_frame_idx"][0, 0]).reshape(-1)[0])
                        label = f"{label} frame={frame_idx}"
                    panels.append((label, obs_g_sub["visual"][0, 0]))
            panels.append(("goal", self.obs_g["visual"][0, 0]))

            images = []
            for label, visual in panels:
                img = visual_to_bgr(visual)
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                img_resized = cv2.resize(img, REAL_RES, interpolation=cv2.INTER_AREA)
                images.append(img_resized)

            combined = cv2.hconcat(images)
            save_path = "subgoal_plan.png"
            cv2.imwrite(save_path, combined)
            print(f"Saved subgoal visualization: {os.path.abspath(save_path)} ({len(images)} poses)")
        except Exception as e:
            print(f"Warning: could not save subgoal visualization: {e}")
            import traceback
            traceback.print_exc()

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump({
                "obs_0": self.obs_0, "obs_g": self.obs_g, "state_0": self.state_0,
                "state_g": self.state_g, "gt_actions": self.gt_actions, "goal_H": self.goal_H,
            }, f)
        print(f"Dumped plan targets to {os.path.abspath('plan_targets.pkl')}")

    def save_expert_video_if_requested(self):
        diag_cfg = self.cfg_dict.get("diagnostics", {})
        if not diag_cfg.get("save_expert_video", False):
            return
        if not hasattr(self.env, "save_expert_video"):
            print("[Expert Video] skipped: env does not support save_expert_video.")
            return

        fps = int(diag_cfg.get("expert_video_fps", 10))
        max_steps = int(diag_cfg.get("expert_video_max_steps", 500))
        start_from_t0 = bool(diag_cfg.get("expert_video_start_from_t0", False))
        filenames = [
            os.path.abspath(f"expert_trajectory_seed{seed}.mp4")
            for seed in self.eval_seed
        ]
        try:
            self.env.save_expert_video(
                self.eval_seed,
                filenames,
                max_steps=max_steps,
                fps=fps,
                start_from_t0=start_from_t0,
            )
        except Exception as e:
            print(f"[Expert Video] warning: failed to save expert video: {e}")

    def _apply_phase_config(self, phase_idx):
        """Apply the shared MPC config for this phase without replacing sub_planner."""
        planner_cfg = self.cfg_dict["planner"]
        self.planner.logging_prefix = f"mpc_phase{phase_idx}"
        self.planner.save_video = bool(planner_cfg.get("save_video", self.planner.save_video))
        self.planner.save_rollout_images = bool(
            self.cfg_dict.get("diagnostics", {}).get("save_planner_rollout_images", True)
        )
        requested_taken = planner_cfg.get("n_taken_actions", self.planner.n_taken_actions)
        self.planner.n_taken_actions = min(int(requested_taken), int(self.planner.sub_planner.horizon))
        max_iter = planner_cfg.get("max_iter", self.planner.max_iter)
        self.planner.max_iter = np.inf if max_iter is None else max_iter

    def _setup_phase_transition(self, actions, phase_idx, num_phases, cur_obs):
        """Set up continuity between phases: base_history, init_actions, evaluator init_cond."""
        from planning.mpc_park_guided import MPCPlannerGuided as MPCPlanner
        if not isinstance(self.planner, MPCPlanner):
            return

        # base_history: cumulative actions from all previous phases
        if phase_idx == 0:
            self.planner.base_history = None
            self.evaluator.history_actions = None
        else:
            self.planner.base_history = actions.detach()
            self.evaluator.history_actions = None

        # evaluator always starts from original start for env rollout
        self.evaluator.assign_init_cond(obs_0=self.obs_0, state_0=self.state_0)

        # Set init_actions for ClampedWM to enable correct encoding
        num_hist = 3
        wm = self.planner.sub_planner.wm
        if hasattr(wm, "init_actions") and phase_idx > 0:
            # Use last 2 actions from completed phase(s) as init_actions
            if actions.shape[1] >= 2:
                init_a = actions[:, -2:, :].detach().cpu()
                wm.init_actions = init_a

    def _repeat_pair_as_macro_action(self, physical_actions, pair):
        """Fill one normalized macro-action sequence with a physical [steer, throttle] pair."""
        pair = torch.tensor(pair, dtype=physical_actions.dtype)
        if self.action_dim % 2 == 0:
            tiled = pair.repeat(self.action_dim // 2)
            physical_actions[:] = tiled
        else:
            physical_actions[..., :2] = pair

    def _normalize_physical_macro_actions(self, physical_actions):
        if self.action_dim % 2 == 0:
            per_step = rearrange(physical_actions.cpu(), "b t (f d) -> b (t f) d", d=2)
            normalized = self.data_preprocessor.normalize_actions(per_step)
            normalized = rearrange(
                normalized, "b (t f) d -> b t (f d)", t=physical_actions.shape[1], f=self.action_dim // 2
            )
            return normalized.to(self.device)
        return self.data_preprocessor.normalize_actions(physical_actions.cpu()).to(self.device)

    def _controller_pair_to_pose(self, cur_state, goal_state, reverse=False):
        cur_state = np.asarray(cur_state, dtype=np.float32)
        goal_state = np.asarray(goal_state, dtype=np.float32)
        vec = goal_state[:2] - cur_state[:2]
        aim = math.atan2(vec[1], vec[0])
        def norm_angle(angle):
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
            return angle
        if reverse:
            aim = norm_angle(aim + math.pi)
        heading_err = norm_angle(aim - cur_state[2])
        if reverse:
            steer = -np.clip(heading_err * 3.5, -1.0, 1.0)
            throttle = -0.6
        else:
            steer = np.clip(heading_err * 2.5, -1.0, 1.0)
            throttle = 0.8
        return [float(steer), float(throttle)]

    def _denormalize_accumulated_actions(self, accumulated_actions):
        if accumulated_actions is None:
            return None
        history_exec = rearrange(
            accumulated_actions.detach().cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        return self.data_preprocessor.denormalize_actions(history_exec).numpy()

    def _expert_oracle_enabled(self):
        return bool(self.expert_oracle_cfg.get("enabled", False))

    def _expert_guided_enabled(self):
        return bool(self.cfg_dict.get("expert_guided", {}).get("enabled", False))

    def _expert_oracle_horizon(self):
        frames = self.expert_oracle_cfg.get("frames", None)
        if frames is not None:
            frames = int(frames)
            if frames % int(self.frameskip) != 0:
                raise ValueError(
                    f"expert_oracle.frames={frames} must be divisible by frameskip={self.frameskip}."
                )
            return max(1, frames // int(self.frameskip))
        return int(self.expert_oracle_cfg.get("horizon", self.phase_H))

    def _infer_expert_oracle_reverse(self, phase_idx):
        mode = self.expert_oracle_cfg.get("reverse_mode", "stage")
        if mode == "always_forward":
            return False
        if mode == "always_reverse":
            return True

        goal_obs = self._current_phase_goal_obs(phase_idx)
        if "expert_stage" in goal_obs:
            stages = np.asarray(goal_obs["expert_stage"]).reshape(-1)
            valid = stages[stages >= 0]
            if len(valid) > 0:
                stage_id = int(valid[0])
                reverse = stage_id >= EXPERT_STAGE_TO_ID["REVERSE"]
                stage_name = EXPERT_ID_TO_STAGE.get(stage_id, f"id={stage_id}")
                print(
                    f"[Expert Oracle] phase={phase_idx} target_stage={stage_name} reverse={reverse}",
                    flush=True,
                )
                return reverse

        reverse = phase_idx >= max(1, len(self.obs_g_sub_list) // 2)
        print(
            f"[Expert Oracle] phase={phase_idx} no expert_stage found; "
            f"fallback reverse={reverse}",
            flush=True,
        )
        return reverse

    def _current_phase_goal_obs(self, phase_idx):
        if phase_idx < len(self.obs_g_sub_list):
            return self.obs_g_sub_list[phase_idx]
        return self.obs_g

    def _plan_phase_with_expert_oracle(self, phase_idx, cur_obs, cur_state, goal_state, accumulated_actions):
        horizon = self._expert_oracle_horizon()
        reverse = self._infer_expert_oracle_reverse(phase_idx)
        goal_obs = self._current_phase_goal_obs(phase_idx)

        dist_start = np.linalg.norm(np.asarray(cur_state)[:, :2] - np.asarray(goal_state)[:, :2], axis=1)
        source = self.expert_oracle_cfg.get("source", "trace")
        predicted_final_state = np.asarray(cur_state, dtype=np.float32)
        has_predicted_final_state = False
        if source == "trace" and "expert_action_segment" in goal_obs:
            segment = np.asarray(goal_obs["expert_action_segment"][:, 0], dtype=np.float32)
            n_steps = horizon * int(self.frameskip)
            if segment.shape[1] != n_steps:
                fixed = np.zeros((segment.shape[0], n_steps, 2), dtype=np.float32)
                take = min(segment.shape[1], n_steps)
                fixed[:, :take] = segment[:, :take]
                segment = fixed
            expert_physical = segment.reshape(segment.shape[0], horizon, self.action_dim)
            print(
                f"[Expert Oracle] phase={phase_idx} source=trace "
                f"target_frame={np.asarray(goal_obs.get('expert_frame_idx', [-1])).reshape(-1)[0]}",
                flush=True,
            )
        else:
            history_exec = self._denormalize_accumulated_actions(accumulated_actions)
            expert_physical, predicted_final_state = self.env.rollout_controller_actions(
                self.eval_seed,
                np.asarray(cur_state, dtype=np.float32),
                np.asarray(goal_state, dtype=np.float32),
                horizon,
                reverse,
                history_exec,
            )
            has_predicted_final_state = True
            print(f"[Expert Oracle] phase={phase_idx} source=controller", flush=True)
        phase_actions = self._normalize_physical_macro_actions(
            torch.tensor(expert_physical, dtype=torch.float32)
        )

        self.evaluator.history_actions = accumulated_actions.detach() if accumulated_actions is not None else None
        self.evaluator.assign_init_cond(obs_0=cur_obs, state_0=self.state_0)
        action_len = np.full(phase_actions.shape[0], phase_actions.shape[1], dtype=float)
        filename = f"expert_oracle_phase{phase_idx}"
        logs, successes, e_obses, e_states = self.evaluator.eval_actions(
            phase_actions.detach(),
            action_len,
            filename=filename,
            save_video=bool(self.expert_oracle_cfg.get("save_video", True)),
        )

        e_final_obs = self.evaluator._get_trajdict_last_n(
            e_obses, action_len * self.evaluator.frameskip + 1, n=3
        )
        e_final_state = self.evaluator._get_traj_last(
            e_states, action_len * self.evaluator.frameskip + 1
        )[:, 0]

        dist_end = np.linalg.norm(e_final_state[:, :2] - np.asarray(goal_state)[:, :2], axis=1)
        head_end = np.abs(np.arctan2(
            np.sin(e_final_state[:, 2] - np.asarray(goal_state)[:, 2]),
            np.cos(e_final_state[:, 2] - np.asarray(goal_state)[:, 2]),
        ))
        print(
            f"[Expert Oracle] phase={phase_idx} H={horizon} frames={horizon * self.frameskip} "
            f"dist_start={dist_start.mean():.3f} dist_end={dist_end.mean():.3f} "
            f"head_end={head_end.mean():.3f} success={np.mean(successes.astype(float)):.3f}",
            flush=True,
        )
        if has_predicted_final_state and np.max(np.abs(predicted_final_state - e_final_state)) > 1e-3:
            print(
                "[Expert Oracle] note: controller-generation final state differs from evaluator "
                f"by max_abs={np.max(np.abs(predicted_final_state - e_final_state)):.4f}",
                flush=True,
            )

        logs = {f"expert_oracle_phase{phase_idx}/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value)
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        self.evaluator.history_actions = None
        return phase_actions, e_final_obs, e_final_state

    def _phase_expert_prior_actions(self, phase_idx):
        goal_obs = self._current_phase_goal_obs(phase_idx)
        if "expert_action_segment" not in goal_obs:
            raise KeyError(
                f"phase {phase_idx} has no expert_action_segment; "
                "expert-guided MPPI requires trace-based expert subgoals."
            )
        horizon = int(self.planner.sub_planner.horizon)
        n_steps = horizon * int(self.frameskip)
        segment = np.asarray(goal_obs["expert_action_segment"][:, 0], dtype=np.float32)
        if segment.shape[1] != n_steps:
            fixed = np.zeros((segment.shape[0], n_steps, 2), dtype=np.float32)
            take = min(segment.shape[1], n_steps)
            fixed[:, :take] = segment[:, :take]
            segment = fixed
        expert_physical = segment.reshape(segment.shape[0], horizon, self.action_dim)
        prior_actions = self._normalize_physical_macro_actions(
            torch.tensor(expert_physical, dtype=torch.float32)
        )
        frame_idx = np.asarray(goal_obs.get("expert_frame_idx", [-1])).reshape(-1)[0]
        print(
            f"[Expert Guided] phase={phase_idx} prior=trace "
            f"target_frame={frame_idx} H={horizon} frames={n_steps}",
            flush=True,
        )
        return prior_actions

    def _build_action_probe_candidates(self, cur_state, goal_state, horizon, accumulated_actions=None):
        """Return hand-written candidate actions in normalized planner space."""
        base_pairs = [
            ("stop", [0.0, 0.0]),
            ("straight_fwd", [0.0, 0.8]),
            ("slow_fwd", [0.0, 0.35]),
            ("left_fwd", [-1.0, 0.8]),
            ("right_fwd", [1.0, 0.8]),
            ("left_rev", [-1.0, -0.6]),
            ("right_rev", [1.0, -0.6]),
            ("straight_rev", [0.0, -0.6]),
            ("pose_ctrl_fwd", self._controller_pair_to_pose(cur_state, goal_state, reverse=False)),
            ("pose_ctrl_rev", self._controller_pair_to_pose(cur_state, goal_state, reverse=True)),
        ]

        names = []
        pairs = []
        physical = torch.zeros(len(base_pairs), horizon, self.action_dim, dtype=torch.float32)
        for i, (name, pair) in enumerate(base_pairs):
            names.append(name)
            pairs.append(pair)
            self._repeat_pair_as_macro_action(physical[i], pair)

        expert_physical = []
        history_exec = self._denormalize_accumulated_actions(accumulated_actions)
        for name, reverse in [("expert_fwd", False), ("expert_rev", True)]:
            try:
                expert_actions, _ = self.env.rollout_controller_actions(
                    self.eval_seed[:1],
                    np.asarray(cur_state, dtype=np.float32)[None],
                    np.asarray(goal_state, dtype=np.float32)[None],
                    horizon,
                    reverse,
                    history_exec,
                )
                expert_action = torch.tensor(expert_actions[0], dtype=torch.float32)
                if expert_action.shape == (horizon, self.action_dim):
                    names.append(name)
                    pairs.append([
                        float(expert_action[:, 0::2].mean().item()),
                        float(expert_action[:, 1::2].mean().item()),
                    ])
                    expert_physical.append(expert_action.unsqueeze(0))
            except Exception as e:
                print(f"[Action Probe] warning: failed to build {name}: {e}")

        if expert_physical:
            physical = torch.cat([physical] + expert_physical, dim=0)
        actions = self._normalize_physical_macro_actions(physical)
        return names, pairs, actions

    def _compute_probe_wm_losses(self, cur_obs, obs_g, actions):
        trans_obs_0 = move_to_device(self.data_preprocessor.transform_obs(cur_obs), self.device)
        trans_obs_g = move_to_device(self.data_preprocessor.transform_obs(obs_g), self.device)
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)
            repeated_obs_0 = {
                key: value[:1].repeat(actions.shape[0], *([1] * (value.ndim - 1)))
                for key, value in trans_obs_0.items()
            }
            repeated_z_g = {
                key: value[:1].repeat(actions.shape[0], *([1] * (value.ndim - 1)))
                for key, value in z_obs_g.items()
            }
            i_z_obses, _ = self.wm.rollout(obs_0=repeated_obs_0, act=actions)
            losses = self.objective_fn(i_z_obses, repeated_z_g)
        return losses.detach().cpu().numpy()

    def _compute_probe_env_metrics(self, goal_obs, goal_state, actions, accumulated_actions=None):
        exec_actions = rearrange(
            actions.detach().cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.data_preprocessor.denormalize_actions(exec_actions).numpy()

        trans_obs_g = move_to_device(self.data_preprocessor.transform_obs(goal_obs), self.device)
        with torch.no_grad():
            z_obs_g = self.wm.encode_obs(trans_obs_g)
        target_z_obs_g = {key: value[:1] for key, value in z_obs_g.items()}

        history_len = 0
        if accumulated_actions is not None:
            history_exec = rearrange(
                accumulated_actions.detach().cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
            )
            history_exec = self.data_preprocessor.denormalize_actions(history_exec).numpy()
            history_len = history_exec.shape[1]
        else:
            history_exec = None

        rows = []
        for i in range(actions.shape[0]):
            candidate_exec = exec_actions[i : i + 1]
            if history_exec is not None:
                full_exec = np.concatenate([history_exec[:1], candidate_exec], axis=1)
            else:
                full_exec = candidate_exec
            e_obses, e_states = self.env.rollout(self.eval_seed[:1], self.state_0[:1], full_exec)
            final_state = e_states[:, -1]
            eval_results = self.env.eval_state(goal_state[:1], final_state)

            e_final_obs = {key: value[:, -1:] for key, value in e_obses.items()}
            trans_e_final_obs = move_to_device(
                self.data_preprocessor.transform_obs(e_final_obs), self.device
            )
            with torch.no_grad():
                e_z_final_obs = self.wm.encode_obs(trans_e_final_obs)
                env_z_loss = self.objective_fn(e_z_final_obs, target_z_obs_g)
            rows.append({
                "distance": float(np.asarray(eval_results["distance"]).reshape(-1)[0]),
                "heading_error": float(np.asarray(eval_results["heading_error"]).reshape(-1)[0]),
                "success": bool(np.asarray(eval_results["success"]).reshape(-1)[0]),
                "env_z_loss": float(env_z_loss.detach().cpu().reshape(-1)[0].item()),
                "history_len": history_len,
            })
        return rows

    def _eval_physical_macro_actions(self, goal_state, physical_macro_actions, accumulated_actions=None):
        exec_actions = physical_macro_actions.reshape(
            1, physical_macro_actions.shape[0] * self.frameskip, 2
        )
        history_exec = self._denormalize_accumulated_actions(accumulated_actions)
        if history_exec is not None:
            full_exec = np.concatenate([history_exec[:1], exec_actions], axis=1)
        else:
            full_exec = exec_actions

        _, e_states = self.env.rollout(self.eval_seed[:1], self.state_0[:1], full_exec)
        final_state = e_states[:, -1]
        eval_results = self.env.eval_state(goal_state[:1], final_state)
        return {
            "distance": float(np.asarray(eval_results["distance"]).reshape(-1)[0]),
            "heading_error": float(np.asarray(eval_results["heading_error"]).reshape(-1)[0]),
            "success": bool(np.asarray(eval_results["success"]).reshape(-1)[0]),
            "final_state": final_state[0],
        }

    def _print_controller_horizon_probe(self, phase_idx, cur_state, goal_state, accumulated_actions, base_horizon):
        horizons = [base_horizon, base_horizon * 2, base_horizon * 4]
        print(
            f"[Action Probe] phase_state: start=({cur_state[0,0]:.3f}, {cur_state[0,1]:.3f}, {cur_state[0,2]:.3f}) "
            f"goal=({goal_state[0,0]:.3f}, {goal_state[0,1]:.3f}, {goal_state[0,2]:.3f})"
        )
        for reverse in [False, True]:
            tag = "expert_rev" if reverse else "expert_fwd"
            for horizon in horizons:
                try:
                    history_exec = self._denormalize_accumulated_actions(accumulated_actions)
                    expert_actions, _ = self.env.rollout_controller_actions(
                        self.eval_seed[:1],
                        np.asarray(cur_state, dtype=np.float32),
                        np.asarray(goal_state, dtype=np.float32),
                        horizon,
                        reverse,
                        history_exec,
                    )
                    row = self._eval_physical_macro_actions(
                        goal_state,
                        expert_actions[0],
                        accumulated_actions=accumulated_actions,
                    )
                    fs = row["final_state"]
                    print(
                        f"[Action Probe] env_only {tag:>10s} H={horizon:<2d} "
                        f"dist={row['distance']:.3f} head={row['heading_error']:.3f} "
                        f"success={row['success']} final=({fs[0]:.3f}, {fs[1]:.3f}, {fs[2]:.3f})"
                    )
                except Exception as e:
                    print(f"[Action Probe] warning: failed env_only {tag} H={horizon}: {e}")

    def _run_action_probe(self, phase_idx, cur_obs, cur_state, goal_obs, goal_state, accumulated_actions):
        diag_cfg = self.cfg_dict.get("diagnostics", {})
        if not diag_cfg.get("action_probe", False):
            return
        if self.n_evals != 1:
            print("[Action Probe] skipped: currently only supports n_evals=1 for clear ranking.")
            return

        from planning.mpc_park_guided import MPCPlannerGuided as MPCPlanner
        horizon = self.planner.sub_planner.horizon if isinstance(self.planner, MPCPlanner) else self.goal_H
        names, pairs, actions = self._build_action_probe_candidates(
            cur_state[0], goal_state[0], horizon, accumulated_actions=accumulated_actions
        )
        wm_losses = self._compute_probe_wm_losses(cur_obs, goal_obs, actions)
        env_rows = self._compute_probe_env_metrics(goal_obs, goal_state, actions, accumulated_actions)
        self._print_controller_horizon_probe(phase_idx, cur_state, goal_state, accumulated_actions, horizon)

        wm_order = np.argsort(wm_losses)
        wm_rank = {int(idx): rank + 1 for rank, idx in enumerate(wm_order)}
        env_z_losses = np.array([row["env_z_loss"] for row in env_rows], dtype=np.float32)
        env_z_order = np.argsort(env_z_losses)
        env_z_rank = {int(idx): rank + 1 for rank, idx in enumerate(env_z_order)}
        env_order = sorted(range(len(names)), key=lambda idx: (env_rows[idx]["distance"], env_rows[idx]["heading_error"]))

        print(f"\n[Action Probe] Phase {phase_idx}: fixed candidates, horizon={horizon}")
        print("[Action Probe] rank_by_env | name | pair | env_dist | env_head | success | wm_loss | wm_rank | env_z_loss | env_z_rank")
        for env_rank, idx in enumerate(env_order, start=1):
            row = env_rows[idx]
            print(
                f"[Action Probe] {env_rank:02d} | {names[idx]:>13s} | "
                f"[{pairs[idx][0]: .3f}, {pairs[idx][1]: .3f}] | "
                f"{row['distance']:.3f} | {row['heading_error']:.3f} | "
                f"{row['success']} | {wm_losses[idx]:.6f} | {wm_rank[idx]:02d} | "
                f"{row['env_z_loss']:.6f} | {env_z_rank[idx]:02d}"
            )

        best_env = env_order[0]
        best_wm = int(wm_order[0])
        best_env_z = int(env_z_order[0])
        print(
            f"[Action Probe] best_env={names[best_env]} "
            f"(dist={env_rows[best_env]['distance']:.3f}, wm_rank={wm_rank[best_env]}) ; "
            f"best_wm={names[best_wm]} "
            f"(wm_loss={wm_losses[best_wm]:.6f}, env_dist={env_rows[best_wm]['distance']:.3f}) ; "
            f"best_env_z={names[best_env_z]} "
            f"(env_z_loss={env_rows[best_env_z]['env_z_loss']:.6f}, env_dist={env_rows[best_env_z]['distance']:.3f})"
        )

        if env_rows[best_env]["distance"] > 1.0:
            print("[Action Probe] conclusion: these hand candidates still do not contain a clearly good env action; sampling/template coverage is the first suspect.")
        elif best_env != best_wm and not env_rows[best_wm]["success"]:
            if best_env_z == best_env or env_z_rank[best_env] <= 2:
                print("[Action Probe] conclusion: real encoded final obs favors the successful candidate, but WM rollout favors a failed one; WM dynamics prediction is the first suspect.")
            else:
                print("[Action Probe] conclusion: both WM rollout/objective and real encoded final obs do not favor the successful candidate; latent objective/representation is the first suspect.")
        elif best_env != best_wm and wm_rank[best_env] > max(2, len(names) // 3):
            print("[Action Probe] conclusion: env has a good candidate, but WM/objective ranks it low; objective or WM prediction is the first suspect.")
        else:
            print("[Action Probe] conclusion: WM/objective can recognize a good hand candidate; CEM sampling/update is the first suspect.")

        del actions, wm_losses, env_rows
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _final_eval_actions(self, actions, action_len, filename_prefix):
        # Final evaluation from original start
        self.evaluator.history_actions = None
        self.evaluator.assign_init_cond(obs_0=self.obs_0, state_0=self.state_0)
        self.evaluator.obs_g = self.obs_g
        self.evaluator.state_g = self.state_g

        final_wm = self.planner.sub_planner.wm if hasattr(self.planner, "sub_planner") else self.wm
        if hasattr(final_wm, "init_actions"):
            final_wm.init_actions = None

        save_final_video = bool(
            self.cfg_dict.get("diagnostics", {}).get("save_final_merged_video", True)
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=save_final_video, filename=filename_prefix
        )
        if save_final_video:
            success_tag = "success" if bool(np.any(successes)) else "failure"
            print(f"[Final Video] saved merged rollout: {filename_prefix}_0_{success_tag}.mp4")
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {key: (value.item() if isinstance(value, (np.float32, np.int32, np.int64)) else value) for
                      key, value in logs.items()}
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs

    def perform_planning(self):
        from planning.mpc_park_guided import MPCPlannerGuided as MPCPlanner
        is_hierarchical = hasattr(self, 'obs_g_sub_list') and isinstance(self.planner, MPCPlanner)

        if is_hierarchical:
            phase_goal_obs_list = list(self.obs_g_sub_list) + [self.obs_g]
            phase_goal_state_list = list(self.state_g_sub_list) + [self.state_g]
            num_phases = len(phase_goal_obs_list)

            cur_obs = self.obs_0
            cur_state = self.state_0
            all_actions = []

            for phase_idx in range(num_phases):
                target_name = "final goal" if phase_idx == num_phases - 1 else f"subgoal {phase_idx+1}"
                print(f"\n=== Phase {phase_idx}: approach {target_name}/{num_phases} ===")

                # Reset MPC state for each phase
                self.planner.iter = 0
                self.planner.planned_actions = []
                self.planner.is_success = None
                self.planner.action_len = None

                # Use the same subplanner parameters for every phase.
                self._apply_phase_config(phase_idx)

                # Set evaluator goal to current subgoal
                self.evaluator.obs_g = phase_goal_obs_list[phase_idx]
                self.evaluator.state_g = phase_goal_state_list[phase_idx]

                # Accumulated actions so far for this phase's continuity setup
                accumulated = torch.cat(all_actions, dim=1) if all_actions else None
                self._setup_phase_transition(accumulated, phase_idx, num_phases, cur_obs)
                self._run_action_probe(
                    phase_idx=phase_idx,
                    cur_obs=cur_obs,
                    cur_state=cur_state,
                    goal_obs=phase_goal_obs_list[phase_idx],
                    goal_state=phase_goal_state_list[phase_idx],
                    accumulated_actions=accumulated,
                )

                if self._expert_guided_enabled():
                    self.planner.prior_actions = self._phase_expert_prior_actions(phase_idx)
                    phase_actions, _ = self.planner.plan(obs_0=cur_obs, obs_g=phase_goal_obs_list[phase_idx])
                elif self._expert_oracle_enabled():
                    phase_actions, cur_obs, cur_state = self._plan_phase_with_expert_oracle(
                        phase_idx=phase_idx,
                        cur_obs=cur_obs,
                        cur_state=cur_state,
                        goal_state=phase_goal_state_list[phase_idx],
                        accumulated_actions=accumulated,
                    )
                else:
                    self.planner.prior_actions = None
                    phase_actions, _ = self.planner.plan(obs_0=cur_obs, obs_g=phase_goal_obs_list[phase_idx])
                all_actions.append(phase_actions)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not self._expert_oracle_enabled():
                    # Get final obs/state from planner for next phase's WM start
                    cur_obs = self.planner.final_obs_0
                    cur_state = self.planner.final_state_0
                    if cur_obs is None:
                        # Fallback: use evaluator's current obs_0
                        cur_obs, _ = self.evaluator.get_init_cond()
                    if cur_state is None:
                        _, cur_state = self.evaluator.get_init_cond()

                # Set init_actions for next phase
                wm = self.planner.sub_planner.wm
                if hasattr(wm, "init_actions") and phase_actions.shape[1] >= 2:
                    init_a = phase_actions[:, -2:, :].detach().cpu()
                    # Pad to 3 frames: last 2 actions, last one repeated for 3rd
                    init_a = torch.cat([init_a, init_a[:, -1:, :]], dim=1)
                    wm.init_actions = init_a

            actions = torch.cat(all_actions, dim=1)
            action_len = np.full(actions.shape[0], actions.shape[1], dtype=float)
        else:
            actions_init = self.gt_actions if self.debug_dset_init else None
            actions, action_len = self.planner.plan(
                obs_0=self.obs_0, obs_g=self.obs_g, actions=actions_init,
            )

        prefix = getattr(self.planner, "logging_prefix", "final")
        return self._final_eval_actions(actions, action_len, f"{prefix}_merged_final")

def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict)
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    
    # 加载数据集
    datasets, traj_dsets = hydra.utils.call(
        model_cfg.env.dataset, num_hist=model_cfg.num_hist, num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip
    )
    dset = traj_dsets["valid"]

    diag_cfg = cfg_dict.get("diagnostics", {})
    if diag_cfg.get("dataset_action_distribution", False):
        diagnose_parking_dataset_actions(
            datasets,
            traj_dsets=traj_dsets,
            max_files=diag_cfg.get("dataset_max_files", None),
        )

    # 🔥🔥🔥 核心修复：把真实数据集的统计量，覆盖给 Trajectory 数据集 🔥🔥🔥
    real_dset = datasets["train"] # 获取带有真实 mean/std 的数据集
    dset.action_mean = real_dset.action_mean
    dset.action_std = real_dset.action_std
    dset.state_mean = real_dset.state_mean
    dset.state_std = real_dset.state_std
    dset.proprio_mean = real_dset.proprio_mean
    dset.proprio_std = real_dset.proprio_std

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # 👇 ================= 新增：防止模型动作幻觉的 WM Wrapper ================= 👇
    # 1. 计算物理边界 [-1, 1] 在当前模型归一化空间中的合法范围
    act_mean_10d = dset.action_mean.to(device)
    act_std_10d = dset.action_std.to(device)
    norm_min = (torch.tensor([-1.0] * 10, device=device) - act_mean_10d) / act_std_10d
    norm_max = (torch.tensor([1.0] * 10, device=device) - act_mean_10d) / act_std_10d

    # 2. 编写一个极其轻量级的拦截器，伪装成原本的 model
    # 2. 编写一个极其轻量级的拦截器，伪装成原本的 model
    class ClampedWM:
        def __init__(self, original_wm, n_min, n_max):
            self.original_wm = original_wm
            self.n_min = n_min
            self.n_max = n_max
            self.init_actions = None  # (B, 3, D) normalized actions for obs_0 encoding

        def rollout(self, obs_0, act, **kwargs):
            # 将 Planner 的动作限制在物理范围内
            clipped_act = torch.max(torch.min(act, self.n_max), self.n_min)
            num_hist = obs_0['visual'].shape[1]

            B, _, D = clipped_act.shape

            # 🔥🔥🔥 核心修复：使用真实的 init_actions 而非 dummy 零动作
            #
            # 原问题：WM.encode() 将 obs_0 的 3 帧画面与 dummy 零动作配对编码。
            # 但第 3 帧画面(setpoint 场景下车靠近车位)与零动作不符→编码空间 OOD→预测崩溃
            #
            # 修复：传入真实的归一化历史动作，使 WM 的编码输入与训练时的分布严格一致。
            #
            # 关键细节：obs_0 的 3 帧间隔 frameskip 步物理推演。设 frames=[f0,f1,f2]：
            #   - f0 → f1 由 history_actions[-2] 驱动（第 n-1 个 macro-action）
            #   - f1 → f2 由 history_actions[-1] 驱动（第 n   个 macro-action）
            #   - f2 → f3 由 clipped_act[0]   驱动（第 n+1 个 macro-action，规划的第 1 个动作）
            # 所以 encode 配对应为：
            #   encode(f0, history_actions[-2])     → 正确
            #   encode(f1, history_actions[-1])     → 正确
            #   encode(f2, history_actions[-1])     → 近似（停车场景动作≈0，误差小）
            # 最后第 3 帧用 history[-1] 复制填充，避免偷取 clipped_act[0] 缩短预测长度。
            if self.init_actions is not None:
                init_prev = self.init_actions.to(device=clipped_act.device, dtype=clipped_act.dtype)
                # Expand from (B_init, 2, D) to (B, 2, D) e.g. CEM sampling (1→400)
                if init_prev.shape[0] < B:
                    ratio = B // init_prev.shape[0]
                    init_prev = init_prev.repeat(ratio, 1, 1)
                    remainder = B - init_prev.shape[0]
                    if remainder > 0:
                        init_prev = torch.cat([init_prev, init_prev[:remainder]], dim=0)
                # 取最后 2 帧历史动作，第 3 帧复制第 2 帧作为近似
                init_act = torch.cat([
                    init_prev,                       # (B, 2, D): history_actions[-2:]
                    init_prev[:, -1:, :]             # (B, 1, D): 复制 last 作为近似
                ], dim=1)                              # (B, 3, D)
            else:
                init_act = torch.zeros(B, num_hist, D, device=clipped_act.device)
            full_act = torch.cat([init_act, clipped_act], dim=1)

            # 丢给底层的世界模型去想象
            z_obses, z = self.original_wm.rollout(obs_0, full_act, **kwargs)

            sliced_z_obses = {}
            for k, v in z_obses.items():
                # 🔥 核心修复 2：严格切片，死死卡住目标长度，彻底消灭 6 and 7 的报错！
                # 目标长度 = 初始 1 帧 + Planner 预测的未来 N 帧
                target_len = clipped_act.shape[1] + 1
                sliced_z_obses[k] = v[:, num_hist - 1 : num_hist - 1 + target_len]

            return sliced_z_obses, z
            
        def __getattr__(self, name):
            # 其他所有方法原封不动转交还原模型
            return getattr(self.original_wm, name)

    # 3. 给原模型穿上紧身衣
    model = ClampedWM(model, norm_min, norm_max)
    # 👆 ===================================================================== 👆

    frameskip = model_cfg.frameskip

    # 🔍 在正式规划之前，先运行专家策略调试视频，观察在 plan_park_3 的环境下
    #    专家能否顺利完成泊车（纯物理仿真，不经过世界模型）
    #    使用与 Planner 完全一致的 eval_seed 列表，确保场景一一对应
    eval_seeds = [cfg_dict["seed"] + n for n in range(cfg_dict["n_evals"])]

    # 提取 action_mean 和 action_std
    act_mean = dset.raw_action_mean.numpy()
    act_std = dset.raw_action_std.numpy()

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
        # 🔥 把 mean 和 std 传给 Wrapper
        return ParkingDinoWrapper(env, frameskip=frameskip, action_mean=act_mean, action_std=act_std)

    env = SubprocVectorEnv([create_wrapped_env for _ in range(cfg_dict["n_evals"])])

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict, wm=model, dset=dset, env=env,
        env_name=model_cfg.env.name, frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )
    logs = plan_workspace.perform_planning()
    return logs

@hydra.main(config_path="conf", config_name="plan_park_guided")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)

if __name__ == "__main__":
    main()
