import glob
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np


def _wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _local_delta(start_state, goal_state):
    dx = goal_state[..., 0] - start_state[..., 0]
    dy = goal_state[..., 1] - start_state[..., 1]
    h = start_state[..., 2]
    c = np.cos(-h)
    s = np.sin(-h)
    local_x = c * dx - s * dy
    local_y = s * dx + c * dy
    dh = _wrap_angle(goal_state[..., 2] - start_state[..., 2])
    dist = np.sqrt(dx * dx + dy * dy)
    return np.stack([local_x, local_y, dh, dist], axis=-1).astype(np.float32)


def _as_hwc_uint8(image):
    image = np.asarray(image)
    while image.ndim > 3:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.min() >= -1.1 and image.max() <= 1.1:
            image = (image + 1.0) * 127.5
        elif image.max() <= 1.1:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def extract_ego_pose_from_image(image):
    """Estimate ego pose in image coordinates from the rendered ego colors.

    Returns [x_px, y_up_px, heading], where y_up is -row. This is not a metric
    world pose; it is only used consistently for nearest-neighbor maneuver
    retrieval across the same BEV renderer.
    """
    image = _as_hwc_uint8(image).astype(np.int16)
    head = np.array([100, 200, 255], dtype=np.int16)
    tail = np.array([255, 175, 35], dtype=np.int16)
    mid = np.array([100, 100, 100], dtype=np.int16)

    head_mask = np.linalg.norm(image - head, axis=-1) < 40
    mask = (
        head_mask
        | (np.linalg.norm(image - tail, axis=-1) < 45)
        | (np.linalg.norm(image - mid, axis=-1) < 25)
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    best_label = -1
    best_score = -1
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        score = int((head_mask & (labels == label)).sum())
        if score > best_score:
            best_score = score
            best_label = label
    if best_label < 0 or best_score <= 0:
        return None

    comp = labels == best_label
    ys, xs = np.where(comp)
    hys, hxs = np.where(head_mask & comp)
    cx = float(xs.mean())
    cy = float(ys.mean())
    if len(hxs) > 0:
        hx = float(hxs.mean())
        hy = float(hys.mean())
    else:
        hx, hy = cx, cy

    vx = hx - cx
    vy = -(hy - cy)
    if vx * vx + vy * vy < 4.0 and len(xs) >= 5:
        pts = np.stack([xs, -ys], axis=1).astype(np.float32)
        pts = pts - pts.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        vx, vy = float(vt[0, 0]), float(vt[0, 1])
        if vx * (hx - cx) + vy * (-(hy - cy)) < 0:
            vx, vy = -vx, -vy
    heading = math.atan2(vy, vx)
    return np.array([cx, -cy, heading], dtype=np.float32)


def extract_ego_poses_from_images(images):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    poses = []
    for image in images:
        pose = extract_ego_pose_from_image(image)
        if pose is None:
            raise ValueError("Failed to extract ego pose from image for retrieval.")
        poses.append(pose)
    return np.stack(poses, axis=0)


@dataclass
class RetrievalResult:
    actions: np.ndarray
    distances: np.ndarray
    file_indices: np.ndarray
    start_frames: np.ndarray
    start_states: np.ndarray
    end_states: np.ndarray
    start_proprios: np.ndarray
    end_proprios: np.ndarray


class TrajectoryRetrievalPrior:
    """Nearest-neighbor action prior from offline parking trajectories.

    The library stores short physical action segments. Retrieval uses relative
    start-to-goal geometry in the vehicle's local frame, so it is a data prior
    rather than a test-time expert/controller call.
    """

    def __init__(
        self,
        data_path,
        horizon,
        frameskip,
        top_k=1,
        max_files=None,
        max_segments_per_file=None,
        stride=1,
        weights=None,
        include_absolute_weight=0.0,
        start_proprio_weight=None,
        feature_source="image_pose",
    ):
        self.data_path = data_path
        self.horizon = int(horizon)
        self.frameskip = int(frameskip)
        self.top_k = int(top_k)
        self.max_files = None if max_files is None else int(max_files)
        self.max_segments_per_file = (
            None if max_segments_per_file is None else int(max_segments_per_file)
        )
        self.stride = max(1, int(stride))
        self.weights = np.asarray(
            weights if weights is not None else [1.0, 1.0, 2.0, 0.25],
            dtype=np.float32,
        )
        self.include_absolute_weight = float(include_absolute_weight)
        self.start_proprio_weight = (
            None
            if start_proprio_weight is None
            else np.asarray(start_proprio_weight, dtype=np.float32)
        )
        self.feature_source = str(feature_source)
        self._built = False

    @property
    def raw_horizon(self):
        return self.horizon * self.frameskip

    def build(self):
        files = sorted(glob.glob(os.path.join(self.data_path, "*.npz")))
        if self.max_files is not None:
            files = files[: self.max_files]
        if not files:
            raise FileNotFoundError(f"No .npz files found for retrieval: {self.data_path}")

        features = []
        actions = []
        file_indices = []
        start_frames = []
        start_states = []
        end_states = []
        start_proprios = []
        end_proprios = []

        for file_idx, fpath in enumerate(files):
            try:
                with np.load(fpath) as data:
                    states = np.asarray(data["state"], dtype=np.float32)
                    raw_actions = np.asarray(data["action"], dtype=np.float32)
                    images = np.asarray(data["image"], dtype=np.uint8) if self.feature_source == "image_pose" else None
            except Exception as exc:
                print(f"[RetrievalPrior] skip unreadable file {fpath}: {exc}", flush=True)
                continue

            max_start = min(len(raw_actions), len(states) - 1) - self.raw_horizon
            if max_start < 0:
                continue
            starts = np.arange(0, max_start + 1, self.stride, dtype=np.int32)
            if (
                self.max_segments_per_file is not None
                and len(starts) > self.max_segments_per_file
            ):
                idx = np.linspace(0, len(starts) - 1, self.max_segments_per_file).astype(np.int32)
                starts = starts[idx]

            image_poses = None
            if self.feature_source == "image_pose":
                needed = sorted(set(starts.tolist() + (starts + self.raw_horizon).tolist()))
                image_poses = {}
                for frame_idx in needed:
                    pose = extract_ego_pose_from_image(images[frame_idx])
                    if pose is not None:
                        image_poses[int(frame_idx)] = pose

            for start in starts:
                end = start + self.raw_horizon
                segment = raw_actions[start:end]
                if segment.shape != (self.raw_horizon, 2):
                    continue
                if self.feature_source == "image_pose":
                    s0 = image_poses.get(int(start))
                    sg = image_poses.get(int(end))
                    if s0 is None or sg is None:
                        continue
                elif self.feature_source == "state":
                    s0 = states[start]
                    sg = states[end]
                else:
                    raise ValueError(f"Unknown retrieval feature_source: {self.feature_source}")
                features.append(_local_delta(s0[None], sg[None])[0])
                actions.append(segment.reshape(self.horizon, self.frameskip * 2))
                file_indices.append(file_idx)
                start_frames.append(start)
                start_states.append(s0)
                end_states.append(sg)
                start_proprios.append(states[start])
                end_proprios.append(states[end])

        if not actions:
            raise RuntimeError(
                f"Retrieval library is empty: data_path={self.data_path}, "
                f"horizon={self.horizon}, frameskip={self.frameskip}"
            )

        self.files = files
        self.features = np.asarray(features, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.file_indices = np.asarray(file_indices, dtype=np.int32)
        self.start_frames = np.asarray(start_frames, dtype=np.int32)
        self.start_states = np.asarray(start_states, dtype=np.float32)
        self.end_states = np.asarray(end_states, dtype=np.float32)
        self.start_proprios = np.asarray(start_proprios, dtype=np.float32)
        self.end_proprios = np.asarray(end_proprios, dtype=np.float32)
        self._built = True
        print(
            f"[RetrievalPrior] built {len(self.actions)} segments from {len(files)} files "
            f"(H={self.horizon}, frameskip={self.frameskip}, raw={self.raw_horizon}, "
            f"feature_source={self.feature_source}).",
            flush=True,
        )
        return self

    def retrieve(self, current_state, goal_state, top_k=None, current_proprio=None):
        if not self._built:
            self.build()
        current_state = np.asarray(current_state, dtype=np.float32)
        goal_state = np.asarray(goal_state, dtype=np.float32)
        if current_state.ndim == 1:
            current_state = current_state[None]
        if goal_state.ndim == 1:
            goal_state = goal_state[None]
        if current_state.shape[0] != goal_state.shape[0]:
            raise ValueError(
                f"current_state and goal_state batch mismatch: "
                f"{current_state.shape} vs {goal_state.shape}"
            )
        if current_proprio is not None:
            current_proprio = np.asarray(current_proprio, dtype=np.float32)
            if current_proprio.ndim == 1:
                current_proprio = current_proprio[None]
            if current_proprio.shape[0] != current_state.shape[0]:
                raise ValueError(
                    f"current_proprio and current_state batch mismatch: "
                    f"{current_proprio.shape} vs {current_state.shape}"
                )

        k = int(top_k if top_k is not None else self.top_k)
        k = max(1, min(k, len(self.actions)))
        query = _local_delta(current_state, goal_state)
        results = []
        all_distances = []

        weighted_library = self.features * self.weights
        for b in range(query.shape[0]):
            diff = weighted_library - query[b] * self.weights
            dist = np.sum(diff * diff, axis=-1)
            if self.include_absolute_weight > 0:
                abs_diff = self.start_states[:, :2] - current_state[b, :2]
                dist = dist + self.include_absolute_weight * np.sum(abs_diff * abs_diff, axis=-1)
            if (
                current_proprio is not None
                and self.start_proprio_weight is not None
                and self.start_proprio_weight.size > 0
            ):
                dim = min(self.start_proprio_weight.shape[0], self.start_proprios.shape[1])
                proprio_diff = (
                    self.start_proprios[:, :dim] - current_proprio[b, :dim]
                ) * self.start_proprio_weight[:dim]
                dist = dist + np.sum(proprio_diff * proprio_diff, axis=-1)
            idx = np.argpartition(dist, k - 1)[:k]
            idx = idx[np.argsort(dist[idx])]
            results.append(idx)
            all_distances.append(dist)

        indices = np.stack(results, axis=0)
        return RetrievalResult(
            actions=self.actions[indices],
            distances=np.take_along_axis(np.stack(all_distances, axis=0), indices, axis=1),
            file_indices=self.file_indices[indices],
            start_frames=self.start_frames[indices],
            start_states=self.start_states[indices],
            end_states=self.end_states[indices],
            start_proprios=self.start_proprios[indices],
            end_proprios=self.end_proprios[indices],
        )

    def proposal_mean(
        self,
        current_state,
        goal_state,
        top_k=None,
        mode="nearest",
        current_proprio=None,
    ):
        result = self.retrieve(
            current_state,
            goal_state,
            top_k=top_k,
            current_proprio=current_proprio,
        )
        if mode in ("nearest", "bank") or result.actions.shape[1] == 1:
            return result.actions[:, 0], result
        if mode == "mean":
            return result.actions.mean(axis=1), result
        if mode == "softmax":
            weights = np.exp(-result.distances / max(float(np.std(result.distances)), 1e-6))
            weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
            return np.sum(result.actions * weights[:, :, None, None], axis=1), result
        raise ValueError(f"Unknown retrieval proposal mode: {mode}")

    def proposal_mean_from_images(
        self,
        current_images,
        goal_images,
        top_k=None,
        mode="nearest",
        current_proprio=None,
    ):
        current_pose = extract_ego_poses_from_images(current_images)
        goal_pose = extract_ego_poses_from_images(goal_images)
        actions, result = self.proposal_mean(
            current_pose,
            goal_pose,
            top_k=top_k,
            mode=mode,
            current_proprio=current_proprio,
        )
        return actions, result, current_pose, goal_pose
