from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib import patches

from config import EnvConfig


PRIMARY_REGION_NAMES = ["normal", "slippery", "damping"]
ANALYSIS_REGION_NAMES = [
    "normal",
    "slippery",
    "damping",
    "disturbance",
    "slippery_disturbance",
]


@dataclass
class Region:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, position: np.ndarray) -> bool:
        return (
            self.x_min <= position[0] <= self.x_max
            and self.y_min <= position[1] <= self.y_max
        )


class MultiRegionNavEnv(gym.Env):
    """
    Lightweight continuous-control navigation environment.

    The main goal is not physical realism. The environment instead creates a few
    clearly different local dynamics so that a MoE actor has a chance to develop
    expert specialization under different region modes.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mode: str = "old", config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.mode = mode
        self.rng = np.random.default_rng(self.config.seed)

        self.action_space = spaces.Box(
            low=-self.config.action_limit,
            high=self.config.action_limit,
            shape=(self.config.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.obs_dim,),
            dtype=np.float32,
        )

        self.position = np.zeros(2, dtype=np.float32)
        self.start_position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.goal_points: list[np.ndarray] = []
        self.current_goal_index = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        self.primary_regions: list[Region] = []
        self.disturbance_regions: list[Region] = []

    def describe_mode(self) -> str:
        descriptions = {
            "old": (
                "Old task distribution: multi-goal navigation over maps with normal, slippery, "
                "and damping regions. Disturbance regions are absent by default, so the agent "
                "must chain several waypoints while traversing stable but diverse local dynamics."
            ),
            "new": (
                "New task distribution: still multi-goal navigation, but disturbance regions "
                "become common and often overlap with slippery areas. The agent now has to "
                "finish a waypoint sequence while resisting persistent external pushes on "
                "high-inertia surfaces."
            ),
        }
        return descriptions[self.mode]

    def describe_stage(self, stage_name: str) -> str:
        descriptions = {
            "acquisition": (
                "Stage A / acquisition: learn the base navigation skill on old-task maps. "
                "This is where early growth schedules should improve stability and speed."
            ),
            "maturation": (
                "Stage B / maturation: continue training on old-task maps while lowering "
                "gate temperature, shrinking top-k routing, and freezing part of the model. "
                "This tests specialization and stability after growth."
            ),
            "relearning": (
                "Stage C / relearning: switch to new-task maps and measure how quickly the "
                "policy adapts to disturbance-heavy dynamics."
            ),
            "relearning_plastic": (
                "Stage C / relearning, plastic branch: start from the acquisition checkpoint "
                "before maturation, then adapt on new-task maps."
            ),
            "relearning_mature": (
                "Stage C / relearning, mature branch: start from the matured and partially "
                "frozen checkpoint, then adapt on new-task maps."
            ),
        }
        return descriptions.get(stage_name, "Training stage description is not available.")

    def set_mode(self, mode: str) -> None:
        if mode not in {"old", "new"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = seed
        self.rng = np.random.default_rng(self.config.seed)

    def _sample_rect(self, min_size: float, max_size: float) -> Region:
        width = float(self.rng.uniform(min_size, max_size))
        height = float(self.rng.uniform(min_size, max_size))
        x_min = float(
            self.rng.uniform(
                self.config.world_min + self.config.spawn_margin,
                self.config.world_max - self.config.spawn_margin - width,
            )
        )
        y_min = float(
            self.rng.uniform(
                self.config.world_min + self.config.spawn_margin,
                self.config.world_max - self.config.spawn_margin - height,
            )
        )
        return Region(
            name="normal",
            x_min=x_min,
            x_max=x_min + width,
            y_min=y_min,
            y_max=y_min + height,
        )

    def _sample_layout(self) -> None:
        self.primary_regions = []
        self.disturbance_regions = []

        primary_names = list(PRIMARY_REGION_NAMES)
        self.rng.shuffle(primary_names)
        for name in primary_names[: self.config.num_primary_regions]:
            region = self._sample_rect(min_size=2.2, max_size=3.6)
            region.name = name
            self.primary_regions.append(region)

        disturbance_count = (
            self.config.num_disturbance_regions_old
            if self.mode == "old"
            else self.config.num_disturbance_regions_new
        )
        for _ in range(disturbance_count):
            region = self._sample_rect(min_size=2.0, max_size=3.2)
            region.name = "disturbance"
            self.disturbance_regions.append(region)

    def _sample_start_position(self) -> np.ndarray:
        low = self.config.world_min + self.config.spawn_margin
        high = self.config.world_max - self.config.spawn_margin
        return self.rng.uniform(low, high, size=(2,)).astype(np.float32)

    def _sample_random_point(self) -> np.ndarray:
        low = self.config.world_min + self.config.spawn_margin
        high = self.config.world_max - self.config.spawn_margin
        return self.rng.uniform(low, high, size=(2,)).astype(np.float32)

    def _region_center(self, region: Region) -> np.ndarray:
        return np.array(
            [
                0.5 * (region.x_min + region.x_max),
                0.5 * (region.y_min + region.y_max),
            ],
            dtype=np.float32,
        )

    def _sample_goal_sequence(self, start: np.ndarray) -> list[np.ndarray]:
        num_goals = int(
            self.rng.integers(self.config.num_goals_min, self.config.num_goals_max + 1)
        )

        candidates: list[np.ndarray] = [self._region_center(region) for region in self.primary_regions]
        candidates.extend(self._region_center(region) for region in self.disturbance_regions)
        while len(candidates) < max(num_goals + 1, 5):
            candidates.append(self._sample_random_point())

        goals: list[np.ndarray] = []
        current = start.astype(np.float32)
        used_indices: set[int] = set()
        for _ in range(num_goals):
            best_idx = None
            best_distance = -1.0
            for idx, candidate in enumerate(candidates):
                if idx in used_indices:
                    continue
                if np.linalg.norm(candidate - current) < self.config.min_waypoint_distance:
                    continue
                distance = float(np.linalg.norm(candidate - current))
                if distance > best_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is None:
                attempts = 0
                while attempts < 200:
                    candidate = self._sample_random_point()
                    if np.linalg.norm(candidate - current) >= self.config.min_waypoint_distance:
                        goals.append(candidate)
                        current = candidate
                        break
                    attempts += 1
            else:
                goal = candidates[best_idx].astype(np.float32)
                used_indices.add(best_idx)
                goals.append(goal)
                current = goal
        return goals

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self._sample_layout()
        start = self._sample_start_position()
        self.start_position = start.copy()
        self.position = start
        self.goal_points = self._sample_goal_sequence(start)
        self.current_goal_index = 0
        self.goal = self.goal_points[self.current_goal_index].copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        obs = self._get_obs()
        info = self._build_info(control_cost=0.0, success=False, goal_reached=False)
        return obs, info

    def _base_region_params(self) -> dict[str, float]:
        return {"alpha": 0.86, "beta": 0.95, "disturbance": 0.0}

    def _primary_region_params(self, region_name: str) -> dict[str, float]:
        params = {
            "normal": {"alpha": 0.86, "beta": 0.95},
            "slippery": {"alpha": 0.97, "beta": 0.58},
            "damping": {"alpha": 0.62, "beta": 0.72},
        }
        return params[region_name]

    def _region_features(self, position: np.ndarray) -> dict[str, Any]:
        params = self._base_region_params()
        primary_name = "normal"
        for region in self.primary_regions:
            if region.contains(position):
                primary_name = region.name
                params.update(self._primary_region_params(region.name))
                break

        disturbance_strength = 0.0
        in_disturbance = False
        for region in self.disturbance_regions:
            if region.contains(position):
                disturbance_strength = (
                    self.config.disturbance_strength_new
                    if self.mode == "new"
                    else self.config.disturbance_strength_old
                )
                in_disturbance = True
                break

        params["disturbance"] = disturbance_strength
        one_hot = np.zeros(4, dtype=np.float32)
        if in_disturbance and primary_name == "slippery":
            region_id = 4
            region_name = "slippery_disturbance"
            one_hot[1] = 1.0
        elif in_disturbance:
            region_id = 3
            region_name = "disturbance"
            one_hot[3] = 1.0
        else:
            region_id = {"normal": 0, "slippery": 1, "damping": 2}[primary_name]
            region_name = primary_name
            one_hot[region_id] = 1.0

        return {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "disturbance": params["disturbance"],
            "region_id": region_id,
            "region_name": region_name,
            "one_hot": one_hot,
        }

    def _disturbance_vector(self, strength: float) -> np.ndarray:
        if strength <= 0.0:
            return np.zeros(2, dtype=np.float32)
        x, y = float(self.position[0]), float(self.position[1])
        base = np.array(
            [np.sin(1.7 * y + 0.35 * self.step_count), np.cos(1.3 * x - 0.25 * self.step_count)],
            dtype=np.float32,
        )
        noise = self.rng.normal(loc=0.0, scale=0.15, size=(2,)).astype(np.float32)
        return strength * (base + noise)

    def _get_obs(self) -> np.ndarray:
        region = self._region_features(self.position)
        total_goals = max(len(self.goal_points), 1)
        goal_progress = self.current_goal_index / max(total_goals - 1, 1)
        remaining_goals = (total_goals - 1 - self.current_goal_index) / max(total_goals - 1, 1)
        obs = np.concatenate(
            [
                self.position.astype(np.float32),
                self.velocity.astype(np.float32),
                (self.goal - self.position).astype(np.float32),
                self.prev_action.astype(np.float32),
                region["one_hot"],
                np.array(
                    [region["alpha"], region["beta"], region["disturbance"], goal_progress, remaining_goals],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    def _build_info(self, control_cost: float, success: bool, goal_reached: bool = False) -> dict[str, Any]:
        region = self._region_features(self.position)
        return {
            "region_id": int(region["region_id"]),
            "region_name": region["region_name"],
            "control_cost": float(control_cost),
            "success": bool(success),
            "goal_reached": bool(goal_reached),
            "goal_index": int(self.current_goal_index),
            "num_goals": int(len(self.goal_points)),
            "distance": float(np.linalg.norm(self.goal - self.position)),
        }

    def get_layout_summary(self) -> dict[str, Any]:
        def _serialize(region: Region) -> dict[str, Any]:
            return {
                "name": region.name,
                "x_min": region.x_min,
                "x_max": region.x_max,
                "y_min": region.y_min,
                "y_max": region.y_max,
            }

        return {
            "mode": self.mode,
            "mode_description": self.describe_mode(),
            "world_min": self.config.world_min,
            "world_max": self.config.world_max,
            "start": self.start_position.tolist(),
            "goal": self.goal.tolist(),
            "goal_points": [goal.tolist() for goal in self.goal_points],
            "current_goal_index": int(self.current_goal_index),
            "primary_regions": [_serialize(region) for region in self.primary_regions],
            "disturbance_regions": [_serialize(region) for region in self.disturbance_regions],
        }

    def render_rgb_array(
        self,
        trajectory: list[np.ndarray] | None = None,
        title: str | None = None,
        overlay_lines: list[str] | None = None,
    ) -> np.ndarray:
        colors = {
            "normal": "#9ecae1",
            "slippery": "#fdd0a2",
            "damping": "#a1d99b",
            "disturbance": "#fcaeae",
        }

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.set_xlim(self.config.world_min, self.config.world_max)
        ax.set_ylim(self.config.world_min, self.config.world_max)
        ax.set_aspect("equal")
        ax.set_facecolor("#f7f7f7")
        ax.grid(alpha=0.2, linewidth=0.4)

        for region in self.primary_regions:
            rect = patches.Rectangle(
                (region.x_min, region.y_min),
                region.x_max - region.x_min,
                region.y_max - region.y_min,
                linewidth=1.0,
                edgecolor="#4d4d4d",
                facecolor=colors[region.name],
                alpha=0.55,
            )
            ax.add_patch(rect)
            ax.text(
                (region.x_min + region.x_max) * 0.5,
                (region.y_min + region.y_max) * 0.5,
                region.name,
                fontsize=8,
                ha="center",
                va="center",
            )

        for region in self.disturbance_regions:
            rect = patches.Rectangle(
                (region.x_min, region.y_min),
                region.x_max - region.x_min,
                region.y_max - region.y_min,
                linewidth=1.2,
                linestyle="--",
                edgecolor="#a50f15",
                facecolor=colors["disturbance"],
                alpha=0.35,
            )
            ax.add_patch(rect)

        if trajectory:
            traj = np.asarray(trajectory, dtype=np.float32)
            ax.plot(traj[:, 0], traj[:, 1], color="#2b8cbe", linewidth=2.0, label="trajectory")

        if self.goal_points:
            goals = np.asarray(self.goal_points, dtype=np.float32)
            if len(goals) > 1:
                ax.plot(goals[:, 0], goals[:, 1], linestyle=":", color="#31a354", alpha=0.5)
            for idx, goal in enumerate(goals):
                if idx < self.current_goal_index:
                    ax.scatter(goal[0], goal[1], marker="o", s=70, c="#9e9e9e")
                elif idx == self.current_goal_index:
                    ax.scatter(goal[0], goal[1], marker="*", s=220, c="#31a354", label="current goal")
                else:
                    ax.scatter(goal[0], goal[1], marker="X", s=90, c="#74c476")
        ax.scatter(self.position[0], self.position[1], marker="o", s=90, c="#de2d26", label="agent")
        ax.legend(loc="upper right", fontsize=8)

        if title:
            ax.set_title(title)
        if overlay_lines:
            ax.text(
                0.02,
                0.02,
                "\n".join(overlay_lines),
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        rgb = rgba[..., :3].copy()
        plt.close(fig)
        return rgb

    def render(self):
        return self.render_rgb_array()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).clip(-self.config.action_limit, self.config.action_limit)
        self.step_count += 1

        prev_distance = float(np.linalg.norm(self.goal - self.position))
        region = self._region_features(self.position)
        disturbance = self._disturbance_vector(region["disturbance"])

        self.velocity = (
            region["alpha"] * self.velocity
            + region["beta"] * action
            + disturbance * self.config.dt
        )
        self.velocity = np.clip(self.velocity, -self.config.max_speed, self.config.max_speed)
        self.position = self.position + self.config.dt * self.velocity

        current_distance = float(np.linalg.norm(self.goal - self.position))
        goal_reached = current_distance <= self.config.success_radius
        success = False
        out_of_bounds = bool(
            np.any(self.position < self.config.world_min)
            or np.any(self.position > self.config.world_max)
        )
        terminated = out_of_bounds
        truncated = self.step_count >= self.config.max_steps

        progress_reward = self.config.progress_reward_scale * (prev_distance - current_distance)
        distance_penalty = self.config.distance_penalty_scale * current_distance
        control_cost = self.config.action_penalty_scale * float(np.sum(np.square(action)))
        smoothness_cost = self.config.smoothness_penalty_scale * float(
            np.sum(np.square(action - self.prev_action))
        )

        reward = progress_reward - distance_penalty - control_cost - smoothness_cost
        if goal_reached:
            if self.current_goal_index < len(self.goal_points) - 1:
                reward += self.config.waypoint_bonus
                self.current_goal_index += 1
                self.goal = self.goal_points[self.current_goal_index].copy()
            else:
                reward += self.config.success_bonus
                success = True
                terminated = True
        if out_of_bounds:
            reward -= self.config.out_of_bounds_penalty

        self.prev_action = action.astype(np.float32)
        obs = self._get_obs()
        info = self._build_info(
            control_cost=control_cost + smoothness_cost,
            success=success,
            goal_reached=goal_reached,
        )
        return obs, float(reward), terminated, truncated, info
