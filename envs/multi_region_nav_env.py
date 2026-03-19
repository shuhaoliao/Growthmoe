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

    The task is unordered multi-goal coverage, not ordered waypoint chaining.
    Each episode places one goal in each terrain type of interest so the agent
    has to cover multiple local dynamics while keeping its path short.
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
        self.goal_points = np.zeros((self.config.max_goals_supported, 2), dtype=np.float32)
        self.goal_slot_names = ["inactive"] * self.config.max_goals_supported
        self.goal_active_mask = np.zeros(self.config.max_goals_supported, dtype=bool)
        self.visited_goals = np.zeros(self.config.max_goals_supported, dtype=bool)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.path_length = 0.0

        self.primary_regions: list[Region] = []
        self.disturbance_regions: list[Region] = []

    def describe_mode(self) -> str:
        descriptions = {
            "old": (
                "Old task distribution: unordered multi-goal coverage on maps with normal, "
                "slippery, and damping regions. One goal is anchored in each primary terrain, "
                "so success requires visiting all terrain-specific targets in any order."
            ),
            "new": (
                "New task distribution: unordered multi-goal coverage with an additional goal "
                "inside disturbance-heavy terrain. The policy must still visit every goal, but "
                "now one target lies in disturbance or slippery-disturbance dynamics."
            ),
        }
        return descriptions[self.mode]

    def describe_stage(self, stage_name: str) -> str:
        descriptions = {
            "acquisition": (
                "Stage A / acquisition: learn unordered terrain-spanning goal coverage on old-task maps."
            ),
            "maturation": (
                "Stage B / maturation: continue on old-task maps while lowering gate temperature, "
                "shrinking top-k routing, and freezing part of the model."
            ),
            "relearning": (
                "Stage C / relearning: switch to new-task maps and measure adaptation under "
                "disturbance-heavy dynamics."
            ),
            "relearning_plastic": (
                "Stage C / relearning, plastic branch: start from the acquisition checkpoint "
                "before maturation, then adapt on new-task maps."
            ),
            "relearning_mature": (
                "Stage C / relearning, mature branch: start from the matured and partially frozen "
                "checkpoint, then adapt on new-task maps."
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

    def _sample_point_in_box(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> np.ndarray:
        return np.array(
            [
                self.rng.uniform(x_min, x_max),
                self.rng.uniform(y_min, y_max),
            ],
            dtype=np.float32,
        )

    def _sample_point_in_region(self, region: Region, margin: float | None = None) -> np.ndarray:
        margin = self.config.goal_region_margin if margin is None else margin
        x_min = min(region.x_min + margin, region.x_max - 1e-3)
        x_max = max(region.x_max - margin, x_min + 1e-3)
        y_min = min(region.y_min + margin, region.y_max - 1e-3)
        y_max = max(region.y_max - margin, y_min + 1e-3)
        return self._sample_point_in_box(x_min, x_max, y_min, y_max)

    def _sample_overlap_point(self, first: Region, second: Region) -> np.ndarray | None:
        margin = 0.25
        x_min = max(first.x_min, second.x_min) + margin
        x_max = min(first.x_max, second.x_max) - margin
        y_min = max(first.y_min, second.y_min) + margin
        y_max = min(first.y_max, second.y_max) - margin
        if x_max <= x_min or y_max <= y_min:
            return None
        return self._sample_point_in_box(x_min, x_max, y_min, y_max)

    def _goal_slot_names_for_mode(self) -> list[str]:
        if self.mode == "new":
            return ["normal", "slippery", "damping", "disturbance"][
                : self.config.num_goals_new
            ]
        return ["normal", "slippery", "damping"][: self.config.num_goals_old]

    def _candidate_point_for_slot(self, slot_name: str) -> np.ndarray:
        primary_map = {region.name: region for region in self.primary_regions}
        if slot_name in primary_map:
            return self._sample_point_in_region(primary_map[slot_name])

        if slot_name == "disturbance" and self.disturbance_regions:
            slippery_region = primary_map.get("slippery")
            if slippery_region is not None:
                shuffled = list(self.disturbance_regions)
                self.rng.shuffle(shuffled)
                for disturbance_region in shuffled:
                    overlap_point = self._sample_overlap_point(slippery_region, disturbance_region)
                    if overlap_point is not None:
                        return overlap_point
            region = self.disturbance_regions[
                int(self.rng.integers(0, len(self.disturbance_regions)))
            ]
            return self._sample_point_in_region(region)

        return self._sample_random_point()

    def _is_valid_goal_candidate(
        self,
        point: np.ndarray,
        start: np.ndarray,
        chosen: list[np.ndarray],
        enforce_start_distance: bool,
    ) -> bool:
        if enforce_start_distance and (
            np.linalg.norm(point - start) < self.config.min_start_goal_distance
        ):
            return False
        for existing in chosen:
            if np.linalg.norm(point - existing) < self.config.min_waypoint_distance:
                return False
        return True

    def _sample_goal_points(self, start: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
        slot_names = self._goal_slot_names_for_mode()
        max_goals = self.config.max_goals_supported
        goals = np.zeros((max_goals, 2), dtype=np.float32)
        active_mask = np.zeros(max_goals, dtype=bool)
        chosen: list[np.ndarray] = []

        for slot_idx, slot_name in enumerate(slot_names):
            goal_point: np.ndarray | None = None
            for attempt in range(80):
                candidate = self._candidate_point_for_slot(slot_name)
                enforce_start = attempt < 40
                if self._is_valid_goal_candidate(candidate, start, chosen, enforce_start):
                    goal_point = candidate
                    break
            if goal_point is None:
                fallback = self._sample_random_point()
                for attempt in range(120):
                    enforce_start = attempt < 60
                    if self._is_valid_goal_candidate(fallback, start, chosen, enforce_start):
                        goal_point = fallback
                        break
                    fallback = self._sample_random_point()
            if goal_point is None:
                goal_point = self._sample_random_point()

            goals[slot_idx] = goal_point.astype(np.float32)
            active_mask[slot_idx] = True
            chosen.append(goals[slot_idx].copy())

        padded_names = slot_names + ["inactive"] * (max_goals - len(slot_names))
        return goals, padded_names[:max_goals], active_mask

    def _count_active_goals(self) -> int:
        return int(self.goal_active_mask.sum())

    def _count_visited_goals(self) -> int:
        return int((self.goal_active_mask & self.visited_goals).sum())

    def _all_goals_visited(self) -> bool:
        active_count = self._count_active_goals()
        return active_count > 0 and self._count_visited_goals() >= active_count

    def _nearest_unvisited(self, position: np.ndarray) -> tuple[np.ndarray, float, int | None]:
        best_idx = None
        best_distance = np.inf
        for idx in range(self.config.max_goals_supported):
            if not self.goal_active_mask[idx] or self.visited_goals[idx]:
                continue
            distance = float(np.linalg.norm(self.goal_points[idx] - position))
            if distance < best_distance:
                best_distance = distance
                best_idx = idx

        if best_idx is None:
            return np.zeros(2, dtype=np.float32), 0.0, None
        return self.goal_points[best_idx].copy(), float(best_distance), int(best_idx)

    def _mark_visited_goals(self) -> np.ndarray:
        newly_visited = np.zeros(self.config.max_goals_supported, dtype=bool)
        for idx in range(self.config.max_goals_supported):
            if not self.goal_active_mask[idx] or self.visited_goals[idx]:
                continue
            distance = np.linalg.norm(self.goal_points[idx] - self.position)
            if distance <= self.config.success_radius:
                newly_visited[idx] = True
        self.visited_goals |= newly_visited
        return newly_visited

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.seed(seed)
        self._sample_layout()
        start = self._sample_start_position()
        self.start_position = start.copy()
        self.position = start
        self.velocity = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.path_length = 0.0

        self.goal_points, self.goal_slot_names, self.goal_active_mask = self._sample_goal_points(start)
        self.visited_goals = np.zeros(self.config.max_goals_supported, dtype=bool)
        self.goal, _, _ = self._nearest_unvisited(self.position)

        obs = self._get_obs()
        info = self._build_info(control_cost=0.0, success=False, new_goals_visited=0)
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
        active_count = max(self._count_active_goals(), 1)
        visited_count = self._count_visited_goals()

        goal_rel = np.zeros((self.config.max_goals_supported, 2), dtype=np.float32)
        for idx in range(self.config.max_goals_supported):
            if self.goal_active_mask[idx]:
                goal_rel[idx] = self.goal_points[idx] - self.position

        coverage_ratio = visited_count / active_count
        remaining_ratio = (active_count - visited_count) / active_count

        obs = np.concatenate(
            [
                self.position.astype(np.float32),
                self.velocity.astype(np.float32),
                self.prev_action.astype(np.float32),
                goal_rel.reshape(-1).astype(np.float32),
                self.visited_goals.astype(np.float32),
                self.goal_active_mask.astype(np.float32),
                region["one_hot"],
                np.array(
                    [
                        region["alpha"],
                        region["beta"],
                        region["disturbance"],
                        coverage_ratio,
                        remaining_ratio,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        return obs

    def _build_info(
        self,
        control_cost: float,
        success: bool,
        new_goals_visited: int = 0,
    ) -> dict[str, Any]:
        region = self._region_features(self.position)
        nearest_goal, nearest_distance, nearest_goal_idx = self._nearest_unvisited(self.position)
        active_count = max(self._count_active_goals(), 1)
        visited_count = self._count_visited_goals()
        return {
            "region_id": int(region["region_id"]),
            "region_name": region["region_name"],
            "control_cost": float(control_cost),
            "success": bool(success),
            "new_goals_visited": int(new_goals_visited),
            "goals_visited": int(visited_count),
            "num_goals": int(active_count),
            "coverage_ratio": float(visited_count / active_count),
            "remaining_goals": int(active_count - visited_count),
            "nearest_goal_index": -1 if nearest_goal_idx is None else int(nearest_goal_idx),
            "nearest_goal": nearest_goal.tolist(),
            "distance": float(nearest_distance),
            "path_length": float(self.path_length),
            "visited_mask": self.visited_goals.astype(np.int32).tolist(),
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

        nearest_goal, nearest_distance, nearest_idx = self._nearest_unvisited(self.position)
        return {
            "mode": self.mode,
            "mode_description": self.describe_mode(),
            "world_min": self.config.world_min,
            "world_max": self.config.world_max,
            "start": self.start_position.tolist(),
            "goal": nearest_goal.tolist(),
            "goal_points": self.goal_points[self.goal_active_mask].tolist(),
            "goal_slot_names": [
                self.goal_slot_names[idx]
                for idx in range(self.config.max_goals_supported)
                if self.goal_active_mask[idx]
            ],
            "visited_mask": self.visited_goals.astype(np.int32).tolist(),
            "active_goal_mask": self.goal_active_mask.astype(np.int32).tolist(),
            "nearest_goal_index": -1 if nearest_idx is None else int(nearest_idx),
            "nearest_goal_distance": float(nearest_distance),
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

        nearest_goal, _, nearest_idx = self._nearest_unvisited(self.position)
        for idx in range(self.config.max_goals_supported):
            if not self.goal_active_mask[idx]:
                continue
            goal = self.goal_points[idx]
            slot_name = self.goal_slot_names[idx]
            if self.visited_goals[idx]:
                ax.scatter(goal[0], goal[1], marker="o", s=70, c="#9e9e9e")
            elif nearest_idx is not None and idx == nearest_idx:
                ax.scatter(goal[0], goal[1], marker="*", s=220, c="#31a354", label="nearest unvisited")
            else:
                ax.scatter(goal[0], goal[1], marker="X", s=90, c="#74c476")
            ax.text(goal[0] + 0.08, goal[1] + 0.08, slot_name, fontsize=7, color="#1b4332")

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

        _, prev_distance, _ = self._nearest_unvisited(self.position)
        prev_position = self.position.copy()
        region = self._region_features(self.position)
        disturbance = self._disturbance_vector(region["disturbance"])

        self.velocity = (
            region["alpha"] * self.velocity
            + region["beta"] * action
            + disturbance * self.config.dt
        )
        self.velocity = np.clip(self.velocity, -self.config.max_speed, self.config.max_speed)
        self.position = self.position + self.config.dt * self.velocity
        self.path_length += float(np.linalg.norm(self.position - prev_position))

        newly_visited = self._mark_visited_goals()
        new_goals_visited = int(newly_visited.sum())
        self.goal, current_distance, _ = self._nearest_unvisited(self.position)

        out_of_bounds = bool(
            np.any(self.position < self.config.world_min)
            or np.any(self.position > self.config.world_max)
        )
        all_goals_visited = self._all_goals_visited()
        success = bool(all_goals_visited and not out_of_bounds)
        terminated = bool(out_of_bounds or success)
        truncated = self.step_count >= self.config.max_steps

        prev_distance_value = prev_distance if np.isfinite(prev_distance) else 0.0
        current_distance_value = current_distance if np.isfinite(current_distance) else 0.0
        progress_reward = self.config.progress_reward_scale * (
            prev_distance_value - current_distance_value
        )
        distance_penalty = self.config.distance_penalty_scale * current_distance_value
        control_cost = self.config.action_penalty_scale * float(np.sum(np.square(action)))
        smoothness_cost = self.config.smoothness_penalty_scale * float(
            np.sum(np.square(action - self.prev_action))
        )
        step_penalty = self.config.step_penalty_scale

        reward = (
            progress_reward
            - distance_penalty
            - control_cost
            - smoothness_cost
            - step_penalty
            + self.config.goal_bonus * new_goals_visited
        )
        if success:
            reward += self.config.success_bonus
        if out_of_bounds:
            reward -= self.config.out_of_bounds_penalty

        self.prev_action = action.astype(np.float32)
        obs = self._get_obs()
        info = self._build_info(
            control_cost=control_cost + smoothness_cost,
            success=success,
            new_goals_visited=new_goals_visited,
        )
        return obs, float(reward), terminated, truncated, info
