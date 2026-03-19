from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from config import EnvConfig

try:
    from gymnasium.envs.box2d.bipedal_walker import (
        BipedalWalker,
        MOTORS_TORQUE,
        TERRAIN_GRASS,
        TERRAIN_HEIGHT,
        TERRAIN_LENGTH,
        TERRAIN_STARTPAD,
        TERRAIN_STEP,
    )

    _BIPEDAL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - handled at runtime for missing Box2D
    BipedalWalker = gym.Env  # type: ignore[assignment]
    MOTORS_TORQUE = 80
    TERRAIN_GRASS = 10
    TERRAIN_HEIGHT = 400 / 30.0 / 4
    TERRAIN_LENGTH = 200
    TERRAIN_STARTPAD = 20
    TERRAIN_STEP = 14 / 30.0
    _BIPEDAL_IMPORT_ERROR = exc


class DiverseBipedalWalkerEnv(BipedalWalker):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}
    region_labels = ["flat", "uphill", "downhill", "rough"]

    def __init__(
        self,
        mode: str = "old",
        config: EnvConfig | None = None,
        render_mode: str = "rgb_array",
    ):
        if _BIPEDAL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "BipedalWalker support is unavailable. Install Box2D first, "
                'for example `pip install swig` and then `pip install "gymnasium[box2d]"`.'
            ) from _BIPEDAL_IMPORT_ERROR

        self.config = config or EnvConfig(env_name="bipedal_diverse")
        self.mode = mode
        self.section_plan: list[dict[str, float | int | str]] = []
        self.terrain_type_per_step: list[str] = ["flat"] * TERRAIN_LENGTH
        self.active_region_mask = np.zeros(len(self.region_labels), dtype=bool)
        self.visited_region_mask = np.zeros(len(self.region_labels), dtype=bool)
        self.position = np.zeros(2, dtype=np.float32)
        self.path_length = 0.0
        self.progress_ratio = 0.0
        self._bootstrapping_reset = False
        self._type_to_id = {name: idx for idx, name in enumerate(self.region_labels)}
        super().__init__(render_mode=render_mode, hardcore=False)

    def describe_mode(self) -> str:
        descriptions = {
            "old": (
                "Single-task terrain benchmark with flat, steep uphill, steep downhill, and rough segments. "
                "This mode emphasizes whether MoE routes specialize across terrain regimes "
                "and whether growth accelerates early learning."
            ),
            "new": (
                "A harder variant of the same terrain-mixture benchmark with longer steep ramps, "
                "denser pits, rougher undulations, and more pronounced terrain features."
            ),
        }
        return descriptions[self.mode]

    def describe_stage(self, stage_name: str) -> str:
        descriptions = {
            "acquisition": (
                "Single-task acquisition on mixed-terrain BipedalWalker. "
                "The comparison target is training speed and terrain-conditioned specialization."
            ),
            "maturation": (
                "Optional same-task maturation on mixed terrain. "
                "Useful only if you want to inspect routing entropy reduction on the same benchmark."
            ),
            "relearning": (
                "Optional harder-terrain continuation. "
                "This project currently focuses on single-task specialization, so this stage is secondary."
            ),
            "relearning_plastic": (
                "Plastic branch on the harder terrain mixture. "
                "This stage is optional for the current benchmark focus."
            ),
            "relearning_mature": (
                "Mature branch on the harder terrain mixture. "
                "This stage is optional for the current benchmark focus."
            ),
        }
        return descriptions.get(stage_name, "Terrain-mix benchmark stage.")

    def set_mode(self, mode: str) -> None:
        if mode not in {"old", "new"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = seed

    def _mode_settings(self) -> dict[str, Any]:
        section_min = max(10, int(self.config.bipedal_section_min_steps))
        section_max = max(section_min + 2, int(self.config.bipedal_section_max_steps))
        if self.mode == "new":
            return {
                "weights": np.asarray([0.18, 0.25, 0.22, 0.35], dtype=np.float64),
                "section_min": max(14, section_min - 2),
                "section_max": max(section_min, section_max - 2),
                "slope_scale": float(self.config.bipedal_new_slope_scale),
                "roughness": float(self.config.bipedal_new_roughness),
                "flat_noise": float(self.config.bipedal_flat_noise) * 1.35,
                "height_clip": float(self.config.bipedal_height_clip),
                "slope_angle_min_deg": float(self.config.bipedal_slope_angle_min_deg),
                "slope_angle_max_deg": float(self.config.bipedal_slope_angle_max_deg),
                "rough_pit_depth": float(self.config.bipedal_rough_pit_depth) * 1.25,
                "rough_pit_spacing": max(2, int(self.config.bipedal_rough_pit_spacing) - 1),
            }
        return {
            "weights": np.asarray([0.30, 0.24, 0.24, 0.22], dtype=np.float64),
            "section_min": section_min,
            "section_max": section_max,
            "slope_scale": float(self.config.bipedal_old_slope_scale),
            "roughness": float(self.config.bipedal_old_roughness),
            "flat_noise": float(self.config.bipedal_flat_noise),
            "height_clip": float(self.config.bipedal_height_clip),
            "slope_angle_min_deg": float(self.config.bipedal_slope_angle_min_deg),
            "slope_angle_max_deg": float(self.config.bipedal_slope_angle_max_deg),
            "rough_pit_depth": float(self.config.bipedal_rough_pit_depth),
            "rough_pit_spacing": int(self.config.bipedal_rough_pit_spacing),
        }

    def _append_section_record(self, name: str, start_idx: int, end_idx: int) -> None:
        if end_idx <= start_idx:
            return
        if (
            self.section_plan
            and self.section_plan[-1]["name"] == name
            and int(self.section_plan[-1]["step_end"]) == start_idx
        ):
            self.section_plan[-1]["step_end"] = end_idx
            self.section_plan[-1]["x_end"] = end_idx * TERRAIN_STEP
            return
        self.section_plan.append(
            {
                "name": name,
                "step_start": start_idx,
                "step_end": end_idx,
                "x_start": start_idx * TERRAIN_STEP,
                "x_end": end_idx * TERRAIN_STEP,
            }
        )

    def _sample_section_sequence(self, settings: dict[str, Any]) -> list[tuple[str, int]]:
        remaining = TERRAIN_LENGTH - TERRAIN_STARTPAD
        sequence: list[tuple[str, int]] = []
        required = ["flat", "uphill", "downhill", "rough"]
        self.np_random.shuffle(required)

        while remaining > 0:
            if required:
                terrain_name = required.pop(0)
            else:
                terrain_name = str(self.np_random.choice(self.region_labels, p=settings["weights"]))
            section_length = int(
                self.np_random.integers(settings["section_min"], settings["section_max"] + 1)
            )
            section_length = min(section_length, remaining)
            sequence.append((terrain_name, section_length))
            remaining -= section_length
        return sequence

    def _sample_section_profile(
        self,
        terrain_name: str,
        section_length: int,
        current_y: float,
        settings: dict[str, Any],
    ) -> tuple[list[float], float]:
        values: list[float] = []
        center_y = TERRAIN_HEIGHT
        clip_delta = settings["height_clip"]
        flat_noise = settings["flat_noise"]

        if terrain_name == "flat":
            for _ in range(section_length):
                drift = 0.16 * (center_y - current_y)
                current_y += drift + float(self.np_random.uniform(-flat_noise, flat_noise))
                current_y = float(np.clip(current_y, center_y - clip_delta, center_y + clip_delta))
                values.append(current_y)
            return values, current_y

        if terrain_name in {"uphill", "downhill"}:
            base_angle_deg = float(
                self.np_random.uniform(
                    settings["slope_angle_min_deg"],
                    settings["slope_angle_max_deg"],
                )
            )
            angle_deg = float(
                np.clip(
                    base_angle_deg * float(settings["slope_scale"]),
                    settings["slope_angle_min_deg"],
                    settings["slope_angle_max_deg"] - 2.0,
                )
            )
            slope_step = math.tan(math.radians(angle_deg)) * TERRAIN_STEP
            if terrain_name == "downhill":
                slope_step *= -1.0
            section_origin = current_y
            local_clip = max(clip_delta, abs(slope_step) * section_length + 0.75)
            local_min = section_origin - local_clip
            local_max = section_origin + local_clip
            section_bias = 0.0
            if terrain_name == "uphill":
                section_bias = 0.20 * clip_delta
            else:
                section_bias = -0.20 * clip_delta
            for _ in range(section_length):
                drift = 0.01 * ((center_y + section_bias) - current_y)
                noise = float(self.np_random.uniform(-0.7 * flat_noise, 0.7 * flat_noise))
                current_y += slope_step + drift + noise
                current_y = float(np.clip(current_y, local_min, local_max))
                values.append(current_y)
            return values, current_y

        amplitude = float(self.np_random.uniform(0.08, 0.14) * settings["roughness"])
        amplitude_secondary = amplitude * float(self.np_random.uniform(0.4, 0.8))
        frequency = float(self.np_random.uniform(1.8, 3.2))
        secondary_frequency = frequency * float(self.np_random.uniform(2.4, 4.2))
        phase_a = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        phase_b = float(self.np_random.uniform(0.0, 2.0 * math.pi))
        section_origin = current_y
        local_clip = max(
            clip_delta,
            amplitude + amplitude_secondary + 4.0 * float(settings["rough_pit_depth"]),
        )
        local_min = section_origin - local_clip
        local_max = section_origin + local_clip
        pit_spacing = max(2, int(settings["rough_pit_spacing"]))
        pit_count = max(2, section_length // pit_spacing)
        candidate_centers = np.arange(2, max(3, section_length - 2))
        if len(candidate_centers) >= pit_count:
            pit_centers = sorted(
                int(value)
                for value in self.np_random.choice(candidate_centers, size=pit_count, replace=False)
            )
        else:
            pit_centers = [int(value) for value in candidate_centers]
        pit_specs = [
            (
                center,
                float(self.np_random.uniform(0.9, 1.8)),
                float(
                    self.np_random.uniform(
                        0.7 * settings["rough_pit_depth"],
                        1.3 * settings["rough_pit_depth"],
                    )
                ),
            )
            for center in pit_centers
        ]

        for step_idx in range(section_length):
            progress = (step_idx + 1) / max(section_length, 1)
            wave = amplitude * math.sin(2.0 * math.pi * frequency * progress + phase_a)
            wave += amplitude_secondary * math.sin(
                2.0 * math.pi * secondary_frequency * progress + phase_b
            )
            pits = 0.0
            for center, width, depth in pit_specs:
                distance = (step_idx - center) / max(width, 1e-6)
                pits -= depth * math.exp(-0.5 * distance * distance)
            target = section_origin + wave + pits + 0.10 * (center_y - section_origin) * progress
            noise = float(self.np_random.uniform(-1.9 * flat_noise, 1.9 * flat_noise))
            current_y = 0.52 * current_y + 0.48 * target + noise
            current_y = float(np.clip(current_y, local_min, local_max))
            values.append(current_y)
        return values, current_y

    def _generate_terrain(self, hardcore: bool) -> None:  # pragma: no cover - exercised in runtime
        del hardcore
        settings = self._mode_settings()
        self.terrain = []
        self.terrain_x = [i * TERRAIN_STEP for i in range(TERRAIN_LENGTH)]
        self.terrain_y = []
        self.terrain_poly = []
        self.section_plan = []
        self.terrain_type_per_step = []

        current_y = TERRAIN_HEIGHT
        for step_idx in range(TERRAIN_STARTPAD):
            self.terrain_y.append(current_y)
            self.terrain_type_per_step.append("flat")
        self._append_section_record("flat", 0, TERRAIN_STARTPAD)

        current_step = TERRAIN_STARTPAD
        for terrain_name, section_length in self._sample_section_sequence(settings):
            values, current_y = self._sample_section_profile(
                terrain_name=terrain_name,
                section_length=section_length,
                current_y=current_y,
                settings=settings,
            )
            self.terrain_y.extend(values)
            self.terrain_type_per_step.extend([terrain_name] * len(values))
            self._append_section_record(terrain_name, current_step, current_step + len(values))
            current_step += len(values)

        if len(self.terrain_y) < TERRAIN_LENGTH:
            fill_count = TERRAIN_LENGTH - len(self.terrain_y)
            self.terrain_y.extend([current_y] * fill_count)
            self.terrain_type_per_step.extend(["flat"] * fill_count)
            self._append_section_record("flat", current_step, current_step + fill_count)
        elif len(self.terrain_y) > TERRAIN_LENGTH:
            self.terrain_y = self.terrain_y[:TERRAIN_LENGTH]
            self.terrain_type_per_step = self.terrain_type_per_step[:TERRAIN_LENGTH]

        for idx in range(TERRAIN_LENGTH - 1):
            edge_vertices = [
                (self.terrain_x[idx], self.terrain_y[idx]),
                (self.terrain_x[idx + 1], self.terrain_y[idx + 1]),
            ]
            self.fd_edge.shape.vertices = edge_vertices
            terrain_body = self.world.CreateStaticBody(fixtures=self.fd_edge)
            terrain_name = self.terrain_type_per_step[idx]
            if terrain_name == "flat":
                color = (78, 176, 101)
            elif terrain_name == "uphill":
                color = (224, 165, 77)
            elif terrain_name == "downhill":
                color = (88, 156, 218)
            else:
                color = (163, 120, 196)
            terrain_body.color1 = color
            terrain_body.color2 = color
            self.terrain.append(terrain_body)

            fill_poly = edge_vertices + [(edge_vertices[1][0], 0), (edge_vertices[0][0], 0)]
            self.terrain_poly.append((fill_poly, color))

        self.terrain.reverse()

    def _sync_position(self) -> None:
        if getattr(self, "hull", None) is None:
            return
        self.position = np.array(
            [float(self.hull.position.x), float(self.hull.position.y)],
            dtype=np.float32,
        )

    def _region_name_at_position(self, position: np.ndarray) -> str:
        step_idx = int(np.clip(position[0] / TERRAIN_STEP, 0, TERRAIN_LENGTH - 1))
        if not self.terrain_type_per_step:
            return "flat"
        return self.terrain_type_per_step[step_idx]

    def _region_info(self, position: np.ndarray) -> tuple[int, str]:
        region_name = self._region_name_at_position(position)
        return self._type_to_id[region_name], region_name

    def _active_region_count(self) -> int:
        return int(self.active_region_mask.sum())

    def _visited_region_count(self) -> int:
        return int((self.active_region_mask & self.visited_region_mask).sum())

    def _finish_x(self) -> float:
        return float((TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP)

    def _build_info(
        self,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        base_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del truncated
        region_id, region_name = self._region_info(self.position)
        self.visited_region_mask[region_id] = True
        active_count = max(self._active_region_count(), 1)
        visited_count = self._visited_region_count()
        success = bool(self.position[0] >= self._finish_x())
        control_cost = float(0.00035 * MOTORS_TORQUE * np.clip(np.abs(action), 0, 1).sum())
        self.progress_ratio = float(np.clip(self.position[0] / max(self._finish_x(), 1e-6), 0.0, 1.0))
        info = dict(base_info or {})
        info.update(
            {
                "region_id": int(region_id),
                "region_name": region_name,
                "control_cost": control_cost,
                "success": success,
                "new_goals_visited": 0,
                "goals_visited": int(visited_count),
                "num_goals": int(active_count),
                "coverage_ratio": float(visited_count / active_count),
                "remaining_goals": int(active_count - visited_count),
                "path_length": float(self.path_length),
                "distance": float(max(self._finish_x() - self.position[0], 0.0)),
                "progress_ratio": self.progress_ratio,
                "visited_mask": self.visited_region_mask.astype(np.int32).tolist(),
                "terminated": bool(terminated),
            }
        )
        return info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options and "mode" in options:
            self.set_mode(str(options["mode"]))
        self._bootstrapping_reset = True
        obs, _ = super().reset(seed=seed, options=options)
        self._bootstrapping_reset = False

        self.path_length = 0.0
        self.progress_ratio = 0.0
        self._sync_position()
        self.active_region_mask = np.zeros(len(self.region_labels), dtype=bool)
        for terrain_name in self.terrain_type_per_step:
            self.active_region_mask[self._type_to_id[terrain_name]] = True
        self.visited_region_mask = np.zeros(len(self.region_labels), dtype=bool)
        region_id, _ = self._region_info(self.position)
        self.visited_region_mask[region_id] = True
        info = self._build_info(
            action=np.zeros(self.action_space.shape, dtype=np.float32),
            terminated=False,
            truncated=False,
            base_info={},
        )
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        prev_position = self.position.copy()
        obs, reward, terminated, truncated, info = super().step(action)
        self._sync_position()
        if not self._bootstrapping_reset:
            self.path_length += float(np.linalg.norm(self.position - prev_position))
        enhanced_info = self._build_info(
            action=action,
            terminated=terminated,
            truncated=truncated,
            base_info=info,
        )
        return obs, reward, terminated, truncated, enhanced_info

    def get_layout_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "mode_description": self.describe_mode(),
            "terrain_step": TERRAIN_STEP,
            "terrain_length": TERRAIN_LENGTH,
            "finish_x": self._finish_x(),
            "section_plan": self.section_plan,
            "active_regions": [
                name
                for idx, name in enumerate(self.region_labels)
                if self.active_region_mask[idx]
            ],
            "visited_regions": [
                name
                for idx, name in enumerate(self.region_labels)
                if self.visited_region_mask[idx]
            ],
            "progress_ratio": self.progress_ratio,
        }

    def render_rgb_array(
        self,
        trajectory: list[np.ndarray] | None = None,
        title: str | None = None,
        overlay_lines: list[str] | None = None,
    ) -> np.ndarray:
        del trajectory
        frame = self.render()
        if frame is None:
            return np.zeros((400, 600, 3), dtype=np.uint8)

        image = Image.fromarray(frame).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        text_lines: list[str] = []
        if title:
            text_lines.append(title)
        if overlay_lines:
            text_lines.extend(overlay_lines)
        if text_lines:
            line_height = 18
            box_height = 12 + line_height * len(text_lines)
            draw.rectangle((10, 10, 360, 10 + box_height), fill=(255, 255, 255, 180))
            for idx, line in enumerate(text_lines):
                draw.text((18, 16 + idx * line_height), line, fill=(20, 20, 20, 255))

        progress_left = 14
        progress_top = image.size[1] - 24
        progress_width = 220
        progress_fill = int(progress_width * float(np.clip(self.progress_ratio, 0.0, 1.0)))
        draw.rectangle(
            (progress_left, progress_top, progress_left + progress_width, progress_top + 10),
            outline=(255, 255, 255, 220),
            fill=(255, 255, 255, 80),
        )
        draw.rectangle(
            (progress_left, progress_top, progress_left + progress_fill, progress_top + 10),
            fill=(46, 204, 113, 220),
        )

        composed = Image.alpha_composite(image, overlay).convert("RGB")
        return np.asarray(composed, dtype=np.uint8)
