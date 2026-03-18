from __future__ import annotations

import math
from typing import Any

import numpy as np

from config import ExperimentConfig


def clamp_progress(progress: float) -> float:
    return max(0.0, min(1.0, float(progress)))


def linear_curve(progress: float) -> float:
    return clamp_progress(progress)


def sigmoid_curve(progress: float, midpoint: float, k: float) -> float:
    progress = clamp_progress(progress)
    return 1.0 / (1.0 + math.exp(-k * (progress - midpoint)))


def gompertz_curve(progress: float, b: float, c: float) -> float:
    progress = clamp_progress(progress)
    return math.exp(-b * math.exp(-c * progress))


def interpolate(start: float, end: float, weight: float) -> float:
    return start + (end - start) * clamp_progress(weight)


def old_total_training_steps(cfg: ExperimentConfig) -> int:
    return max(cfg.stage_steps["acquisition"] + cfg.stage_steps["maturation"], 1)


def resolved_action_limit(cfg: ExperimentConfig) -> float:
    env_limit = getattr(cfg.env, "action_limit", None)
    if env_limit is not None:
        return float(env_limit)
    return float(cfg.growth.action_limit)


def gpo_growth_weight(
    progress: float,
    cfg: ExperimentConfig,
    global_step: int | float | None = None,
) -> float:
    progress = clamp_progress(progress)
    if global_step is None:
        return progress

    gcfg = cfg.growth
    total_steps = float(old_total_training_steps(cfg))
    capped_step = max(0.0, min(float(global_step), total_steps))
    raw_start = math.exp(-1.0)
    raw_end = math.exp(-math.exp(-gcfg.gpo_growth_rate * total_steps))
    if raw_end - raw_start <= 1e-8:
        return progress

    raw_value = math.exp(-math.exp(-gcfg.gpo_growth_rate * capped_step))
    return clamp_progress((raw_value - raw_start) / (raw_end - raw_start))


def action_growth_scale(
    progress: float,
    cfg: ExperimentConfig,
    global_step: int | float | None = None,
) -> float:
    if not cfg.use_action_growth:
        return 1.0
    gcfg = cfg.growth
    if gcfg.action_schedule == "gpo_paper":
        weight = gpo_growth_weight(progress, cfg, global_step=global_step)
        action_limit = resolved_action_limit(cfg)
        beta_scale = interpolate(gcfg.gpo_beta_start_scale, gcfg.gpo_beta_end_scale, weight)
        return action_limit * beta_scale
    if gcfg.action_schedule == "linear":
        weight = linear_curve(progress)
    elif gcfg.action_schedule == "sigmoid":
        weight = sigmoid_curve(progress, gcfg.sigmoid_midpoint, gcfg.sigmoid_k)
    elif gcfg.action_schedule == "gompertz":
        raw = gompertz_curve(progress, gcfg.gompertz_b, gcfg.gompertz_c)
        min_raw = gompertz_curve(0.0, gcfg.gompertz_b, gcfg.gompertz_c)
        max_raw = gompertz_curve(1.0, gcfg.gompertz_b, gcfg.gompertz_c)
        weight = (raw - min_raw) / max(max_raw - min_raw, 1e-8)
    else:
        raise ValueError(f"Unknown action schedule: {gcfg.action_schedule}")
    return interpolate(gcfg.beta_start, gcfg.beta_end, weight)


def clip_action(action: np.ndarray, action_limit: float) -> np.ndarray:
    action_limit = max(float(action_limit), 0.0)
    return np.clip(np.asarray(action, dtype=np.float32), -action_limit, action_limit).astype(np.float32)


def apply_action_growth(
    action: np.ndarray,
    beta: float,
    action_limit: float,
    eps: float = 1e-6,
) -> np.ndarray:
    safe_beta = max(float(beta), eps)
    clipped = clip_action(action, action_limit)
    ratio_limited = clip_action(clipped, min(float(action_limit), 0.5 * safe_beta))
    transformed = safe_beta * np.tanh(ratio_limited / safe_beta)
    return clip_action(transformed, action_limit)


def capacity_progress(progress: float, cfg: ExperimentConfig) -> float:
    if not cfg.use_capacity_growth:
        return 1.0
    return clamp_progress(progress)


def gating_temperature(progress: float, cfg: ExperimentConfig) -> float:
    if not cfg.use_moe:
        return 1.0
    pc = capacity_progress(progress, cfg)
    weight = sigmoid_curve(pc, cfg.moe.gate_temperature_midpoint, 8.0)
    return interpolate(cfg.moe.init_temperature, cfg.moe.final_temperature, weight)


def topk_schedule(progress: float, cfg: ExperimentConfig) -> int:
    if not cfg.use_moe:
        return 1
    pc = capacity_progress(progress, cfg)
    if pc < cfg.moe.topk_midpoint:
        return cfg.moe.early_topk
    if pc < 0.85:
        return cfg.moe.mid_topk
    return cfg.moe.late_topk


def balance_coef(progress: float, cfg: ExperimentConfig) -> float:
    if not cfg.use_moe:
        return 0.0
    decay = sigmoid_curve(progress, cfg.growth.balance_decay_midpoint, 10.0)
    return interpolate(cfg.growth.lambda_balance_early, cfg.growth.lambda_balance_late, decay)


def gate_entropy_coef(progress: float, cfg: ExperimentConfig) -> float:
    if not cfg.use_moe:
        return 0.0
    ramp = sigmoid_curve(progress, cfg.growth.entropy_ramp_midpoint, 10.0)
    return interpolate(cfg.growth.lambda_gate_entropy_early, cfg.growth.lambda_gate_entropy_late, ramp)


def build_schedule_snapshot(
    progress: float,
    cfg: ExperimentConfig,
    global_step: int | float | None = None,
) -> dict[str, Any]:
    progress = clamp_progress(progress)
    return {
        "progress": progress,
        "beta": action_growth_scale(progress, cfg, global_step=global_step),
        "temperature": gating_temperature(progress, cfg),
        "topk": topk_schedule(progress, cfg),
        "balance_coef": balance_coef(progress, cfg),
        "gate_entropy_coef": gate_entropy_coef(progress, cfg),
    }


def build_schedule_table(cfg: ExperimentConfig, num_points: int = 200) -> dict[str, list[float]]:
    xs = [i / max(num_points - 1, 1) for i in range(num_points)]
    data = {
        "progress": xs,
        "beta": [],
        "temperature": [],
        "topk": [],
        "balance_coef": [],
        "gate_entropy_coef": [],
    }
    old_total_steps = old_total_training_steps(cfg)
    for progress in xs:
        snapshot = build_schedule_snapshot(progress, cfg, global_step=progress * old_total_steps)
        data["beta"].append(snapshot["beta"])
        data["temperature"].append(snapshot["temperature"])
        data["topk"].append(snapshot["topk"])
        data["balance_coef"].append(snapshot["balance_coef"])
        data["gate_entropy_coef"].append(snapshot["gate_entropy_coef"])
    return data
