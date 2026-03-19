from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REGION_LABELS = [
    "normal",
    "slippery",
    "damping",
    "disturbance",
    "slippery_disturbance",
]


def init_usage_stats(
    num_experts: int,
    num_regions: int | None = None,
    region_labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = list(region_labels) if region_labels is not None else list(REGION_LABELS)
    num_regions = num_regions or len(labels)
    return {
        "num_experts": int(num_experts),
        "region_labels": labels[:num_regions],
        "expert_weight_sum": np.zeros(num_experts, dtype=np.float64),
        "region_expert_sum": np.zeros((num_regions, num_experts), dtype=np.float64),
        "region_counts": np.zeros(num_regions, dtype=np.float64),
        "gate_entropy_sum": 0.0,
        "samples": 0,
    }


def update_usage_stats(
    stats: dict[str, Any],
    gate_weights: np.ndarray | None,
    region_ids: np.ndarray | list[int] | None,
    gate_entropy: np.ndarray | float | None,
) -> None:
    if gate_weights is None:
        return
    gate_weights = np.asarray(gate_weights, dtype=np.float64)
    if gate_weights.ndim == 1:
        gate_weights = gate_weights[None, :]
    stats["expert_weight_sum"] += gate_weights.sum(axis=0)
    stats["samples"] += int(gate_weights.shape[0])

    if gate_entropy is not None:
        entropy_array = np.asarray(gate_entropy, dtype=np.float64).reshape(-1)
        stats["gate_entropy_sum"] += float(entropy_array.sum())

    if region_ids is None:
        return
    region_ids = np.asarray(region_ids, dtype=np.int64).reshape(-1)
    for idx, region_id in enumerate(region_ids):
        if region_id < 0 or region_id >= len(stats["region_counts"]):
            continue
        stats["region_counts"][region_id] += 1.0
        stats["region_expert_sum"][region_id] += gate_weights[idx]


def summarize_usage_stats(
    stats: dict[str, Any], low_usage_threshold: float = 0.08
) -> dict[str, Any]:
    region_labels = list(stats.get("region_labels", REGION_LABELS))
    if stats["samples"] <= 0:
        num_experts = int(stats["num_experts"])
        return {
            "overall_usage": [0.0] * num_experts,
            "region_usage": [[0.0] * num_experts for _ in region_labels],
            "region_counts": [0.0] * len(region_labels),
            "region_labels": region_labels,
            "gate_entropy_mean": 0.0,
            "dormant_experts": list(range(num_experts)),
        }

    overall = stats["expert_weight_sum"] / max(stats["samples"], 1)
    region_usage = np.zeros_like(stats["region_expert_sum"])
    for region_idx, count in enumerate(stats["region_counts"]):
        if count > 0:
            region_usage[region_idx] = stats["region_expert_sum"][region_idx] / count

    gate_entropy_mean = stats["gate_entropy_sum"] / max(stats["samples"], 1)
    dormant = [idx for idx, usage in enumerate(overall.tolist()) if usage < low_usage_threshold]
    return {
        "overall_usage": overall.tolist(),
        "region_usage": region_usage.tolist(),
        "region_counts": stats["region_counts"].tolist(),
        "region_labels": region_labels,
        "gate_entropy_mean": float(gate_entropy_mean),
        "dormant_experts": dormant,
    }


def save_usage_summary(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def steps_to_threshold(
    rows: list[dict[str, Any]], metric_key: str, threshold: float
) -> int | None:
    for row in rows:
        try:
            value = float(row[metric_key])
            step = int(float(row["phase_step"]))
        except (KeyError, ValueError):
            continue
        if value >= threshold:
            return step
    return None
