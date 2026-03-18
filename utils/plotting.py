from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import config_from_dict
from utils.analysis import REGION_LABELS, load_json, steps_to_threshold
from utils.logger import ensure_dir
from utils.schedules import build_schedule_table


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _series(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        try:
            xs.append(float(row[x_key]))
            ys.append(float(row[y_key]))
        except (KeyError, ValueError):
            continue
    return np.asarray(xs), np.asarray(ys)


def _save_line_plot(
    output_path: Path,
    curves: list[tuple[np.ndarray, np.ndarray, str]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7, 4))
    for xs, ys, label in curves:
        if len(xs) == 0:
            continue
        plt.plot(xs, ys, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_experiment(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    plot_dir = ensure_dir(run_dir / "plots")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as f:
        cfg = config_from_dict(json.load(f))

    acquisition_rows = _read_csv(run_dir / "acquisition" / "metrics.csv")
    maturation_rows = _read_csv(run_dir / "maturation" / "metrics.csv")
    plastic_rows = _read_csv(run_dir / "relearning_plastic" / "metrics.csv")
    mature_rows = _read_csv(run_dir / "relearning_mature" / "metrics.csv")
    generic_relearn_rows = _read_csv(run_dir / "relearning" / "metrics.csv")

    if acquisition_rows or maturation_rows:
        _save_line_plot(
            plot_dir / "reward_curve.png",
            [
                _series(acquisition_rows, "global_env_step", "reward_mean") + ("acquisition",),
                _series(maturation_rows, "global_env_step", "reward_mean") + ("maturation",),
            ],
            "Training Reward",
            "Environment Steps",
            "Episode Reward",
        )
        _save_line_plot(
            plot_dir / "success_curve.png",
            [
                _series(acquisition_rows, "global_env_step", "success_rate") + ("acquisition",),
                _series(maturation_rows, "global_env_step", "success_rate") + ("maturation",),
            ],
            "Success Rate",
            "Environment Steps",
            "Success Rate",
        )
        _save_line_plot(
            plot_dir / "control_cost_curve.png",
            [
                _series(acquisition_rows, "global_env_step", "control_cost_mean") + ("acquisition",),
                _series(maturation_rows, "global_env_step", "control_cost_mean") + ("maturation",),
            ],
            "Control Cost",
            "Environment Steps",
            "Control Cost",
        )
        _save_line_plot(
            plot_dir / "gate_entropy_curve.png",
            [
                _series(acquisition_rows, "global_env_step", "gate_entropy_mean") + ("acquisition",),
                _series(maturation_rows, "global_env_step", "gate_entropy_mean") + ("maturation",),
            ],
            "Gate Entropy",
            "Environment Steps",
            "Gate Entropy",
        )

    summary_path = run_dir / "maturation" / "usage_summary.json"
    if not summary_path.exists():
        summary_path = run_dir / "acquisition" / "usage_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        overall = np.asarray(summary["overall_usage"], dtype=np.float32)
        plt.figure(figsize=(6, 4))
        plt.bar(np.arange(len(overall)), overall)
        plt.title("Expert Usage")
        plt.xlabel("Expert")
        plt.ylabel("Average Gate Weight")
        plt.tight_layout()
        plt.savefig(plot_dir / "expert_usage_bar.png", dpi=160)
        plt.close()

        heatmap = np.asarray(summary["region_usage"], dtype=np.float32)
        plt.figure(figsize=(7, 4))
        plt.imshow(heatmap, aspect="auto", cmap="viridis")
        plt.colorbar(label="Average Gate Weight")
        plt.yticks(np.arange(len(REGION_LABELS)), REGION_LABELS)
        plt.xlabel("Expert")
        plt.title("Region-Expert Usage Heatmap")
        plt.tight_layout()
        plt.savefig(plot_dir / "region_expert_heatmap.png", dpi=160)
        plt.close()

    if plastic_rows or mature_rows or generic_relearn_rows:
        curves: list[tuple[np.ndarray, np.ndarray, str]] = []
        if plastic_rows:
            curves.append(_series(plastic_rows, "phase_step", "success_rate") + ("plastic",))
        if mature_rows:
            curves.append(_series(mature_rows, "phase_step", "success_rate") + ("mature",))
        if generic_relearn_rows:
            curves.append(_series(generic_relearn_rows, "phase_step", "success_rate") + ("relearning",))
        _save_line_plot(
            plot_dir / "relearning_curve.png",
            curves,
            "Relearning on New Tasks",
            "Relearning Steps",
            "Success Rate",
        )

        summary = {}
        if plastic_rows:
            summary["plastic_steps_to_threshold"] = steps_to_threshold(
                plastic_rows, "success_rate", cfg.success_threshold
            )
        if mature_rows:
            summary["mature_steps_to_threshold"] = steps_to_threshold(
                mature_rows, "success_rate", cfg.success_threshold
            )
        with (plot_dir / "relearning_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    table = build_schedule_table(cfg)
    plt.figure(figsize=(8, 5))
    plt.plot(table["progress"], table["beta"], label="beta(t)")
    plt.plot(table["progress"], table["temperature"], label="temperature(t)")
    plt.plot(table["progress"], table["topk"], label="top-k(t)")
    plt.title("Growth Schedules")
    plt.xlabel("Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "schedule_curves.png", dpi=160)
    plt.close()


def plot_reward_comparison(
    run_specs: list[tuple[str, str | Path]],
    output_path: str | Path,
    stage_name: str = "acquisition",
    metric_key: str = "reward_mean",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[tuple[np.ndarray, np.ndarray, str]] = []
    for label, run_dir in run_specs:
        run_dir = Path(run_dir)
        rows = _read_csv(run_dir / stage_name / "metrics.csv")
        if not rows:
            continue
        curves.append(_series(rows, "global_env_step", metric_key) + (label,))
    _save_line_plot(
        output_path=output_path,
        curves=curves,
        title=f"{stage_name.capitalize()} Reward Comparison",
        xlabel="Environment Steps",
        ylabel="Episode Reward",
    )
