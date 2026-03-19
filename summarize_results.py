from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate_policy
from utils.analysis import load_json, steps_to_threshold
from utils.plotting import plot_metric_comparison, plot_reward_comparison


def _read_metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_group(group_dir: str | Path, device: str = "cuda") -> dict[str, object]:
    group_dir = Path(group_dir)
    summary_dir = group_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {
        "baseline": group_dir / "baseline",
        "gpo_only": group_dir / "gpo_only",
        "moe_only": group_dir / "moe_only",
        "full": group_dir / "full",
    }

    reward_plot = summary_dir / "reward_comparison_acquisition.png"
    plot_reward_comparison(
        [
            ("baseline", run_dirs["baseline"]),
            ("gpo_only", run_dirs["gpo_only"]),
            ("moe_only", run_dirs["moe_only"]),
            ("full", run_dirs["full"]),
        ],
        output_path=reward_plot,
        stage_name="acquisition",
    )
    success_plot = summary_dir / "success_comparison_acquisition.png"
    plot_metric_comparison(
        [
            ("baseline", run_dirs["baseline"]),
            ("gpo_only", run_dirs["gpo_only"]),
            ("moe_only", run_dirs["moe_only"]),
            ("full", run_dirs["full"]),
        ],
        output_path=success_plot,
        stage_name="acquisition",
        metric_key="success_rate",
        title="Acquisition Success Comparison",
        ylabel="Success Rate",
    )

    checkpoints = {
        "baseline": run_dirs["baseline"] / "acquisition" / "stage_a_end.pt",
        "gpo_only": run_dirs["gpo_only"] / "acquisition" / "stage_a_end.pt",
        "moe_only": run_dirs["moe_only"] / "acquisition" / "stage_a_end.pt",
        "full": run_dirs["full"] / "maturation" / "stage_b_mature.pt",
    }
    if not checkpoints["full"].exists():
        checkpoints["full"] = run_dirs["full"] / "acquisition" / "stage_a_end.pt"

    results: dict[str, object] = {}
    for name, checkpoint in checkpoints.items():
        results[name] = evaluate_policy(
            checkpoint_path=checkpoint,
            env_mode="old",
            num_episodes=12,
            deterministic=True,
            device=device,
            gif_path=summary_dir / f"{name}_old.gif",
            map_path=summary_dir / f"{name}_old_map.png",
            fps=8,
        )

    plastic_checkpoint = run_dirs["full"] / "relearning_plastic" / "relearning_plastic_end.pt"
    if plastic_checkpoint.exists():
        results["full_plastic_new"] = evaluate_policy(
            checkpoint_path=plastic_checkpoint,
            env_mode="new",
            num_episodes=12,
            deterministic=True,
            device=device,
            gif_path=summary_dir / "full_plastic_new.gif",
            map_path=summary_dir / "full_plastic_new_map.png",
            fps=8,
        )

    mature_checkpoint = run_dirs["full"] / "relearning_mature" / "latest.pt"
    if mature_checkpoint.exists():
        results["full_mature_new"] = evaluate_policy(
            checkpoint_path=mature_checkpoint,
            env_mode="new",
            num_episodes=12,
            deterministic=True,
            device=device,
            gif_path=summary_dir / "full_mature_new.gif",
            map_path=summary_dir / "full_mature_new_map.png",
            fps=8,
        )

    acquisition_rows = {
        name: _read_metrics(run_dir / "acquisition" / "metrics.csv")
        for name, run_dir in run_dirs.items()
    }
    success_threshold = 0.7
    final_rewards = []
    for rows in acquisition_rows.values():
        if rows:
            try:
                final_rewards.append(float(rows[-1]["reward_mean"]))
            except (KeyError, TypeError, ValueError):
                pass
    reward_threshold = max(final_rewards) * 0.8 if final_rewards else 0.0

    convergence_speed: dict[str, object] = {
        "success_threshold": success_threshold,
        "reward_threshold": reward_threshold,
        "by_method": {},
    }
    for name, rows in acquisition_rows.items():
        if not rows:
            continue
        reward_auc = 0.0
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            try:
                xs.append(float(row["global_env_step"]))
                ys.append(float(row["reward_mean"]))
            except (KeyError, TypeError, ValueError):
                continue
        if len(xs) >= 2:
            import numpy as np

            reward_auc = float(np.trapz(np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)))
        convergence_speed["by_method"][name] = {
            "steps_to_success_threshold": steps_to_threshold(
                rows, "success_rate", success_threshold
            ),
            "steps_to_reward_threshold": steps_to_threshold(
                rows, "reward_mean", reward_threshold
            ),
            "reward_auc": reward_auc,
        }

    final_performance = {
        name: {
            "avg_reward": metrics["avg_reward"],
            "success_rate": metrics["success_rate"],
        }
        for name, metrics in results.items()
        if name in {"baseline", "gpo_only", "moe_only", "full"}
    }

    moe_specialization: dict[str, object] = {}
    for name in ("moe_only", "full"):
        usage_summary_path = run_dirs[name] / "acquisition" / "usage_summary.json"
        heatmap_path = run_dirs[name] / "plots" / "region_expert_heatmap.png"
        if usage_summary_path.exists():
            usage_summary = load_json(usage_summary_path)
            moe_specialization[name] = {
                "region_labels": usage_summary.get("region_labels", []),
                "overall_usage": usage_summary.get("overall_usage", []),
                "region_usage": usage_summary.get("region_usage", []),
                "heatmap_path": str(heatmap_path) if heatmap_path.exists() else None,
            }

    focused_summary = {
        "final_performance": final_performance,
        "convergence_speed": convergence_speed,
        "moe_specialization": moe_specialization,
        "plots": {
            "reward_comparison": str(reward_plot),
            "success_comparison": str(success_plot),
        },
    }
    with (summary_dir / "focused_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(focused_summary, f, ensure_ascii=False, indent=2)

    summary = {
        "group_dir": str(group_dir),
        "reward_plot": str(reward_plot),
        "success_plot": str(success_plot),
        "focused_metrics": focused_summary,
        "results": results,
    }
    with (summary_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a timestamped four-experiment run group.")
    parser.add_argument("--group-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_group(group_dir=args.group_dir, device=args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
