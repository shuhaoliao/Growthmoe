from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate_policy
from utils.plotting import plot_reward_comparison


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

    checkpoints = {
        "baseline": run_dirs["baseline"] / "acquisition" / "stage_a_end.pt",
        "gpo_only": run_dirs["gpo_only"] / "acquisition" / "stage_a_end.pt",
        "moe_only": run_dirs["moe_only"] / "acquisition" / "stage_a_end.pt",
        "full": run_dirs["full"] / "maturation" / "stage_b_mature.pt",
    }

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

    results["full_plastic_new"] = evaluate_policy(
        checkpoint_path=run_dirs["full"] / "relearning_plastic" / "relearning_plastic_end.pt",
        env_mode="new",
        num_episodes=12,
        deterministic=True,
        device=device,
        gif_path=summary_dir / "full_plastic_new.gif",
        map_path=summary_dir / "full_plastic_new_map.png",
        fps=8,
    )
    results["full_mature_new"] = evaluate_policy(
        checkpoint_path=run_dirs["full"] / "relearning_mature" / "latest.pt",
        env_mode="new",
        num_episodes=12,
        deterministic=True,
        device=device,
        gif_path=summary_dir / "full_mature_new.gif",
        map_path=summary_dir / "full_mature_new_map.png",
        fps=8,
    )

    summary = {
        "group_dir": str(group_dir),
        "reward_plot": str(reward_plot),
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
