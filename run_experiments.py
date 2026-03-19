from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from envs import SUPPORTED_ENVS
from train import run_training


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four core experiments with a timestamped root folder.")
    parser.add_argument("--preset", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--tag", type=str, default="compare4")
    parser.add_argument("--env", type=str, default="multi_region_nav", choices=list(SUPPORTED_ENVS))
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = make_timestamp()
    group_dir = Path(args.output_root) / f"{timestamp}_{args.tag}_seed{args.seed}"
    group_dir.mkdir(parents=True, exist_ok=True)

    full_stage = "acquisition" if args.env == "bipedal_diverse" else "all"
    specs = [
        ("baseline", "acquisition"),
        ("gpo_only", "acquisition"),
        ("moe_only", "acquisition"),
        ("full", full_stage),
    ]
    outputs: dict[str, str] = {}

    for exp_name, stage in specs:
        run_args = SimpleNamespace(
            exp=exp_name,
            preset=args.preset,
            stage=stage,
            resume=None,
            seed=args.seed,
            device=args.device,
            run_dir=str(group_dir / exp_name),
            env=args.env,
        )
        outputs[exp_name] = str(run_training(run_args))

    manifest = {
        "timestamp": timestamp,
        "group_dir": str(group_dir),
        "device": args.device,
        "preset": args.preset,
        "seed": args.seed,
        "env": args.env,
        "runs": outputs,
    }
    with (group_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.skip_summary:
        from summarize_results import summarize_group

        summary = summarize_group(group_dir=group_dir, device=args.device)
        with (group_dir / "summary_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Finished. Group directory: {group_dir}")


if __name__ == "__main__":
    main()
