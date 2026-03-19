from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import config_from_dict
from evaluate import evaluate_policy


def _load_config(run_dir: Path):
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def _primary_checkpoint(run_dir: Path, exp_name: str) -> Path:
    if exp_name == "full":
        matured = run_dir / "maturation" / "stage_b_mature.pt"
        if matured.exists():
            return matured
    acquisition = run_dir / "acquisition" / "stage_a_end.pt"
    if acquisition.exists():
        return acquisition
    raise FileNotFoundError(f"No primary checkpoint found under {run_dir}")


def _save_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def postprocess_run(
    run_dir: str | Path,
    exp_name: str,
    device: str = "cpu",
    fps: int = 8,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    final_eval_dir = run_dir / "final_eval"
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "run_dir": str(run_dir),
        "exp_name": exp_name,
        "env_name": cfg.env.env_name,
        "artifacts": {},
    }

    primary_metrics = evaluate_policy(
        checkpoint_path=_primary_checkpoint(run_dir, exp_name),
        env_mode="old",
        num_episodes=cfg.eval_episodes,
        deterministic=True,
        device=device,
        gif_path=final_eval_dir / "final_policy.gif",
        map_path=final_eval_dir / "final_policy.png",
        fps=fps,
    )
    _save_metrics(final_eval_dir / "final_policy.json", primary_metrics)
    summary["artifacts"]["final_policy"] = primary_metrics

    if exp_name == "full":
        plastic_ckpt = run_dir / "relearning_plastic" / "relearning_plastic_end.pt"
        if plastic_ckpt.exists():
            plastic_metrics = evaluate_policy(
                checkpoint_path=plastic_ckpt,
                env_mode="new",
                num_episodes=cfg.eval_episodes,
                deterministic=True,
                device=device,
                gif_path=final_eval_dir / "plastic_new.gif",
                map_path=final_eval_dir / "plastic_new.png",
                fps=fps,
            )
            _save_metrics(final_eval_dir / "plastic_new.json", plastic_metrics)
            summary["artifacts"]["plastic_new"] = plastic_metrics

        mature_ckpt = run_dir / "relearning_mature" / "latest.pt"
        if mature_ckpt.exists():
            mature_metrics = evaluate_policy(
                checkpoint_path=mature_ckpt,
                env_mode="new",
                num_episodes=cfg.eval_episodes,
                deterministic=True,
                device=device,
                gif_path=final_eval_dir / "mature_new.gif",
                map_path=final_eval_dir / "mature_new.png",
                fps=fps,
            )
            _save_metrics(final_eval_dir / "mature_new.json", mature_metrics)
            summary["artifacts"]["mature_new"] = mature_metrics

    _save_metrics(final_eval_dir / "postprocess_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export final evaluation artifacts for a single run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True, choices=["baseline", "gpo_only", "moe_only", "full"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = postprocess_run(
        run_dir=args.run_dir,
        exp_name=args.exp,
        device=args.device,
        fps=args.fps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
