from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from algos.ppo import PPOTrainer
from config import config_to_dict, make_config
from envs import SUPPORTED_ENVS
from utils.plotting import plot_experiment


def _default_run_dir(exp_name: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{timestamp}_{exp_name}_seed{seed}"


def _save_run_config(run_dir: Path, cfg) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, ensure_ascii=False, indent=2)


def _require_checkpoint(run_dir: Path, explicit_resume: str | None, stage_name: str) -> Path:
    if explicit_resume:
        return Path(explicit_resume)
    default_map = {
        "maturation": run_dir / "acquisition" / "stage_a_end.pt",
        "relearning_plastic": run_dir / "acquisition" / "stage_a_end.pt",
        "relearning_mature": run_dir / "maturation" / "stage_b_mature.pt",
        "relearning": run_dir / "acquisition" / "stage_a_end.pt",
    }
    path = default_map[stage_name]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def run_training(args: argparse.Namespace) -> Path:
    cfg = make_config(args.exp, args.preset, seed=args.seed, env_name=args.env)
    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(args.exp, args.seed)
    _save_run_config(run_dir, cfg)

    if args.stage in {"acquisition", "all"}:
        trainer = PPOTrainer(cfg, run_dir=run_dir, device=args.device)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train_stage(
            stage_name="acquisition",
            env_mode="old",
            total_steps=cfg.stage_steps["acquisition"],
            mature=False,
        )

    if args.stage == "maturation" or (args.stage == "all" and cfg.use_mature):
        trainer = PPOTrainer(cfg, run_dir=run_dir, device=args.device)
        checkpoint_path = _require_checkpoint(run_dir, args.resume, "maturation")
        checkpoint = trainer.load_checkpoint(checkpoint_path, reset_optimizer=True)
        if cfg.use_mature:
            trainer.apply_maturation_freeze(checkpoint.get("usage_summary"))
        trainer.train_stage(
            stage_name="maturation",
            env_mode="old",
            total_steps=cfg.stage_steps["maturation"],
            mature=cfg.use_mature,
        )

    if args.stage in {"relearning", "all"}:
        if cfg.exp_name == "full":
            plastic_trainer = PPOTrainer(cfg, run_dir=run_dir, device=args.device)
            plastic_ckpt = _require_checkpoint(run_dir, args.resume, "relearning_plastic")
            plastic_state = plastic_trainer.load_checkpoint(plastic_ckpt, reset_optimizer=True)
            plastic_trainer.train_stage(
                stage_name="relearning_plastic",
                env_mode="new",
                total_steps=cfg.stage_steps["relearning"],
                mature=False,
                fixed_progress=plastic_state.get("schedule_anchor", 0.0),
            )

            mature_trainer = PPOTrainer(cfg, run_dir=run_dir, device=args.device)
            mature_ckpt = _require_checkpoint(run_dir, None if args.resume else None, "relearning_mature")
            mature_state = mature_trainer.load_checkpoint(mature_ckpt, reset_optimizer=True)
            mature_trainer.train_stage(
                stage_name="relearning_mature",
                env_mode="new",
                total_steps=cfg.stage_steps["relearning"],
                mature=True,
                fixed_progress=mature_state.get("schedule_anchor", 1.0),
            )
        else:
            trainer = PPOTrainer(cfg, run_dir=run_dir, device=args.device)
            checkpoint_path = _require_checkpoint(run_dir, args.resume, "relearning")
            checkpoint = trainer.load_checkpoint(checkpoint_path, reset_optimizer=True)
            trainer.train_stage(
                stage_name="relearning",
                env_mode="new",
                total_steps=cfg.stage_steps["relearning"],
                mature=False,
                fixed_progress=checkpoint.get("schedule_anchor", 0.0),
            )

    plot_experiment(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train growth-style MoE RL prototype.")
    parser.add_argument(
        "--exp",
        type=str,
        default="full",
        choices=["baseline", "gpo_only", "moe_only", "full"],
    )
    parser.add_argument("--preset", type=str, default="quick", choices=["quick", "full"])
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["acquisition", "maturation", "relearning", "all"],
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--env", type=str, default="multi_region_nav", choices=list(SUPPORTED_ENVS))
    return parser.parse_args()


if __name__ == "__main__":
    run_dir = run_training(parse_args())
    print(f"Training finished. Run directory: {run_dir}")
