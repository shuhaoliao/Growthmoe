from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from algos.ppo import PPOTrainer
from config import config_from_dict
from envs import build_env
from utils.analysis import init_usage_stats, summarize_usage_stats, update_usage_stats
from utils.schedules import apply_action_growth, build_schedule_snapshot, clip_action, resolved_action_limit


def _rollout_episode(
    trainer: PPOTrainer,
    env,
    schedule: dict[str, float],
    deterministic: bool,
    capture_frames: bool = False,
    stage_description: str | None = None,
) -> tuple[dict[str, float], list[np.ndarray], dict[str, object]]:
    obs, info = env.reset()
    done = False
    episode_reward = 0.0
    episode_length = 0
    episode_control = 0.0
    trajectory = [env.position.copy()]
    frames: list[np.ndarray] = []
    gate_entropy_values: list[float] = []
    gate_weight_records: list[np.ndarray] = []
    region_id_records: list[int] = []

    while not done:
        if capture_frames:
            overlay = [
                f"mode={env.mode}",
                f"region={info['region_name']}",
                f"step={episode_length}",
                f"visited={info['goals_visited']}/{info['num_goals']}",
                f"coverage={info['coverage_ratio']:.2f}",
            ]
            if stage_description:
                overlay.append(stage_description)
            frames.append(
                env.render_rgb_array(
                    trajectory=trajectory,
                    title=f"{env.mode} task rollout",
                    overlay_lines=overlay,
                )
            )

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=trainer.device).unsqueeze(0)
        obs_tensor = trainer._normalize_obs(obs_tensor)
        action_tensor, _, extras = trainer.actor.act(
            obs_tensor,
            deterministic=deterministic,
            temperature=schedule["temperature"],
            topk=schedule["topk"],
        )
        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        action_limit = resolved_action_limit(trainer.cfg)
        bounded_action = clip_action(action, action_limit)
        exec_action = (
            apply_action_growth(
                bounded_action,
                beta=schedule["beta"],
                action_limit=action_limit,
            )
            if trainer.cfg.use_action_growth
            else bounded_action
        )

        obs, reward, terminated, truncated, info = env.step(exec_action)
        done = terminated or truncated
        trajectory.append(env.position.copy())
        if trainer.cfg.use_moe and extras["gate_entropy"] is not None:
            gate_entropy_values.append(float(extras["gate_entropy"].item()))
            gate_weight_records.append(extras["gate_weights"].squeeze(0).cpu().numpy())
            region_id_records.append(int(info["region_id"]))

        episode_reward += float(reward)
        episode_length += 1
        episode_control += float(info["control_cost"])

    if capture_frames:
        frames.append(
            env.render_rgb_array(
                trajectory=trajectory,
                title=f"{env.mode} task rollout",
                overlay_lines=[
                    f"done=1 success={int(info['success'])}",
                    f"steps={episode_length}",
                    f"visited={info['goals_visited']}/{info['num_goals']}",
                    f"coverage={info['coverage_ratio']:.2f}",
                    f"reward={episode_reward:.2f}",
                    f"control={episode_control:.2f}",
                ],
            )
        )

    summary = {
        "reward": episode_reward,
        "length": episode_length,
        "control_cost": episode_control,
        "success": float(info["success"]),
        "coverage": float(info["coverage_ratio"]),
        "goals_visited": float(info["goals_visited"]),
        "path_length": float(info["path_length"]),
        "gate_entropy": float(np.mean(gate_entropy_values)) if gate_entropy_values else 0.0,
    }
    episode_artifacts = {
        "trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
        "layout": env.get_layout_summary(),
        "final_info": info,
        "gate_weights": [record.tolist() for record in gate_weight_records],
        "region_ids": region_id_records,
        "gate_entropy_values": gate_entropy_values,
    }
    return summary, frames, episode_artifacts


def save_gif(frames: list[np.ndarray], gif_path: str | Path, fps: int = 8) -> None:
    if not frames:
        return
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / max(fps, 1))
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )


@torch.no_grad()
def evaluate_policy(
    checkpoint_path: str | Path,
    env_mode: str,
    num_episodes: int,
    deterministic: bool = True,
    device: str = "cpu",
    gif_path: str | Path | None = None,
    map_path: str | Path | None = None,
    fps: int = 8,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_dict(checkpoint["cfg"])
    stage_name = checkpoint.get("stage_name", "unknown")

    trainer = PPOTrainer(cfg, run_dir=Path(checkpoint_path).parent.parent, device=device)
    trainer.load_checkpoint(checkpoint_path, reset_optimizer=True)

    env = build_env(cfg.env, mode=env_mode)
    env.reset(seed=cfg.seed + 999)

    reward_list: list[float] = []
    length_list: list[int] = []
    control_cost_list: list[float] = []
    success_list: list[float] = []
    coverage_list: list[float] = []
    goals_visited_list: list[float] = []
    path_length_list: list[float] = []
    usage_stats = (
        init_usage_stats(
            cfg.moe.num_experts,
            region_labels=list(getattr(env, "region_labels", [])),
        )
        if cfg.use_moe
        else None
    )
    completed = 0
    fixed_progress = float(checkpoint.get("schedule_anchor", 1.0))
    schedule = build_schedule_snapshot(
        fixed_progress,
        cfg,
        global_step=checkpoint.get("global_env_step", 0),
    )
    stage_description = env.describe_stage(stage_name)
    captured_frames: list[np.ndarray] = []
    episode_artifacts: dict[str, object] | None = None

    while completed < num_episodes:
        summary, frames, artifacts = _rollout_episode(
            trainer=trainer,
            env=env,
            schedule=schedule,
            deterministic=deterministic,
            capture_frames=(gif_path is not None and completed == 0),
            stage_description=stage_description,
        )
        reward_list.append(summary["reward"])
        length_list.append(summary["length"])
        control_cost_list.append(summary["control_cost"])
        success_list.append(summary["success"])
        coverage_list.append(summary["coverage"])
        goals_visited_list.append(summary["goals_visited"])
        path_length_list.append(summary["path_length"])
        if cfg.use_moe and usage_stats is not None:
            gate_records = np.asarray(artifacts["gate_weights"], dtype=np.float32)
            entropy_records = np.asarray(artifacts["gate_entropy_values"], dtype=np.float32)
            region_ids = np.asarray(artifacts["region_ids"], dtype=np.int64)
            if len(gate_records) > 0:
                update_usage_stats(
                    usage_stats,
                    gate_weights=gate_records,
                    region_ids=region_ids,
                    gate_entropy=entropy_records,
                )
        if frames and not captured_frames:
            captured_frames = frames
            episode_artifacts = artifacts
        completed += 1

    usage_summary = (
        summarize_usage_stats(usage_stats, cfg.growth.low_usage_threshold)
        if usage_stats is not None
        else None
    )
    result = {
        "success_rate": float(np.mean(success_list)),
        "avg_episode_length": float(np.mean(length_list)),
        "avg_reward": float(np.mean(reward_list)),
        "avg_control_cost": float(np.mean(control_cost_list)),
        "avg_coverage": float(np.mean(coverage_list)),
        "avg_goals_visited": float(np.mean(goals_visited_list)),
        "avg_path_length": float(np.mean(path_length_list)),
        "gate_entropy": float(usage_summary["gate_entropy_mean"]) if usage_summary else 0.0,
        "task_mode": env_mode,
        "task_description": env.describe_mode(),
        "stage_name": stage_name,
        "stage_description": stage_description,
    }
    if usage_summary is not None:
        result["overall_usage"] = usage_summary["overall_usage"]
        result["region_usage"] = usage_summary["region_usage"]
    if gif_path is not None and captured_frames:
        save_gif(captured_frames, gif_path=gif_path, fps=fps)
        result["gif_path"] = str(Path(gif_path))
    if map_path is not None and episode_artifacts is not None:
        map_path = Path(map_path)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_frame = captured_frames[0] if captured_frames else None
        if map_frame is not None:
            Image.fromarray(map_frame).save(map_path)
        result["map_path"] = str(map_path)
        with map_path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(episode_artifacts, f, ensure_ascii=False, indent=2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved policy checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, default="old", choices=["old", "new"])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--gif-path", type=str, default=None)
    parser.add_argument("--map-path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate_policy(
        checkpoint_path=args.checkpoint,
        env_mode=args.mode,
        num_episodes=args.episodes,
        deterministic=not args.stochastic,
        device=args.device,
        gif_path=args.gif_path,
        map_path=args.map_path,
        fps=args.fps,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
