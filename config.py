from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    world_min: float = -5.0
    world_max: float = 5.0
    dt: float = 0.1
    max_steps: int = 150
    success_radius: float = 0.3
    action_limit: float = 1.0
    max_speed: float = 3.0
    spawn_margin: float = 0.75
    min_start_goal_distance: float = 5.0
    min_waypoint_distance: float = 2.5
    num_goals_min: int = 2
    num_goals_max: int = 4
    num_primary_regions: int = 3
    num_disturbance_regions_old: int = 0
    num_disturbance_regions_new: int = 2
    disturbance_strength_old: float = 0.0
    disturbance_strength_new: float = 0.45
    progress_reward_scale: float = 2.0
    distance_penalty_scale: float = 0.08
    action_penalty_scale: float = 0.04
    smoothness_penalty_scale: float = 0.03
    waypoint_bonus: float = 4.0
    success_bonus: float = 15.0
    out_of_bounds_penalty: float = 7.0
    seed: int = 42

    @property
    def obs_dim(self) -> int:
        return 17

    @property
    def action_dim(self) -> int:
        return 2


@dataclass
class PPOConfig:
    rollout_steps: int = 1024
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    normalize_observations: bool = True


@dataclass
class MoEConfig:
    num_experts: int = 4
    shared_hidden: int = 128
    expert_hidden: int = 64
    init_temperature: float = 2.0
    final_temperature: float = 0.2
    early_topk: int = 4
    mid_topk: int = 2
    late_topk: int = 1
    gate_temperature_midpoint: float = 0.55
    topk_midpoint: float = 0.6
    log_std_init: float = -0.6


@dataclass
class GrowthConfig:
    action_schedule: str = "gpo_paper"
    beta_start: float = 0.2
    beta_end: float = 1.0
    gompertz_b: float = 5.0
    gompertz_c: float = 7.0
    sigmoid_k: float = 10.0
    sigmoid_midpoint: float = 0.45
    gpo_growth_rate: float = 3e-5
    action_limit: float = 1.0
    gpo_beta_start_scale: float = 0.2
    gpo_beta_end_scale: float = 8.0
    lambda_balance_early: float = 1e-2
    lambda_balance_late: float = 1e-4
    lambda_gate_entropy_early: float = 1e-4
    lambda_gate_entropy_late: float = 1e-2
    balance_decay_midpoint: float = 0.6
    entropy_ramp_midpoint: float = 0.6
    freeze_shared_first_layer: bool = True
    freeze_low_usage_experts: bool = True
    low_usage_threshold: float = 0.08


@dataclass
class ExperimentConfig:
    exp_name: str = "full"
    preset: str = "quick"
    seed: int = 42
    use_moe: bool = True
    use_action_growth: bool = True
    use_capacity_growth: bool = True
    use_mature: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    stage_steps: dict[str, int] = field(
        default_factory=lambda: {
            "acquisition": 80_000,
            "maturation": 20_000,
            "relearning": 40_000,
        }
    )
    success_threshold: float = 0.7
    eval_episodes: int = 32


def _make_stage_steps(preset: str) -> dict[str, int]:
    if preset == "full":
        return {
            "acquisition": 200_000,
            "maturation": 50_000,
            "relearning": 100_000,
        }
    return {
        "acquisition": 80_000,
        "maturation": 20_000,
        "relearning": 40_000,
    }


def make_config(exp_name: str, preset: str = "quick", seed: int = 42) -> ExperimentConfig:
    cfg = ExperimentConfig(exp_name=exp_name, preset=preset, seed=seed)
    cfg.stage_steps = _make_stage_steps(preset)
    cfg.env.seed = seed
    cfg.growth.action_limit = cfg.env.action_limit

    if exp_name == "baseline":
        cfg.use_moe = False
        cfg.use_action_growth = False
        cfg.use_capacity_growth = False
        cfg.use_mature = False
        cfg.ppo.ent_coef = 1e-3
    elif exp_name == "gpo_only":
        cfg.use_moe = False
        cfg.use_action_growth = True
        cfg.use_capacity_growth = False
        cfg.use_mature = False
    elif exp_name == "moe_only":
        cfg.use_moe = True
        cfg.use_action_growth = False
        cfg.use_capacity_growth = True
        cfg.use_mature = False
        cfg.ppo.ent_coef = 0.0
    elif exp_name == "full":
        cfg.use_moe = True
        cfg.use_action_growth = True
        cfg.use_capacity_growth = True
        cfg.use_mature = True
        cfg.ppo.ent_coef = 0.0
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")

    if preset == "quick":
        cfg.ppo.rollout_steps = 1024
        cfg.ppo.update_epochs = 6
    elif preset == "full":
        cfg.ppo.rollout_steps = 2048
        cfg.ppo.update_epochs = 10
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return cfg


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    growth_data = dict(data["growth"])
    env_data = dict(data["env"])
    action_limit = env_data.get("action_limit", growth_data.get("action_limit", 1.0))
    env_data["action_limit"] = action_limit
    growth_data["action_limit"] = action_limit
    env = EnvConfig(**env_data)
    ppo = PPOConfig(**data["ppo"])
    moe = MoEConfig(**data["moe"])
    growth = GrowthConfig(**growth_data)
    return ExperimentConfig(
        exp_name=data["exp_name"],
        preset=data["preset"],
        seed=data["seed"],
        use_moe=data["use_moe"],
        use_action_growth=data["use_action_growth"],
        use_capacity_growth=data["use_capacity_growth"],
        use_mature=data["use_mature"],
        env=env,
        ppo=ppo,
        moe=moe,
        growth=growth,
        stage_steps=data["stage_steps"],
        success_threshold=data.get("success_threshold", 0.7),
        eval_episodes=data.get("eval_episodes", 32),
    )
