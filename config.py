from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    env_name: str = "multi_region_nav"
    world_min: float = -5.0
    world_max: float = 5.0
    dt: float = 0.1
    max_steps: int = 150
    success_radius: float = 0.3
    action_limit: float = 1.0
    max_speed: float = 3.0
    spawn_margin: float = 0.75
    min_start_goal_distance: float = 2.5
    min_waypoint_distance: float = 1.75
    num_goals_min: int = 3
    num_goals_max: int = 4
    num_goals_old: int = 3
    num_goals_new: int = 4
    max_goals_supported: int = 4
    num_primary_regions: int = 3
    num_disturbance_regions_old: int = 0
    num_disturbance_regions_new: int = 2
    goal_region_margin: float = 0.45
    disturbance_strength_old: float = 0.0
    disturbance_strength_new: float = 0.45
    progress_reward_scale: float = 2.0
    distance_penalty_scale: float = 0.08
    action_penalty_scale: float = 0.04
    smoothness_penalty_scale: float = 0.03
    step_penalty_scale: float = 0.02
    goal_bonus: float = 6.0
    waypoint_bonus: float = 4.0
    success_bonus: float = 15.0
    out_of_bounds_penalty: float = 7.0
    bipedal_section_min_steps: int = 18
    bipedal_section_max_steps: int = 30
    bipedal_old_slope_scale: float = 1.0
    bipedal_new_slope_scale: float = 1.0
    bipedal_slope_angle_min_deg: float = 30.0
    bipedal_slope_angle_max_deg: float = 40.0
    bipedal_old_roughness: float = 1.0
    bipedal_new_roughness: float = 1.4
    bipedal_flat_noise: float = 0.008
    bipedal_height_clip: float = 1.2
    seed: int = 42

    @property
    def obs_dim(self) -> int:
        if self.env_name == "bipedal_diverse":
            return 24
        if self.env_name != "multi_region_nav":
            raise ValueError(f"Unsupported env_name for obs_dim: {self.env_name}")
        base_state = 2 + 2 + 2
        goal_features = self.max_goals_supported * 2
        goal_masks = self.max_goals_supported * 2
        region_features = 4
        scalar_features = 5
        return base_state + goal_features + goal_masks + region_features + scalar_features

    @property
    def action_dim(self) -> int:
        if self.env_name == "bipedal_diverse":
            return 4
        if self.env_name != "multi_region_nav":
            raise ValueError(f"Unsupported env_name for action_dim: {self.env_name}")
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
class NetworkConfig:
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])


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
    network: NetworkConfig = field(default_factory=NetworkConfig)
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


def _apply_bipedal_defaults(cfg: ExperimentConfig, preset: str) -> None:
    cfg.moe.shared_hidden = 256
    cfg.moe.expert_hidden = 128
    cfg.network.critic_hidden_dims = [256, 256, 128]
    cfg.eval_episodes = 8
    cfg.ppo.learning_rate = 2.5e-4
    cfg.ppo.minibatch_size = 512
    cfg.ppo.ent_coef = 0.0 if cfg.exp_name in {"moe_only", "full"} else 5e-4

    if preset == "quick":
        cfg.stage_steps = {
            "acquisition": 300_000,
            "maturation": 80_000,
            "relearning": 120_000,
        }
        cfg.ppo.rollout_steps = 2048
        cfg.ppo.update_epochs = 8
    else:
        cfg.stage_steps = {
            "acquisition": 2_000_000,
            "maturation": 200_000,
            "relearning": 300_000,
        }
        cfg.ppo.rollout_steps = 4096
        cfg.ppo.update_epochs = 10


def make_config(
    exp_name: str,
    preset: str = "quick",
    seed: int = 42,
    env_name: str = "multi_region_nav",
) -> ExperimentConfig:
    cfg = ExperimentConfig(exp_name=exp_name, preset=preset, seed=seed)
    cfg.stage_steps = _make_stage_steps(preset)
    cfg.env.env_name = env_name
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

    if env_name == "bipedal_diverse":
        _apply_bipedal_defaults(cfg, preset)

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
    network = NetworkConfig(**data.get("network", {}))
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
        network=network,
        growth=growth,
        stage_steps=data["stage_steps"],
        success_threshold=data.get("success_threshold", 0.7),
        eval_episodes=data.get("eval_episodes", 32),
    )
