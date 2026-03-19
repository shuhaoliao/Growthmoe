from __future__ import annotations

from typing import Any

from config import EnvConfig
from envs.diverse_bipedal_walker_env import DiverseBipedalWalkerEnv
from envs.multi_region_nav_env import MultiRegionNavEnv


SUPPORTED_ENVS = ("multi_region_nav", "bipedal_diverse")


def build_env(config: EnvConfig, mode: str = "old") -> Any:
    if config.env_name == "multi_region_nav":
        return MultiRegionNavEnv(mode=mode, config=config)
    if config.env_name == "bipedal_diverse":
        return DiverseBipedalWalkerEnv(mode=mode, config=config)
    raise ValueError(
        f"Unsupported env_name: {config.env_name}. Expected one of {SUPPORTED_ENVS}."
    )
