"""Configuration package for Project 9."""

from config.experiment_config import (
    INIT_SEEDS_REAL,
    INIT_SEEDS_SYN,
    SAMPLE_SIZE_GRID_REAL,
    SAMPLE_SIZE_GRID_SYN,
    WEIGHT_DECAY_GRID,
    WIDTH_GRID,
    ModelConfig,
    ProjectPaths,
    RealDataConfig,
    SyntheticDataConfig,
    TrainConfig,
)

__all__ = [
    "ProjectPaths",
    "SyntheticDataConfig",
    "RealDataConfig",
    "ModelConfig",
    "TrainConfig",
    "WIDTH_GRID",
    "SAMPLE_SIZE_GRID_SYN",
    "SAMPLE_SIZE_GRID_REAL",
    "WEIGHT_DECAY_GRID",
    "INIT_SEEDS_SYN",
    "INIT_SEEDS_REAL",
]

