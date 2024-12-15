# configs.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ProbingConfig:
    epochs: int = 30  # Increased from 20 to allow more training
    prober_arch: str = "1024-512-256-128"  # Expanded architecture for higher capacity
    lr: float = 0.0001  # Reduced learning rate for finer adjustments
    sample_timesteps: Optional[int] = None  # Set to None to use all timesteps
    train_path_expert: str = "/scratch/DL24FA/probe_expert/train"
    val_path_expert: str = "/scratch/DL24FA/probe_expert/val"
    train_path_wall_other: str = "/scratch/DL24FA/probe_wall_other/train"
    val_path_wall_other: str = "/scratch/DL24FA/probe_wall_other/val"
    batch_size: int = 64
    augment: bool = False
    repr_dim: int = 256
    action_dim: int = 2
    model_weights_path: str = "model_weights.pth"
