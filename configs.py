# configs.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class LRSchedule(Enum):
    Constant = auto()
    Cosine = auto()

@dataclass
class ConfigBase:
    # Existing configurations for training can be added here
    # For example:
    # some_param: float = 0.1
    pass

@dataclass
class TrainingConfig(ConfigBase):
    device: str = "cuda"  # Device to use: 'cuda' or 'cpu'
    data_path: str = "/scratch/DL24FA"  # Path to the data directory
    train_split: str = "probe_normal/train"  # Training data split
    val_split: str = "probe_normal/val"  # Validation data split
    batch_size: int = 64  # Batch size
    epochs: int = 2  # Number of training epochs (set to 2 for quick debugging)
    learning_rate: float = 1e-3  # Learning rate for the optimizer
    weight_decay: float = 1e-5  # Weight decay for the optimizer
    scheduler: LRSchedule = LRSchedule.Cosine  # Learning rate scheduler type
    momentum: float = 0.99  # Momentum for updating the target encoder
    repr_dim: int = 256  # Representation dimension
    action_dim: int = 2  # Action dimension
    model_weights_path: str = "model_weights.pth"  # Path to save the trained model
    augment: bool = False  # Whether to apply data augmentation

@dataclass
class ProbingConfig(ConfigBase):
    epochs: int = 10  # Number of probing epochs
    prober_arch: str = "512-256-128"  # Architecture of the prober (e.g., "512-256-128")
    lr: float = 1e-3  # Learning rate for the prober optimizer
    sample_timesteps: Optional[int] = None  # Number of timesteps to sample (set to None to use all)
