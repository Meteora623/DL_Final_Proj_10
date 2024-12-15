# train.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from models import JEPA_Model
from dataset import create_wall_dataloader
from normalizer import Normalizer
from schedulers import Scheduler, LRSchedule
from configs import ConfigBase
from dataclasses import dataclass


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


def get_device(device_str: str):
    """Get the computation device."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(config: TrainingConfig):
    """Load training and validation data loaders."""
    # Training DataLoader
    train_loader = create_wall_dataloader(
        data_path=os.path.join(config.data_path, config.train_split),
        probing=False,  # Set to False as we're training the model, not probing
        device=config.device,
        batch_size=config.batch_size,
        train=True,
        augment=config.augment,  # Apply augmentation if specified
    )

    # Validation DataLoader
    val_loader = create_wall_dataloader(
        data_path=os.path.join(config.data_path, config.val_split),
        probing=False,  # Set to False as we're validating the model, not probing
        device=config.device,
        batch_size=config.batch_size,
        train=False,
        augment=False,  # Typically, no augmentation during validation
    )

    return train_loader, val_loader


def initialize_model(config: TrainingConfig):
    """Initialize the JEPA model."""
    model = JEPA_Model(
        repr_dim=config.repr_dim,
        action_dim=config.action_dim,
        device=config.device
    ).to(config.device)
    return model


def initialize_optimizer_scheduler(model: torch.nn.Module, config: TrainingConfig, train_loader: DataLoader):
    """Initialize the optimizer and learning rate scheduler."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = Scheduler(
        schedule='Cosine',  # Pass schedule as a string
        base_lr=config.learning_rate,
        data_loader=train_loader,
        epochs=config.epochs,
        optimizer=optimizer,
        batch_steps=len(train_loader),
        batch_size=config.batch_size,
    )

    return optimizer, scheduler


def train_epoch(model: JEPA_Model, train_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: Scheduler, config: TrainingConfig, normalizer: Normalizer):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for batch in progress_bar:
        states = batch.states  # [B, T, C, H, W]
        actions = batch.actions  # [B, T-1, 2]
        # Note: In training, probing=False, so batch.locations is None

        # Move data to device
        states = states.to(config.device)
        actions = actions.to(config.device)

        # Forward pass through the model
        pred_encs = model(states=states, actions=actions)  # [T, B, D]

        # Target encoder forward pass
        with torch.no_grad():
            # Extract future states: [B, T-1, C, H, W]
            future_states = states[:, 1:]  # [B, T-1, C, H, W]
            B, T_minus1, C, H, W = future_states.shape
            # Reshape to [B*(T-1), C, H, W]
            future_states_reshaped = future_states.reshape(-1, C, H, W)
            # Pass through target encoder
            target_encs = model.target_encoder(future_states_reshaped.to(config.device))  # [B*(T-1), D]

        # Reshape predicted embeddings to align with target embeddings
        pred_encs = pred_encs[1:].reshape(-1, config.repr_dim)  # [B*(T-1), D]

        # Compute loss (MSE between predicted and target embeddings)
        loss = F.mse_loss(pred_encs, target_encs)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        scheduler.adjust_learning_rate(step_increment=1)

        # Update target encoder
        model.update_target_encoder(momentum=config.momentum)

        # Accumulate loss
        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({"Loss": loss.item()})

    average_loss = epoch_loss / len(train_loader)
    return average_loss


def validate(model: JEPA_Model, val_loader: DataLoader, config: TrainingConfig, normalizer: Normalizer):
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, 2]

            # Move data to device
            states = states.to(config.device)
            actions = actions.to(config.device)

            # Forward pass through the model
            pred_encs = model(states=states, actions=actions)  # [T, B, D]

            # Target encoder forward pass
            future_states = states[:, 1:]  # [B, T-1, C, H, W]
            B, T_minus1, C, H, W = future_states.shape
            # Reshape to [B*(T-1), C, H, W]
            future_states_reshaped = future_states.reshape(-1, C, H, W)
            # Pass through target encoder
            target_encs = model.target_encoder(future_states_reshaped.to(config.device))  # [B*(T-1), D]

            # Reshape predicted embeddings to align with target embeddings
            pred_encs = pred_encs[1:].reshape(-1, config.repr_dim)  # [B*(T-1), D]

            # Compute loss (MSE between predicted and target embeddings)
            loss = F.mse_loss(pred_encs, target_encs)

            # Accumulate loss
            val_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"Validation Loss": loss.item()})

    average_val_loss = val_loss / len(val_loader)
    return average_val_loss


def save_model(model: JEPA_Model, config: TrainingConfig, epoch: int, val_loss: float):
    """Save the model's state dictionary."""
    save_path = config.model_weights_path
    torch.save(model.state_dict(), save_path)
    print(f"Saved model at epoch {epoch} with validation loss {val_loss:.4f} to {save_path}")


def main():
    # Initialize configuration
    config = TrainingConfig()

    # Get device
    device = get_device(config.device)
    config.device = device  # Update device in config in case of CPU fallback

    # Load data
    train_loader, val_loader = load_data(config)

    # Initialize Normalizer (using predefined mean and std)
    normalizer = Normalizer()  # Initialized without arguments

    # Initialize model
    model = initialize_model(config)

    # Initialize optimizer and scheduler
    optimizer, scheduler = initialize_optimizer_scheduler(model, config, train_loader)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}")

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, normalizer)
        print(f"Training Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, config, normalizer)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config, epoch, val_loss)
            print(f"New best validation loss: {best_val_loss:.4f}. Model saved.")
        else:
            print(f"No improvement in validation loss.")

    print("Training complete.")


if __name__ == "__main__":
    main()
