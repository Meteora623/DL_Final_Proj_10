# main.py

import os
import torch
import warnings
from models import JEPA_Model
from dataset import create_wall_dataloader
from normalizer import Normalizer
from evaluator import ProbingEvaluator
from configs import ProbingConfig

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    """
    Evaluates the model using the ProbingEvaluator.

    Args:
        device (torch.device): The device to run the evaluation on.
        model (torch.nn.Module): The trained JEPA model.
        probe_train_ds (DataLoader): DataLoader for the probing training dataset.
        probe_val_ds (dict): Dictionary of DataLoaders for the probing validation datasets.
    """
    config = ProbingConfig()
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        config=config,
        quick_debug=False,
    )

    # Train the prober
    prober = evaluator.train_pred_prober()

    # Evaluate on validation datasets
    avg_losses = evaluator.evaluate_all(prober)

    # Print or log the average losses
    for prefix, loss in avg_losses.items():
        print(f"Validation Loss on {prefix}: {loss:.4f}")

def main():
    """
    The main function to load the model, prepare datasets, and evaluate the model.
    """
    # Determine the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize your JEPA model
    model = JEPA_Model(repr_dim=256, action_dim=2, device=device)

    # Address the FutureWarning from torch.load
    torch_version = torch.__version__
    major, minor = map(int, torch_version.split('.')[:2])
    
    if (major, minor) >= (2, 0):
        # If PyTorch version is 2.0 or higher, use weights_only=True
        model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    else:
        # If not, suppress the FutureWarning (not recommended for long-term)
        warnings.filterwarnings("ignore", category=FutureWarning)
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    
    model.to(device)

    # Prepare probe datasets
    probe_train_ds = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/train",
        probing=True,        # Set to True for probing
        device=device,
        batch_size=64,
        train=True,
        augment=False,        # Typically, no augmentation for probing
    )

    # Corrected Validation Data Path: Changed from 'val1' to 'val'
    probe_val_ds = {
        "val": create_wall_dataloader(
            data_path="/scratch/DL24FA/probe_normal/val",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
            augment=False,
        ),
        # Add more validation datasets here if needed
    }

    # Evaluate the model
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

if __name__ == "__main__":
    main()
