# main.py

import os
import torch
from models import JEPA_Model
from dataset import create_wall_dataloader
from normalizer import Normalizer
from evaluator import ProbingEvaluator
from configs import ProbingConfig


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize your JEPA model
    model = JEPA_Model(repr_dim=256, action_dim=2, device=device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.to(device)

    # Prepare probe datasets
    probe_train_ds = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/train",
        probing=True,  # Assuming probing=True for prober training
        device=device,
        batch_size=64,
        train=True,
        augment=False,  # Typically, no augmentation for probing
    )

    probe_val_ds = {
        "val1": create_wall_dataloader(
            data_path="/scratch/DL24FA/probe_normal/val1",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
            augment=False,
        ),
        "val2": create_wall_dataloader(
            data_path="/scratch/DL24FA/probe_normal/val2",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
            augment=False,
        ),
        # Add more validation datasets as needed
    }

    # Evaluate the model
    evaluate_model(device, model, probe_train_ds, probe_val_ds)


if __name__ == "__main__":
    main()
