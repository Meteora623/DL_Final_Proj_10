# main.py

import torch
import logging
from torch.utils.data import DataLoader
from evaluator import ProbingEvaluator
from configs import ProbingConfig
from models import YourJEPAModel  # Replace with your actual JEPA model import
from normalizer import Normalizer

def create_wall_dataloader(data_path, probing, device, batch_size, train, augment):
    # Implement your DataLoader creation logic here
    # This is a placeholder function. Replace with your actual DataLoader setup.
    # Example:
    # dataset = YourCustomDataset(data_path, probing=probing, train=train, augment=augment)
    # return DataLoader(dataset, batch_size=batch_size, shuffle=train)
    pass

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

    # Log the average losses
    for prefix, loss in avg_losses.items():
        logging.info(f"Validation Loss on {prefix}: {loss:.4f}")

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize your JEPA model
    model = YourJEPAModel()  # Replace with your actual JEPA model initialization
    model.to(device)

    # Load model weights if necessary
    # model.load_state_dict(torch.load("path_to_model_weights.pth"))

    # Prepare probe datasets
    probe_train_ds = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/train",
        probing=True,        # Set to True for probing
        device=device,
        batch_size=64,
        train=True,
        augment=False,        # Typically, no augmentation for probing
    )

    # Define validation datasets for 'normal' and 'wall'
    probe_val_ds = {
        "normal": create_wall_dataloader(
            data_path="/scratch/DL24FA/probe_normal/val_normal",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
            augment=False,
        ),
        "wall": create_wall_dataloader(
            data_path="/scratch/DL24FA/probe_normal/val_wall",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
            augment=False,
        ),
    }

    # Evaluate the model
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

if __name__ == "__main__":
    main()
