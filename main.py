# main.py

import torch
import logging
from torch.utils.data import DataLoader
from evaluator import ProbingEvaluator
from configs import ProbingConfig
from models import JEPAModel  # Ensure JEPAModel is correctly imported
from normalizer import Normalizer

def create_dataloader(data_path, probing, device, batch_size, train, augment):
    """
    Creates a DataLoader for the given dataset.

    Args:
        data_path (str): Path to the dataset directory.
        probing (bool): Whether to use probing mode.
        device (torch.device): Device to load the data on.
        batch_size (int): Batch size.
        train (bool): Whether to create a training DataLoader.
        augment (bool): Whether to apply data augmentation.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    # Implement your actual dataset loading logic here.
    # The following is a placeholder using random data.

    from torch.utils.data import Dataset, TensorDataset

    class CustomDataset(Dataset):
        def __init__(self, data_path, probing, train, augment):
            """
            Initialize your dataset.

            Args:
                data_path (str): Path to the dataset.
                probing (bool): Whether to use probing mode.
                train (bool): Whether it's training data.
                augment (bool): Whether to apply augmentation.
            """
            # Replace the following with actual data loading logic
            # For example, loading from .npy files or any other format
            # Example:
            # self.states = torch.load(os.path.join(data_path, "states.pt"))
            # self.actions = torch.load(os.path.join(data_path, "actions.pt"))
            # self.locations = torch.load(os.path.join(data_path, "locations.pt"))

            # Placeholder: Generate random data
            B = 1000 if train else 200  # Number of samples
            T = 17  # Number of timesteps
            C = 3  # Number of channels
            H = 224  # Height
            W = 224  # Width
            A = 2  # Action dimensions

            self.states = torch.randn(B, T, C, H, W)
            self.actions = torch.randn(B, T, A)
            self.locations = torch.randn(B, T, 2)

            # Apply augmentation if needed
            if augment and train:
                # Implement your augmentation logic here
                pass

        def __len__(self):
            return self.states.shape[0]

        def __getitem__(self, idx):
            return self.states[idx], self.actions[idx], self.locations[idx]

    dataset = CustomDataset(data_path, probing, train, augment)
    shuffle = train
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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
    """
    Main function to train and evaluate the Prober.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize your JEPA model
    model = JEPAModel()  # Replace with your actual JEPA model initialization if different
    model.to(device)

    # Load model weights if necessary
    # Example:
    # model.load_state_dict(torch.load(config.model_weights_path))
    # model.eval()

    # Prepare probe training dataset
    probe_train_ds = create_dataloader(
        data_path="/scratch/DL24FA/probe_expert/train",
        probing=True,        # Set to True for probing
        device=device,
        batch_size=64,
        train=True,
        augment=False,        # Typically, no augmentation for probing
    )

    # Define validation datasets for 'expert' and 'wall_other'
    probe_val_ds = {
        "expert": create_dataloader(
            data_path="/scratch/DL24FA/probe_expert/val",
            probing=True,
            device=device,
            batch_size=64,
            train=False,
            augment=False,
        ),
        "wall_other": create_dataloader(
            data_path="/scratch/DL24FA/probe_wall_other/val",
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
