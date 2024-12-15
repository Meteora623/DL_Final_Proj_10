import torch
from dataset import create_wall_dataloader
from configs import ConfigBase
from evaluator import ProbingEvaluator, ProbingConfig
from models import JEPA_Model


class MainConfig(ConfigBase):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    repr_dim: int = 256
    action_dim: int = 2
    model_weights_path: str = "model_weights.pth"


def load_data(config):
    # Paths to datasets
    train_path = "/scratch/DL24FA/train/"
    normal_val_path = "/scratch/DL24FA/probe_normal/val/"
    wall_val_path = "/scratch/DL24FA/probe_wall/val/"
    expert_val_path = "/scratch/DL24FA/probe_expert/val/"
    wall_other_val_path = "/scratch/DL24FA/probe_wall_other/val/"

    # Train DataLoader
    train_loader = create_wall_dataloader(
        train_path, probing=False, device=config.device, batch_size=config.batch_size, train=True
    )

    # Validation DataLoaders
    val_loaders = {
        "normal": create_wall_dataloader(normal_val_path, probing=True, device=config.device, batch_size=config.batch_size, train=False),
        "wall": create_wall_dataloader(wall_val_path, probing=True, device=config.device, batch_size=config.batch_size, train=False),
        "expert": create_wall_dataloader(expert_val_path, probing=True, device=config.device, batch_size=config.batch_size, train=False),
        "wall_other": create_wall_dataloader(wall_other_val_path, probing=True, device=config.device, batch_size=config.batch_size, train=False),
    }
    return train_loader, val_loaders


def load_model(config):
    """Load the trained JEPA model."""
    model = JEPA_Model(repr_dim=config.repr_dim, action_dim=config.action_dim).to(config.device)
    model.load_state_dict(torch.load(config.model_weights_path, map_location=config.device))
    model.eval()
    return model


def main():
    config = MainConfig()

    # Load datasets
    train_loader, val_loaders = load_data(config)

    # Load model
    model = load_model(config)

    # Initialize evaluator
    evaluator = ProbingEvaluator(
        device=config.device,
        model=model,
        probe_train_ds=train_loader,
        probe_val_ds=val_loaders,
        config=ProbingConfig(),
        quick_debug=False,
    )

    # Train the prober
    prober = evaluator.train_pred_prober()

    # Evaluate on all datasets
    losses = evaluator.evaluate_all(prober)
    for dataset_name, loss in losses.items():
        print(f"Loss on {dataset_name} dataset: {loss:.4f}")


if __name__ == "__main__":
    main()
