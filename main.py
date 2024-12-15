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
    train_path = "/scratch/DL24FA/probe_normal/train/"
    normal_val_path = "/scratch/DL24FA/probe_normal/val/"
    wall_val_path = "/scratch/DL24FA/probe_wall/val/"

    train_ds = create_wall_dataloader(train_path, probing=True, device=config.device, batch_size=config.batch_size)
    normal_val_ds = create_wall_dataloader(normal_val_path, probing=True, device=config.device, batch_size=config.batch_size)
    wall_val_ds = create_wall_dataloader(wall_val_path, probing=True, device=config.device, batch_size=config.batch_size)

    val_ds = {"normal": normal_val_ds, "wall": wall_val_ds}
    return train_ds, val_ds


def load_model(config):
    model = JEPA_Model(repr_dim=config.repr_dim, action_dim=config.action_dim).to(config.device)
    model.load_state_dict(torch.load(config.model_weights_path, map_location=config.device))
    model.eval()
    return model


def main():
    config = MainConfig()
    train_ds, val_ds = load_data(config)
    model = load_model(config)

    evaluator = ProbingEvaluator(
        device=config.device,
        model=model,
        probe_train_ds=train_ds,
        probe_val_ds=val_ds,
        config=ProbingConfig(),
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    losses = evaluator.evaluate_all(prober)
    for name, loss in losses.items():
        print(f"Loss on {name}: {loss:.4f}")


if __name__ == "__main__":
    main()
