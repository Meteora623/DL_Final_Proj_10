import torch
from dataset import create_wall_dataloader
from models import JEPA_Model
from evaluator import ProbingEvaluator, ProbingConfig


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_weights_path = "model_weights.pth"

    model = JEPA_Model(repr_dim=256, action_dim=2, device=device).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    probe_train_ds = create_wall_dataloader("/scratch/DL24FA/probe_normal/train/", probing=True, device=device)
    probe_val_ds = {
        "normal": create_wall_dataloader("/scratch/DL24FA/probe_normal/val/", probing=True, device=device, train=False),
        "wall": create_wall_dataloader("/scratch/DL24FA/probe_wall/val/", probing=True, device=device, train=False),
    }

    evaluator = ProbingEvaluator(device=device, model=model, probe_train_ds=probe_train_ds, probe_val_ds=probe_val_ds)
    prober = evaluator.train_pred_prober()
    losses = evaluator.evaluate_all(prober)

    for key, value in losses.items():
        print(f"Loss on {key}: {value:.4f}")


if __name__ == "__main__":
    main()
