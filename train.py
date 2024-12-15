import torch
from dataset import create_wall_dataloader
from models import JEPA_Model
from schedulers import Scheduler, LRSchedule
from tqdm.auto import tqdm
import torch.nn.functional as F


class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_path: str = "/scratch/DL24FA/train/"
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    repr_dim: int = 256
    action_dim: int = 2
    model_weights_path: str = "model_weights.pth"
    scheduler: LRSchedule = LRSchedule.Cosine
    augment: bool = True


def train_epoch(model, train_loader, optimizer, scheduler, config):
    model.train()
    total_loss = 0
    max_steps = len(train_loader) * config.epochs
    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        states = batch.states
        actions = batch.actions

        optimizer.zero_grad()
        pred_encs = model(states, actions)
        target_encs = model.target_encoder(states[:, 1:])
        loss = F.mse_loss(pred_encs[1:], target_encs)
        loss.backward()
        optimizer.step()

        scheduler.adjust_learning_rate(step, max_steps)

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            states = batch.states
            actions = batch.actions
            pred_encs = model(states, actions)
            target_encs = model.target_encoder(states[:, 1:])
            loss = F.mse_loss(pred_encs[1:], target_encs)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    config = TrainingConfig()
    train_loader = create_wall_dataloader(config.data_path, augment=config.augment, batch_size=config.batch_size)
    val_loader = create_wall_dataloader(config.data_path, augment=False, train=False, batch_size=config.batch_size)

    model = JEPA_Model(repr_dim=config.repr_dim, action_dim=config.action_dim, device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = Scheduler(LRSchedule.Cosine, config.learning_rate, config.batch_size, config.epochs, optimizer)

    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config)
        val_loss = validate(model, val_loader, config)
        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), config.model_weights_path)
            best_val_loss = val_loss
            print("Model saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()
