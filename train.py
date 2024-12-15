import torch
from models import JEPAModel, JEPATrainer
from dataset import create_wall_dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # Load training data
    data_path = "/scratch/DL24FA/train"
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device=device,
        batch_size=64,
        train=True,
    )

    model = JEPAModel(repr_dim=256, momentum=0.99).to(device)
    trainer = JEPATrainer(model, device=device, lr=1e-3, momentum=0.99)

    epochs = 10  # Adjust as needed
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for batch in train_loader:
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, 2]
            loss = trainer.train_step(states, actions)
            total_loss += loss
            count += 1
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # Save trained model
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved as model_weights.pth")
