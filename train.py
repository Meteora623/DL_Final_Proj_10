import torch
from models import JEPAModel, JEPATrainer
from dataset import create_wall_dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    data_path = "/scratch/DL24FA/train"
    print("Loading training data...")
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device="cpu",
        batch_size=128,
        train=True,
    )
    print("Data loader created, total batches:", len(train_loader))
    ds = train_loader.dataset
    print("Dataset length:", len(ds))

    print("Creating model...")
    model = JEPAModel(repr_dim=256, momentum=0.99).to(device)
    print("Model created.")

    # 加上VICReg正则项
    trainer = JEPATrainer(model, device=device, lr=1e-3, momentum=0.99, vicreg_lambda=0.01, vicreg_mu=0.01)
    print("Trainer created.")

    epochs = 1
    print("Start training loop...")
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        print(f"Epoch {epoch+1}/{epochs} start...")
        for batch_idx, batch in enumerate(train_loader):
            states = batch.states.to(device, non_blocking=True)
            actions = batch.actions.to(device, non_blocking=True)

            loss = trainer.train_step(states, actions)
            total_loss += loss
            count += 1
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss}")

        avg_loss = total_loss / count if count > 0 else float('inf')
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss}")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved as model_weights.pth")
