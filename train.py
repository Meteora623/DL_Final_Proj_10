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
        device=device,
        batch_size=64,
        train=True,
    )
    print("Data loader created.")

    # 尝试打印数据集长度和一个样本的形状
    ds = train_loader.dataset
    print("Dataset length:", len(ds))
    sample = ds[0]
    print("Sample states shape:", sample.states.shape)
    print("Sample actions shape:", sample.actions.shape)

    print("Creating model...")
    model = JEPAModel(repr_dim=256, momentum=0.99).to(device)
    print("Model created.")

    trainer = JEPATrainer(model, device=device, lr=1e-3, momentum=0.99)
    print("Trainer created.")

    epochs = 10
    print("Start training loop...")
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        print(f"Epoch {epoch+1}/{epochs} start...")
        for batch_idx, batch in enumerate(train_loader):
            # 在这里插入打印看是否成功拿到batch
            print(f"Processing batch {batch_idx}...")
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, 2]
            # 打印batch形状来确认是否数据正常
            print(f"states shape: {states.shape}, actions shape: {actions.shape}")

            loss = trainer.train_step(states, actions)
            total_loss += loss
            count += 1
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss}")

        avg_loss = total_loss / count if count > 0 else float('inf')
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss}")

    # 保存模型
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved as model_weights.pth")
