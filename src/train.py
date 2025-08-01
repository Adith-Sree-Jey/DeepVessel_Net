import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.model import DeepVesselNet
from src.data_loader import get_dataloaders
from src.loss import FocalTverskyLoss
import matplotlib.pyplot as plt

from src.loss import FocalTverskyLoss
criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.33)
 
# ---------- TRAIN FUNCTION ----------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ---------- VALIDATION FUNCTION ----------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

    return running_loss / len(loader)


# ---------- MAIN TRAINING LOOP ----------
def main():
    # Paths
    BASE_DIR = r"C:\Users\Adith Sree Jey\OneDrive\Desktop\Project\New\DeepVessel_Net-main"
    train_img_dir = os.path.join(BASE_DIR, "Data", "train", "image")
    train_mask_dir = os.path.join(BASE_DIR, "Data", "train", "mask")
    test_img_dir = os.path.join(BASE_DIR, "Data", "test", "image")
    test_mask_dir = os.path.join(BASE_DIR, "Data", "test", "mask")

    # Hyperparameters
    lr = 1e-4
    batch_size = 2  # 512x512 requires smaller batch
    num_epochs = 50
    img_size = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_loader, val_loader = get_dataloaders(
        train_img_dir, train_mask_dir,
        test_img_dir, test_mask_dir,
        batch_size=batch_size, img_size=img_size
    )

    # Model
    model = DeepVesselNet(in_channels=3, out_channels=1).to(device)

    # Loss function (Focal Tversky)
    criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.33)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Tracking losses
    train_losses, val_losses = [], []

    # Best model tracking
    best_val_loss = float("inf")
    checkpoint_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")

    # ---------- LOSS CURVE PLOTTING ----------
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "outputs", "loss_curve.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
