import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import DeepVesselNet
from src.data_loader import get_dataloaders


# ---------- METRIC FUNCTIONS ----------
def dice_coef(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    return ((2. * intersection + smooth) /
            (y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + smooth)).mean().item()

def iou_score(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = (y_pred + y_true).sum(dim=(2, 3)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()

def sensitivity(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    tp = (y_pred * y_true).sum(dim=(2, 3))
    fn = ((1 - y_pred) * y_true).sum(dim=(2, 3))
    return ((tp + smooth) / (tp + fn + smooth)).mean().item()

def specificity(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    tn = ((1 - y_pred) * (1 - y_true)).sum(dim=(2, 3))
    fp = (y_pred * (1 - y_true)).sum(dim=(2, 3))
    return ((tn + smooth) / (tn + fp + smooth)).mean().item()


# ---------- EVALUATION FUNCTION ----------
def evaluate_model():
    BASE_DIR = r"C:\Users\Adith Sree Jey\OneDrive\Desktop\Project\New\DeepVessel_Net-main"
    test_img_dir = os.path.join(BASE_DIR, "Data", "test", "image")
    test_mask_dir = os.path.join(BASE_DIR, "Data", "test", "mask")
    checkpoint_path = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_model.pth")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVesselNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # DataLoader
    _, test_loader = get_dataloaders(
        train_img_dir=test_img_dir, train_mask_dir=test_mask_dir,
        test_img_dir=test_img_dir, test_mask_dir=test_mask_dir,
        batch_size=1, img_size=256
    )

    all_dice, all_iou, all_sens, all_spec = [], [], [], []
    save_dir = os.path.join(BASE_DIR, "outputs", "predictions")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            # Metrics
            all_dice.append(dice_coef(outputs, masks))
            all_iou.append(iou_score(outputs, masks))
            all_sens.append(sensitivity(outputs, masks))
            all_spec.append(specificity(outputs, masks))

            # Save prediction image
            pred_mask = (outputs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            img_np = images.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(save_dir, f"pred_{idx}.png"), pred_mask)

            # Optional: show sample result
            if idx < 3:  # show first 3 examples
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                plt.title("Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(masks.squeeze().cpu(), cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask, cmap="gray")
                plt.title("Prediction")
                plt.axis("off")
                plt.show()

    # Final metrics
    print("\n📊 Evaluation Results:")
    print(f"Dice Coefficient: {np.mean(all_dice):.4f}")
    print(f"IoU Score: {np.mean(all_iou):.4f}")
    print(f"Sensitivity: {np.mean(all_sens):.4f}")
    print(f"Specificity: {np.mean(all_spec):.4f}")


if __name__ == "__main__":
    evaluate_model()
