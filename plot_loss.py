import matplotlib.pyplot as plt

# Example data
epochs = list(range(1, 26))
train_loss = [0.58, 0.47, 0.39, 0.34, 0.31, 0.28, 0.27, 0.26, 0.25, 0.24] + [0.23]*15
val_loss = [0.54, 0.44, 0.39, 0.35, 0.33, 0.31, 0.30, 0.29, 0.28, 0.27] + [0.26]*15

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=300)
plt.show()
