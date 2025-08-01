import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    alpha: weight for False Positives (FP)
    beta:  weight for False Negatives (FN)
    smooth: small constant to avoid division by zero
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure the shapes match
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()

        # True positives, false positives, false negatives
        tp = (y_pred * y_true).sum(dim=(2, 3))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3))

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky_index.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss: puts more focus on hard-to-detect pixels
    gamma > 1 focuses more on difficult examples
    Lower alpha, higher beta increases sensitivity to thin vessels
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        tversky_loss = self.tversky(y_pred, y_true)
        return torch.pow(tversky_loss, self.gamma)
