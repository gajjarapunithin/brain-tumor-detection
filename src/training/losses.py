"""Loss functions for segmentation and classification."""

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Soft Dice loss for binary or multi-label masks."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.reshape(probabilities.size(0), -1)
        targets = targets.float().reshape(targets.size(0), -1)
        intersection = (probabilities * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class DiceBCELoss(nn.Module):
    """Weighted combination of Dice and binary cross-entropy losses."""

    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + (1 - self.dice_weight) * self.bce(logits, targets)


def classification_loss() -> nn.CrossEntropyLoss:
    """Return the standard multi-class loss."""
    return nn.CrossEntropyLoss()
