"""Transfer-learning image classifier."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class TumorClassifier(nn.Module):
    """ResNet50 classifier adapted to one-channel MRI inputs and four classes."""

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        original = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, original.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze feature extraction layers while keeping the head trainable."""
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad = trainable or name.startswith("fc.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
