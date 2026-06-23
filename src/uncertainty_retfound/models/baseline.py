"""Small baseline image classification models."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNNClassifier(nn.Module):
    """A minimal CNN classifier for smoke-testing image pipelines."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)
