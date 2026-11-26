"""Model construction. Add new architectures to ``build_model``."""

from __future__ import annotations

import torch
from torch import nn


def build_model(name: str, num_classes: int, in_channels: int = 3, pretrained: bool = True) -> nn.Module:
    if name != "resnet18":
        raise ValueError(f"Unknown model '{name}'. Available: resnet18")
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_channels != 3:
        # Keep pretrained filters: average the RGB kernels and repeat for each input channel.
        old = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """The convolutional block Grad-CAM reads activations from."""
    return model.layer4


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "training.device is 'cuda' but PyTorch cannot see a GPU. Install a CUDA build of torch "
            "(see README, 'Optional: CUDA') or set device: cpu."
        )
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
