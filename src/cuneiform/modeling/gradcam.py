"""Grad-CAM (Selvaraju et al., 2017): which regions influenced a prediction.

The map shows where activations of the target layer most increased the score of
the chosen class. It is an attribution heuristic, not an explanation of
historical reasoning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def grad_cam(model: nn.Module, layer: nn.Module, inputs: torch.Tensor, class_index: int) -> torch.Tensor:
    """Return a ``[H, W]`` heatmap in [0, 1] for one input ``[1, C, H, W]``."""
    activations: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: activations.__setitem__("grad", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            inputs = inputs.clone().requires_grad_(True)
            scores = model(inputs)
            scores[0, class_index].backward()
    finally:
        handle.remove()

    feature_maps, gradients = activations["value"].detach(), activations["grad"].detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
    cam = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    peak = cam.max()
    return (cam / peak if peak > 0 else cam).cpu()
