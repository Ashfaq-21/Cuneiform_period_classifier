"""Depth map -> normalised model input tensor. Used by training, evaluation and inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cuneiform.config import PreprocessingConfig
from cuneiform.geometry.representation import depth_to_channels

# Channels are in [0, 1]; centre them roughly like ImageNet inputs for the pretrained stem.
CHANNEL_MEAN = 0.5
CHANNEL_STD = 0.25


def model_input(depth: np.ndarray, preprocessing: PreprocessingConfig, input_size: int) -> torch.Tensor:
    """Return a float32 tensor ``[C, input_size, input_size]`` ready for the network."""
    channels = torch.from_numpy(depth_to_channels(depth, preprocessing.input_channels))
    if channels.shape[-1] != input_size:
        channels = F.interpolate(
            channels[None], size=(input_size, input_size), mode="bilinear", align_corners=False
        )[0]
    return (channels - CHANNEL_MEAN) / CHANNEL_STD


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Training-only augmentation on ``[B, C, H, W]``.

    * 180-degree rotation (PCA cannot tell the top of a tablet from the bottom),
    * small rotation (+-10 degrees) and scale (0.9-1.1) via an affine grid.

    No mirroring: a mirrored face is not a physically possible tablet.
    """
    b = batch.shape[0]
    flip = torch.rand(b, generator=generator) < 0.5
    angle = (torch.rand(b, generator=generator) * 20 - 10) * np.pi / 180 + flip.float() * np.pi
    scale = torch.rand(b, generator=generator) * 0.2 + 0.9
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(b, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    grid = F.affine_grid(theta.to(batch.device), list(batch.shape), align_corners=False)
    background = -CHANNEL_MEAN / CHANNEL_STD  # value of an all-zero (empty) pixel
    shifted = batch - background
    return F.grid_sample(shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + background
