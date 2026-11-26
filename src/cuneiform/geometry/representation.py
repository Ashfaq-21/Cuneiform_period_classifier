"""Turn a cached depth map into model input channels and display images.

This module is the single place where depth maps become tensors-to-be, so
training and inference cannot drift apart. To try another surface representation
(e.g. normals or curvature), add a channel name here and to ``InputChannel`` in
:mod:`cuneiform.config`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

RELIEF_SIGMA_FRACTION = 0.03  # smoothing scale for the relief channel, as a fraction of width
RELIEF_CLIP_STD = 3.0


def _mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth)


def normalized_depth(depth: np.ndarray) -> np.ndarray:
    """Heights rescaled to [0, 1] over the tablet; background 0."""
    mask = _mask(depth)
    out = np.zeros(depth.shape, dtype=np.float32)
    if mask.any():
        values = depth[mask]
        lo, hi = values.min(), values.max()
        out[mask] = (values - lo) / max(hi - lo, 1e-6)
    return out


def relief(depth: np.ndarray) -> np.ndarray:
    """Local relief: depth minus a Gaussian-smoothed surface, mapped to [0, 1] (0.5 = flat).

    Uses normalised convolution so the background does not bleed into the edges.
    """
    mask = _mask(depth)
    out = np.zeros(depth.shape, dtype=np.float32)
    if not mask.any():
        return out
    sigma = max(1.0, RELIEF_SIGMA_FRACTION * depth.shape[1])
    filled = np.where(mask, depth, 0.0).astype(np.float64)
    weight = gaussian_filter(mask.astype(np.float64), sigma)
    smooth = gaussian_filter(filled, sigma) / np.maximum(weight, 1e-6)
    residual = (filled - smooth)[mask]
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6  # robust std
    out[mask] = np.clip(0.5 + residual / (2 * RELIEF_CLIP_STD * scale), 0.0, 1.0)
    return out


_CHANNELS = {
    "depth": normalized_depth,
    "relief": relief,
    "mask": lambda depth: _mask(depth).astype(np.float32),
}


def depth_to_channels(depth: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    """Return a float32 array ``[len(channels), H, W]`` with values in [0, 1]."""
    unknown = [c for c in channels if c not in _CHANNELS]
    if unknown:
        raise ValueError(f"unknown input channel(s) {unknown}; available: {sorted(_CHANNELS)}")
    return np.stack([_CHANNELS[c](depth) for c in channels]).astype(np.float32)


def hillshade(depth: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    """Shaded-relief image in [0, 1] for display (background = 1, white)."""
    mask = _mask(depth)
    if not mask.any():
        return np.ones(depth.shape, dtype=np.float32)
    filled = np.where(mask, depth, np.nanmin(depth))
    span = float(np.nanmax(depth) - np.nanmin(depth)) or 1.0
    gy, gx = np.gradient(filled / span * depth.shape[0] * 0.15)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.where(mask, np.clip(shaded, 0, 1), 1.0).astype(np.float32)
