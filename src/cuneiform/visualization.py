"""Display helpers shared by the CLI preview and the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cuneiform.geometry.depth_map import FaceDepthMaps
from cuneiform.geometry.representation import hillshade, normalized_depth, relief


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Colour-mapped depth (viridis), background white. Returns uint8 ``[H, W, 3]``."""
    from matplotlib import colormaps

    mask = np.isfinite(depth)
    rgb = colormaps["viridis"](normalized_depth(depth))[..., :3]
    rgb[~mask] = 1.0
    return (rgb * 255).astype(np.uint8)


def shaded_to_rgb(depth: np.ndarray) -> np.ndarray:
    gray = hillshade(depth)
    return (np.repeat(gray[..., None], 3, axis=2) * 255).astype(np.uint8)


def gradcam_overlay(depth: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heatmap (any size) over the shaded depth map. Returns uint8 RGB."""
    from matplotlib import colormaps
    from PIL import Image

    base = hillshade(depth)
    size = base.shape[::-1]
    heat = np.asarray(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(size, Image.BILINEAR)) / 255.0
    heat = np.where(np.isfinite(depth), heat, 0.0)  # do not colour the background
    colour = colormaps["jet"](heat)[..., :3]
    blended = (1 - alpha * heat[..., None]) * base[..., None] + alpha * heat[..., None] * colour
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def save_depth_preview(maps: FaceDepthMaps, path: Path, title: str = "") -> Path:
    """Write a 2x3 panel: depth, shaded relief and relief channel for both faces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row, face in enumerate(("primary", "secondary")):
        depth = maps.face(face)
        panels = [(depth_to_rgb(depth), "depth"), (shaded_to_rgb(depth), "shaded"),
                  (relief(depth), "relief channel")]
        for col, (image, label) in enumerate(panels):
            axes[row, col].imshow(image, cmap="gray" if image.ndim == 2 else None)
            axes[row, col].set_title(f"{face}: {label}", fontsize=9)
            axes[row, col].axis("off")
    figure.suptitle(f"{title}  ({maps.pixel_size:.3f} units/pixel)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
