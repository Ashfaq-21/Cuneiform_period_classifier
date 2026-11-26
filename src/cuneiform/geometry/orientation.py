"""Put a tablet into a canonical pose before rendering.

Convention after :func:`canonical_pose`:

* origin at the centroid,
* z = thinnest principal axis (the viewing direction),
* y = longest principal axis (tablet height),
* x = remaining axis, forming a right-handed frame.

Which of the two large faces is "primary" is decided later from the rendered
depth maps (see :mod:`cuneiform.geometry.depth_map`), because PCA signs are arbitrary.
"""

from __future__ import annotations

import numpy as np


def principal_axes(vertices: np.ndarray, max_points: int = 200_000, seed: int = 0) -> np.ndarray:
    """Return a 3x3 rotation whose rows are principal axes sorted [longest, middle, thinnest]."""
    points = vertices
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt  # rows sorted by decreasing singular value


def canonical_pose(vertices: np.ndarray) -> np.ndarray:
    """Rotate and centre vertices into the canonical frame described in the module docstring."""
    axes = principal_axes(vertices)
    longest, middle, thinnest = axes
    x_axis = middle
    y_axis = longest
    z_axis = np.cross(x_axis, y_axis)  # right-handed, parallel to `thinnest`
    if np.dot(z_axis, thinnest) < 0:
        z_axis = -z_axis
        x_axis = -x_axis
    rotation = np.stack([x_axis, y_axis, z_axis])
    return (vertices - vertices.mean(axis=0)) @ rotation.T


def flip_to_other_face(vertices: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees about the y axis so the opposite face looks towards +z."""
    flipped = vertices.copy()
    flipped[:, 0] *= -1
    flipped[:, 2] *= -1
    return flipped
