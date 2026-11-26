"""Render an orthographic depth map of one tablet face with Open3D ray casting.

A depth map here is a float32 array ``[resolution, resolution]`` holding the
height (scan units, mm for HeiCuBeDa) of the surface seen from +z, with ``NaN``
where the ray missed the tablet. Row 0 is the top of the tablet (max y).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cuneiform.geometry.mesh_io import load_mesh_arrays, validate_mesh_arrays
from cuneiform.geometry.orientation import canonical_pose, flip_to_other_face

MIN_COVERAGE = 0.05  # a face covering <5% of the frame is almost certainly a bad scan


@dataclass
class FaceDepthMaps:
    """Both large faces of a tablet.

    ``primary`` is the face with more fine surface relief (usually the more densely
    inscribed side). This is a heuristic; obverse/reverse is not known from geometry alone.
    """

    primary: np.ndarray
    secondary: np.ndarray
    pixel_size: float  # scan units per pixel (same for both faces)
    info: dict = field(default_factory=dict)

    def face(self, name: str) -> np.ndarray:
        if name not in ("primary", "secondary"):
            raise KeyError(f"unknown face '{name}', expected 'primary' or 'secondary'")
        return getattr(self, name)


def render_depth_map(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int,
    center_xy: np.ndarray,
    side: float,
) -> np.ndarray:
    """Cast one ray per pixel straight down (-z) and record the hit height."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices.astype(np.float32)), o3d.core.Tensor(triangles.astype(np.uint32))
    )
    top = float(vertices[:, 2].max()) + 1.0
    pixel = side / resolution
    # Pixel centres; row 0 = max y so the image is upright.
    xs = center_xy[0] - side / 2 + (np.arange(resolution) + 0.5) * pixel
    ys = center_xy[1] + side / 2 - (np.arange(resolution) + 0.5) * pixel
    grid_x, grid_y = np.meshgrid(xs, ys)
    rays = np.zeros((resolution, resolution, 6), dtype=np.float32)
    rays[..., 0], rays[..., 1], rays[..., 2] = grid_x, grid_y, top
    rays[..., 5] = -1.0
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    depth = np.where(np.isfinite(t_hit), top - t_hit, np.nan).astype(np.float32)
    return depth


def _fine_relief_score(depth: np.ndarray) -> float:
    """Mean absolute fine-scale relief inside the face (edges excluded); larger = more inscribed.

    Wedges are sharp, millimetre-scale dents, so the inscribed side scores higher
    than a smooth or broken back. Both faces share one pixel size, so scores compare directly.
    """
    from scipy.ndimage import binary_erosion, gaussian_filter

    mask = np.isfinite(depth)
    interior = binary_erosion(mask, iterations=max(2, depth.shape[0] // 20))
    if interior.sum() < 10:
        return 0.0
    sigma = max(1.0, 0.015 * depth.shape[1])
    filled = np.where(mask, depth, 0.0)
    smooth = gaussian_filter(filled, sigma) / np.maximum(gaussian_filter(mask.astype(float), sigma), 1e-6)
    return float(np.abs(filled - smooth)[interior].mean())


def _face_curvature(depth: np.ndarray) -> float:
    """Interquartile height range in the central part of a face; smaller = flatter."""
    h, w = depth.shape
    centre = depth[h // 5 : h - h // 5, w // 5 : w - w // 5]
    values = centre[np.isfinite(centre)]
    if values.size < 10:
        return float("inf")
    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)


def mesh_to_face_depth_maps(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int = 256,
    margin_fraction: float = 0.04,
    source: str = "mesh",
) -> FaceDepthMaps:
    """Orient a mesh canonically and render both large faces at the same scale."""
    from cuneiform.geometry.mesh_io import InvalidScanError

    validate_mesh_arrays(vertices, triangles, source=source)
    posed = canonical_pose(vertices)
    lo, hi = posed[:, :2].min(axis=0), posed[:, :2].max(axis=0)
    side = float(np.max(hi - lo)) * (1 + 2 * margin_fraction)
    center = (lo + hi) / 2
    face_a = render_depth_map(posed, triangles, resolution, center, side)
    flipped = flip_to_other_face(posed)
    face_b = render_depth_map(flipped, triangles, resolution, np.array([-center[0], center[1]]), side)

    for name, depth in (("A", face_a), ("B", face_b)):
        coverage = float(np.isfinite(depth).mean())
        if coverage < MIN_COVERAGE:
            raise InvalidScanError(f"{source}: face {name} covers only {coverage:.1%} of the frame")

    relief_a, relief_b = _fine_relief_score(face_a), _fine_relief_score(face_b)
    primary, secondary = (face_a, face_b) if relief_a >= relief_b else (face_b, face_a)
    return FaceDepthMaps(
        primary=primary,
        secondary=secondary,
        pixel_size=side / resolution,
        info={"relief_a": relief_a, "relief_b": relief_b,
              "curvature_a": _face_curvature(face_a), "curvature_b": _face_curvature(face_b),
              "extent_xy": (hi - lo).tolist(),
              "thickness": float(posed[:, 2].max() - posed[:, 2].min())},
    )


def scan_to_face_depth_maps(path: Path, resolution: int = 256, margin_fraction: float = 0.04) -> FaceDepthMaps:
    """Load a scan file and render its faces. Raises InvalidScanError with a reason on failure."""
    vertices, triangles = load_mesh_arrays(path)
    return mesh_to_face_depth_maps(vertices, triangles, resolution, margin_fraction, source=str(path))
