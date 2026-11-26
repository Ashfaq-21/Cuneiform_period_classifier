"""Loading and validating 3D scans."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class InvalidScanError(ValueError):
    """Raised when a scan cannot be turned into a depth map. The message names the problem."""


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh and return ``(vertices[N,3] float64, triangles[M,3] int32)``.

    Units are whatever the scan uses (HeiCuBeDa: millimetres).
    """
    import open3d as o3d  # imported lazily: open3d is slow to import

    path = Path(path)
    if not path.exists():
        raise InvalidScanError(f"scan file does not exist: {path}")
    if path.suffix.lower() not in {".ply", ".obj", ".stl", ".off"}:
        raise InvalidScanError(f"unsupported scan format '{path.suffix}' (use .ply)")
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    validate_mesh_arrays(vertices, triangles, source=str(path))
    return vertices, triangles


def validate_mesh_arrays(vertices: np.ndarray, triangles: np.ndarray, source: str = "mesh") -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise InvalidScanError(f"{source}: needs at least 4 vertices, got shape {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 4:
        raise InvalidScanError(
            f"{source}: no triangle faces found (point clouds are not supported), shape {triangles.shape}"
        )
    if not np.isfinite(vertices).all():
        raise InvalidScanError(f"{source}: vertices contain NaN or infinite coordinates")
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    if np.min(extent) <= 0:
        raise InvalidScanError(f"{source}: mesh is degenerate (zero extent along an axis: {extent})")
