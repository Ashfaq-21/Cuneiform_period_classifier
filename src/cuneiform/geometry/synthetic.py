"""Synthetic tablet meshes for testing preprocessing (never for reporting accuracy)."""

from __future__ import annotations

import numpy as np


def synthetic_tablet(
    width: float = 40.0,
    height: float = 60.0,
    thickness: float = 15.0,
    wedges: int = 12,
    grid: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A pillow-shaped tablet (mm) with wedge dents pressed into the +z face only.

    Returns ``(vertices, triangles)``: two heightfield sheets (top and bottom) that
    share their outline, which is all the ray caster needs.
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(-width / 2, width / 2, grid)
    ys = np.linspace(-height / 2, height / 2, int(grid * height / width))
    gx, gy = np.meshgrid(xs, ys)
    bulge = (1 - (gx / (width / 2)) ** 2) * (1 - (gy / (height / 2)) ** 2)  # 0 at the rim, 1 at centre
    top = thickness / 2 * (0.3 + 0.7 * bulge)
    bottom = -thickness / 2 * (0.3 + 0.7 * bulge)
    for _ in range(wedges):  # triangular dents, ~1 mm deep
        cx, cy = rng.uniform(-0.35, 0.35) * width, rng.uniform(-0.35, 0.35) * height
        size = rng.uniform(2.0, 4.0)
        dent = np.clip(1 - (np.abs(gx - cx) + np.abs(gy - cy)) / size, 0, None)
        top -= 1.0 * dent

    rows, cols = gx.shape

    def sheet(z: np.ndarray, flip: bool) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])
        idx = np.arange(rows * cols).reshape(rows, cols)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        tris = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
        return vertices, (tris[:, ::-1] if flip else tris)

    v_top, t_top = sheet(top, flip=False)
    v_bottom, t_bottom = sheet(bottom, flip=True)
    vertices = np.vstack([v_top, v_bottom])
    triangles = np.vstack([t_top, t_bottom + len(v_top)]).astype(np.int32)
    return vertices, triangles


def save_ply(vertices: np.ndarray, triangles: np.ndarray, path) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    o3d.io.write_triangle_mesh(str(path), mesh)
