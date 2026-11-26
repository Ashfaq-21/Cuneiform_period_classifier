import numpy as np
import pytest

from cuneiform.geometry.depth_map import _fine_relief_score, mesh_to_face_depth_maps, scan_to_face_depth_maps
from cuneiform.geometry.mesh_io import InvalidScanError
from cuneiform.geometry.representation import depth_to_channels
from cuneiform.geometry.synthetic import save_ply, synthetic_tablet


def _random_rotation(seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_depth_map_shape_background_and_coverage():
    vertices, triangles = synthetic_tablet()
    maps = mesh_to_face_depth_maps(vertices, triangles, resolution=64)
    for depth in (maps.primary, maps.secondary):
        assert depth.shape == (64, 64)
        assert depth.dtype == np.float32
        assert np.isnan(depth[0, 0]) and np.isnan(depth[-1, -1])  # margin is background
        assert 0.4 < np.isfinite(depth).mean() < 0.95
    # 60 mm tall tablet + margin spread over 64 pixels.
    assert maps.pixel_size == pytest.approx(60 * 1.08 / 64, rel=0.05)


def test_tablet_is_upright_and_inscribed_face_is_primary_regardless_of_input_pose():
    vertices, triangles = synthetic_tablet(wedges=25)
    posed = vertices @ _random_rotation(3).T + np.array([100.0, -50.0, 7.0])
    maps = mesh_to_face_depth_maps(posed, triangles, resolution=96)
    rows = np.isfinite(maps.primary).any(axis=1).sum()
    cols = np.isfinite(maps.primary).any(axis=0).sum()
    assert rows > cols  # longest axis (height 60 mm) is vertical in the image
    # Only the +z face of the synthetic tablet has wedges; it must be chosen as primary.
    assert _fine_relief_score(maps.primary) > 3 * _fine_relief_score(maps.secondary)


def test_depth_map_roundtrip_through_ply(tmp_path):
    vertices, triangles = synthetic_tablet(grid=40)
    path = tmp_path / "tablet.ply"
    save_ply(vertices, triangles, path)
    maps = scan_to_face_depth_maps(path, resolution=48)
    assert np.isfinite(maps.primary).any()


def test_invalid_scans_raise_actionable_errors(tmp_path):
    with pytest.raises(InvalidScanError, match="does not exist"):
        scan_to_face_depth_maps(tmp_path / "missing.ply")
    points_only = np.random.default_rng(0).normal(size=(100, 3))
    with pytest.raises(InvalidScanError, match="no triangle faces"):
        mesh_to_face_depth_maps(points_only, np.zeros((0, 3), dtype=np.int32))


def test_model_channels_are_bounded_and_mask_matches_hits():
    vertices, triangles = synthetic_tablet()
    depth = mesh_to_face_depth_maps(vertices, triangles, resolution=64).primary
    channels = depth_to_channels(depth, ["depth", "relief", "mask"])
    assert channels.shape == (3, 64, 64)
    assert channels.min() >= 0.0 and channels.max() <= 1.0
    np.testing.assert_array_equal(channels[2] == 1.0, np.isfinite(depth))
    with pytest.raises(ValueError, match="unknown input channel"):
        depth_to_channels(depth, ["normals"])
