"""End-to-end on synthetic data: cache -> train (1 epoch, CPU) -> Predictor output contract."""

import numpy as np
import pytest
import yaml

from cuneiform.config import load_config
from cuneiform.data.cache import DepthMapCache
from cuneiform.data.manifest import ManifestRow, write_manifest
from cuneiform.data.splits import assign_splits
from cuneiform.geometry.depth_map import mesh_to_face_depth_maps
from cuneiform.geometry.synthetic import save_ply, synthetic_tablet
from cuneiform.inference import Prediction, Predictor
from cuneiform.modeling.bundle import ModelBundleNotFoundError
from cuneiform.modeling.train import train_model

RESOLUTION, INPUT_SIZE = 64, 64


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    config_data = {
        "paths": {name: str(root / name) for name in
                  ("raw_dir", "metadata_json", "zip_index", "processed_dir", "artifacts_dir")},
        "labels": {"Ur III": ["Ur III (ca. 2100-2000 BC)"], "Old Babylonian": ["Old Babylonian"]},
        "dataset": {"split_fractions": {"train": 0.6, "val": 0.2, "test": 0.2}, "seed": 0},
        "preprocessing": {"resolution": RESOLUTION, "faces": ["primary", "secondary"]},
        "training": {"pretrained": False, "input_size": INPUT_SIZE, "epochs": 1, "batch_size": 4, "device": "cpu"},
    }
    config_data["paths"]["manifest"] = str(root / "manifest.csv")
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))
    config = load_config(config_path)

    cache = DepthMapCache(config.paths.processed_dir, config.preprocessing)
    rows = []
    for i in range(10):
        period = "Ur III" if i % 2 else "Old Babylonian"
        vertices, triangles = synthetic_tablet(height=40 + 4 * i, wedges=5 + 3 * i, grid=40, seed=i)
        cache.save(f"T{i}", mesh_to_face_depth_maps(vertices, triangles, RESOLUTION))
        rows.append(ManifestRow(f"T{i}", period, "synthetic", f"T{i}.ply"))
    write_manifest(assign_splits(rows, config.dataset.split_fractions, seed=0), config.paths.manifest)
    bundle_dir = train_model(config, run_name="test_run")
    return config_path, config, bundle_dir, root


def test_predictor_returns_structured_prediction(project):
    config_path, config, bundle_dir, _ = project
    predictor = Predictor.from_latest(config_path, device="cpu")
    result = predictor.predict_tablet("T0")

    assert isinstance(result, Prediction)
    assert result.tablet_id == "T0" and result.model_run == "test_run"
    assert result.predicted_period in config.class_names
    assert set(result.probabilities) == set(config.class_names)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert 0.5 <= result.confidence <= 1.0
    assert [f.face for f in result.faces] == ["primary", "secondary"]
    for face in result.faces:
        assert face.depth_map.shape == (RESOLUTION, RESOLUTION)
        assert face.gradcam.shape == (INPUT_SIZE, INPUT_SIZE)
        assert np.isfinite(face.gradcam).all() and face.gradcam.min() >= 0 and face.gradcam.max() <= 1
    assert set(result.to_dict()) >= {"predicted_period", "confidence", "probabilities"}


def test_predict_scan_uses_same_preprocessing_as_cache(project):
    config_path, _, _, root = project
    predictor = Predictor.from_latest(config_path, device="cpu")
    vertices, triangles = synthetic_tablet(height=40, wedges=5, grid=40, seed=0)  # same mesh as T0
    save_ply(vertices, triangles, root / "T0.ply")
    from_scan = predictor.predict_scan(root / "T0.ply")
    from_cache = predictor.predict_tablet("T0")
    assert from_scan.predicted_period == from_cache.predicted_period
    assert from_scan.confidence == pytest.approx(from_cache.confidence, abs=1e-3)


def test_bundle_stores_class_mapping_and_preprocessing(project):
    _, config, bundle_dir, _ = project
    import json

    metadata = json.loads((bundle_dir / "bundle.json").read_text())
    assert metadata["class_names"] == config.class_names
    assert metadata["config"]["preprocessing"]["resolution"] == RESOLUTION
    assert metadata["cache_key"] == config.preprocessing.cache_key()


def test_missing_model_gives_actionable_error(tmp_path, project):
    config_path, *_ = project
    data = yaml.safe_load(open(config_path))
    data["paths"]["artifacts_dir"] = str(tmp_path / "empty_artifacts")
    other = tmp_path / "config.yaml"
    other.write_text(yaml.safe_dump(data))
    with pytest.raises(ModelBundleNotFoundError, match="cuneiform.cli train"):
        Predictor.from_latest(other, device="cpu")
