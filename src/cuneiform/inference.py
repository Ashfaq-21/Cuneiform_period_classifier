"""Load a trained model bundle and return structured predictions.

Example::

    from cuneiform.inference import Predictor
    predictor = Predictor.from_latest()
    result = predictor.predict_tablet("HS_0768")      # cached depth map
    result = predictor.predict_scan("my_tablet.ply")  # raw scan, same preprocessing
    print(result.predicted_period, result.confidence)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cuneiform.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config
from cuneiform.data.cache import DepthMapCache
from cuneiform.geometry.depth_map import FaceDepthMaps, scan_to_face_depth_maps
from cuneiform.modeling.bundle import ModelBundle, load_bundle, models_root, resolve_bundle_dir
from cuneiform.modeling.gradcam import grad_cam
from cuneiform.modeling.model import gradcam_target_layer, resolve_device
from cuneiform.modeling.transforms import model_input


@dataclass
class FacePrediction:
    face: str
    depth_map: np.ndarray  # [R, R] heights, NaN background
    probabilities: dict[str, float]
    gradcam: np.ndarray  # [input_size, input_size] in [0, 1], for the tablet-level predicted class


@dataclass
class Prediction:
    tablet_id: str | None
    predicted_period: str
    confidence: float  # softmax probability of the predicted class, averaged over faces
    probabilities: dict[str, float]
    faces: list[FacePrediction]
    model_run: str

    def to_dict(self) -> dict:
        """JSON-friendly summary (no arrays)."""
        return {
            "tablet_id": self.tablet_id,
            "predicted_period": self.predicted_period,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "faces": [f.face for f in self.faces],
            "model_run": self.model_run,
        }


class Predictor:
    def __init__(self, bundle: ModelBundle, data_config: ProjectConfig, device: torch.device) -> None:
        self.bundle = bundle
        self.device = device
        # Preprocessing and labels come from the bundle; data paths from this machine's config.
        self.config = bundle.config.model_copy(update={"paths": data_config.paths})
        self.cache = DepthMapCache(self.config.paths.processed_dir, self.config.preprocessing)

    @classmethod
    def from_latest(cls, config_path: Path | str = DEFAULT_CONFIG_PATH, run: str | None = None,
                    device: str = "auto") -> "Predictor":
        """Load the bundle named in ``artifacts/models/LATEST`` (or ``run``).

        Raises ``ModelBundleNotFoundError`` with instructions when no model exists.
        """
        data_config = load_config(config_path)
        torch_device = resolve_device(device)
        directory = resolve_bundle_dir(models_root(data_config), run)
        return cls(load_bundle(directory, torch_device), data_config, torch_device)

    @property
    def class_names(self) -> list[str]:
        return self.bundle.class_names

    def predict_tablet(self, tablet_id: str) -> Prediction:
        """Predict from the cached depth maps of a tablet listed in the manifest."""
        return self.predict_depth_maps(self.cache.load(tablet_id), tablet_id)

    def predict_scan(self, scan_path: Path | str, tablet_id: str | None = None) -> Prediction:
        """Render a raw .ply with the bundle's preprocessing settings, then predict."""
        settings = self.config.preprocessing
        maps = scan_to_face_depth_maps(Path(scan_path), settings.resolution, settings.margin_fraction)
        return self.predict_depth_maps(maps, tablet_id)

    def predict_depth_maps(self, maps: FaceDepthMaps, tablet_id: str | None = None) -> Prediction:
        settings = self.config.preprocessing
        input_size = self.config.training.input_size
        model = self.bundle.model
        inputs = torch.stack([model_input(maps.face(f), settings, input_size) for f in settings.faces]).to(self.device)
        with torch.no_grad():
            face_probs = torch.softmax(model(inputs), dim=1).cpu().numpy()
        mean_probs = face_probs.mean(axis=0)
        predicted = int(mean_probs.argmax())
        layer = gradcam_target_layer(model)
        faces = [
            FacePrediction(
                face=face,
                depth_map=maps.face(face),
                probabilities=dict(zip(self.class_names, face_probs[i].astype(float).tolist())),
                gradcam=grad_cam(model, layer, inputs[i : i + 1], predicted).numpy(),
            )
            for i, face in enumerate(settings.faces)
        ]
        return Prediction(
            tablet_id=tablet_id,
            predicted_period=self.class_names[predicted],
            confidence=float(mean_probs[predicted]),
            probabilities=dict(zip(self.class_names, mean_probs.astype(float).tolist())),
            faces=faces,
            model_run=self.bundle.directory.name,
        )
