"""A model bundle = weights + everything needed to reproduce its preprocessing.

``artifacts/models/<run_name>/``
    ``model.pt``      state dict
    ``bundle.json``   class names, full config, cache key, training history, metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cuneiform.config import ProjectConfig
from cuneiform.modeling.model import build_model

WEIGHTS_FILE = "model.pt"
METADATA_FILE = "bundle.json"
LATEST_POINTER = "LATEST"


class ModelBundleNotFoundError(FileNotFoundError):
    pass


@dataclass
class ModelBundle:
    directory: Path
    model: nn.Module
    config: ProjectConfig
    metadata: dict

    @property
    def class_names(self) -> list[str]:
        return self.metadata["class_names"]


def models_root(config: ProjectConfig) -> Path:
    return config.paths.artifacts_dir / "models"


def save_bundle(directory: Path, model: nn.Module, config: ProjectConfig, metadata: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / WEIGHTS_FILE)
    full = {
        "class_names": config.class_names,
        "config": json.loads(config.model_dump_json()),
        "cache_key": config.preprocessing.cache_key(),
        **metadata,
    }
    (directory / METADATA_FILE).write_text(json.dumps(full, indent=2))
    (directory.parent / LATEST_POINTER).write_text(directory.name)
    return directory


def update_bundle_metadata(directory: Path, **fields) -> None:
    path = directory / METADATA_FILE
    data = json.loads(path.read_text())
    data.update(fields)
    path.write_text(json.dumps(data, indent=2))


def resolve_bundle_dir(models_dir: Path, run: str | None = None) -> Path:
    """Find a bundle directory: an explicit run name/path, or the one named in ``LATEST``."""
    if run:
        candidate = Path(run)
        directory = candidate if candidate.is_dir() else models_dir / run
    else:
        pointer = models_dir / LATEST_POINTER
        if not pointer.exists():
            raise ModelBundleNotFoundError(
                f"No trained model found in {models_dir}. Train one with: python -m cuneiform.cli train"
            )
        directory = models_dir / pointer.read_text().strip()
    for required in (WEIGHTS_FILE, METADATA_FILE):
        if not (directory / required).exists():
            raise ModelBundleNotFoundError(
                f"Model bundle {directory} is incomplete: missing {required}. Re-run training."
            )
    return directory


def load_bundle(directory: Path, device: torch.device | str = "cpu") -> ModelBundle:
    metadata = json.loads((directory / METADATA_FILE).read_text())
    config = ProjectConfig.model_validate(metadata["config"])
    model = build_model(
        config.training.model,
        num_classes=len(metadata["class_names"]),
        in_channels=len(config.preprocessing.input_channels),
        pretrained=False,
    )
    state = torch.load(directory / WEIGHTS_FILE, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return ModelBundle(directory=directory, model=model, config=config, metadata=metadata)
