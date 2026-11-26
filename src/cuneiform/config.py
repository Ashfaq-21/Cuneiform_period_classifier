"""Validated project configuration loaded from ``configs/*.yaml``.

Everything that should stay consistent between preparation, training and
inference (paths, labels, depth-map settings, training settings) lives here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "prototype.yaml"

Face = Literal["primary", "secondary"]
InputChannel = Literal["depth", "relief", "mask"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    raw_dir: Path
    metadata_json: Path
    zip_index: Path
    manifest: Path
    processed_dir: Path
    artifacts_dir: Path

    def resolved(self, root: Path) -> "PathsConfig":
        return PathsConfig(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class DatasetConfig(_Strict):
    max_tablets_per_class: int = Field(150, ge=1)
    delete_raw_after_processing: bool = True
    split_fractions: dict[Literal["train", "val", "test"], float]
    seed: int = 13

    @field_validator("split_fractions")
    @classmethod
    def _fractions_sum_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != {"train", "val", "test"}:
            raise ValueError("split_fractions needs exactly the keys train, val, test")
        if abs(sum(value.values()) - 1.0) > 1e-6 or min(value.values()) <= 0:
            raise ValueError(f"split_fractions must be positive and sum to 1, got {value}")
        return value


class PreprocessingConfig(_Strict):
    """Settings that change the cached depth maps (part of the cache key) plus input channels."""

    resolution: int = Field(256, ge=32, le=2048)
    faces: list[Face] = ["primary"]
    margin_fraction: float = Field(0.04, ge=0.0, lt=0.4)
    input_channels: list[InputChannel] = ["depth", "relief", "mask"]

    @field_validator("faces", "input_channels")
    @classmethod
    def _non_empty_unique(cls, value: list) -> list:
        if not value or len(set(value)) != len(value):
            raise ValueError("must be a non-empty list without duplicates")
        return value

    def cache_key(self) -> str:
        """Short hash of the settings that affect cached depth maps (not input_channels or faces)."""
        # Bump "version" whenever rendering or face selection logic changes.
        relevant = {"resolution": self.resolution, "margin_fraction": self.margin_fraction, "version": 2}
        digest = hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:10]
        return f"r{self.resolution}_{digest}"


class TrainingConfig(_Strict):
    model: Literal["resnet18"] = "resnet18"
    pretrained: bool = True
    input_size: int = Field(224, ge=32)
    epochs: int = Field(25, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    num_workers: int = Field(0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 13


class ProjectConfig(_Strict):
    paths: PathsConfig
    labels: dict[str, list[str]]
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig

    @field_validator("labels")
    @classmethod
    def _labels_valid(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        if len(value) < 2:
            raise ValueError("labels must define at least two classes")
        seen: dict[str, str] = {}
        for label, sources in value.items():
            if not sources:
                raise ValueError(f"label '{label}' has no source period strings")
            for source in sources:
                if source in seen:
                    raise ValueError(f"period string '{source}' is mapped to both '{seen[source]}' and '{label}'")
                seen[source] = label
        return value

    @model_validator(mode="after")
    def _check_input_size(self) -> "ProjectConfig":
        if self.training.input_size > self.preprocessing.resolution:
            raise ValueError("training.input_size cannot exceed preprocessing.resolution")
        return self

    @property
    def class_names(self) -> list[str]:
        """Class names in a stable order; index i is the model's output i."""
        return list(self.labels)

    def label_for_period(self, period_raw: str) -> str | None:
        for label, sources in self.labels.items():
            if period_raw.strip() in sources:
                return label
        return None


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load and validate a YAML config; relative paths resolve against the repo root."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Pass --config configs/prototype.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = ProjectConfig.model_validate(data)
    return config.model_copy(update={"paths": config.paths.resolved(REPO_ROOT)})
