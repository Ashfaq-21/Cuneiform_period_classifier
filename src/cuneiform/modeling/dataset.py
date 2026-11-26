"""PyTorch dataset over cached depth maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from cuneiform.config import ProjectConfig
from cuneiform.data.cache import DepthMapCache
from cuneiform.data.manifest import ManifestRow, read_manifest, validate_manifest
from cuneiform.data.splits import check_split_isolation
from cuneiform.modeling.transforms import model_input


@dataclass(frozen=True)
class Sample:
    tablet_id: str
    face: str
    label_index: int


class DepthMapDataset(Dataset):
    """One item per (tablet, face). Faces of one tablet are always in the same split."""

    def __init__(self, samples: list[Sample], cache: DepthMapCache, config: ProjectConfig) -> None:
        self.samples = samples
        self.cache = cache
        self.config = config
        # Depth maps are small (256x256 float32 = 256 KB), so keep tensors in memory.
        self._tensors: dict[tuple[str, str], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        key = (sample.tablet_id, sample.face)
        if key not in self._tensors:
            depth = self.cache.load(sample.tablet_id).face(sample.face)
            self._tensors[key] = model_input(
                depth, self.config.preprocessing, self.config.training.input_size
            )
        return self._tensors[key], sample.label_index


def load_split_rows(config: ProjectConfig) -> tuple[dict[str, list[ManifestRow]], dict[str, int]]:
    """Return manifest rows grouped by split (only tablets with a cached depth map).

    Also returns ``{split: number of manifest rows without a usable depth map}``.
    """
    rows = read_manifest(config.paths.manifest)
    validate_manifest(rows, config.class_names)
    check_split_isolation(rows)
    cache = DepthMapCache(config.paths.processed_dir, config.preprocessing)
    grouped: dict[str, list[ManifestRow]] = {"train": [], "val": [], "test": []}
    missing: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        if row.split not in grouped:
            continue
        if cache.has(row.tablet_id):
            grouped[row.split].append(row)
        else:
            missing[row.split] += 1
    return grouped, missing


def build_dataset(rows: list[ManifestRow], config: ProjectConfig) -> DepthMapDataset:
    class_index = {name: i for i, name in enumerate(config.class_names)}
    samples = [
        Sample(row.tablet_id, face, class_index[row.period])
        for row in rows
        for face in config.preprocessing.faces
    ]
    return DepthMapDataset(samples, DepthMapCache(config.paths.processed_dir, config.preprocessing), config)
