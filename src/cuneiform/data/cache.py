"""On-disk cache of rendered depth maps, keyed by tablet ID and preprocessing settings.

Layout: ``<processed_dir>/<cache_key>/<tablet_id>.npz`` with arrays ``primary``,
``secondary`` (float32 heights, NaN background) and ``pixel_size``. Changing
resolution or margin changes the cache key, so stale maps are never reused.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cuneiform.config import PreprocessingConfig
from cuneiform.geometry.depth_map import FaceDepthMaps


class DepthMapCache:
    def __init__(self, processed_dir: Path, preprocessing: PreprocessingConfig) -> None:
        self.directory = Path(processed_dir) / preprocessing.cache_key()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.invalid_log = self.directory / "invalid_scans.csv"
        settings_file = self.directory / "settings.json"
        if not settings_file.exists():
            settings_file.write_text(json.dumps(preprocessing.model_dump(), indent=2))

    def path(self, tablet_id: str) -> Path:
        return self.directory / f"{tablet_id}.npz"

    def has(self, tablet_id: str) -> bool:
        return self.path(tablet_id).exists()

    def save(self, tablet_id: str, maps: FaceDepthMaps) -> Path:
        target = self.path(tablet_id)
        temporary = target.with_name(target.stem + ".tmp.npz")  # atomic: never leave half files
        np.savez_compressed(
            temporary,
            primary=maps.primary.astype(np.float32),
            secondary=maps.secondary.astype(np.float32),
            pixel_size=np.float32(maps.pixel_size),
            info=json.dumps(maps.info),
        )
        os.replace(temporary, target)
        return target

    def load(self, tablet_id: str) -> FaceDepthMaps:
        if not self.has(tablet_id):
            raise FileNotFoundError(
                f"No cached depth map for {tablet_id} in {self.directory}. Run 'python -m cuneiform.cli prepare'."
            )
        with np.load(self.path(tablet_id)) as data:
            return FaceDepthMaps(
                primary=data["primary"],
                secondary=data["secondary"],
                pixel_size=float(data["pixel_size"]),
                info=json.loads(str(data["info"])),
            )

    # Invalid-scan log ------------------------------------------------------
    def log_invalid(self, tablet_id: str, reason: str) -> None:
        new_file = not self.invalid_log.exists()
        with open(self.invalid_log, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["tablet_id", "reason", "logged_at"])
            writer.writerow([tablet_id, reason, datetime.now(timezone.utc).isoformat(timespec="seconds")])

    def invalid_ids(self) -> set[str]:
        if not self.invalid_log.exists():
            return set()
        with open(self.invalid_log, newline="", encoding="utf-8") as handle:
            return {row["tablet_id"] for row in csv.DictReader(handle)}
