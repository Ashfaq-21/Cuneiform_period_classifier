"""The dataset manifest: one row per tablet, stored as CSV.

Columns (all required, header row first):

==============  ===============================================================
tablet_id       Unique ID, e.g. ``HS_0768``. All views/faces of a tablet share it.
period          Class label exactly as it appears in ``labels`` in the config.
source          Where the scan came from, e.g. ``HeiCuBeDa`` or ``local``.
local_path      Path to the .ply, relative to the repo root or absolute. May not
                exist once the scan is processed and deleted (the cache is used).
split           ``train``, ``val``, ``test`` or empty (assigned by ``prepare``).
==============  ===============================================================

To use your own scans, write this CSV by hand with an empty ``split`` column
and run ``python -m cuneiform.cli prepare``.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

MANIFEST_COLUMNS = ["tablet_id", "period", "source", "local_path", "split"]
VALID_SPLITS = {"train", "val", "test", ""}


class ManifestError(ValueError):
    """Raised when the manifest is malformed. The message lists every problem found."""


@dataclass(frozen=True)
class ManifestRow:
    tablet_id: str
    period: str
    source: str
    local_path: str
    split: str = ""

    def with_split(self, split: str) -> "ManifestRow":
        return replace(self, split=split)


def read_manifest(path: Path) -> list[ManifestRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run 'python -m cuneiform.cli build-manifest' "
            "(HeiCuBeDa) or write it by hand (see data/README.md)."
        )
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ManifestError(f"{path}: missing column(s) {sorted(missing)}; expected {MANIFEST_COLUMNS}")
        return [ManifestRow(**{c: (row[c] or "").strip() for c in MANIFEST_COLUMNS}) for row in reader]


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    temporary.replace(path)


def validate_manifest(rows: list[ManifestRow], class_names: list[str]) -> None:
    """Check IDs are unique, labels are known and splits are valid. Raises ManifestError."""
    problems: list[str] = []
    duplicates = [tid for tid, n in Counter(r.tablet_id for r in rows).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate tablet_id(s): {duplicates[:10]}")
    for row in rows:
        if not row.tablet_id:
            problems.append("a row has an empty tablet_id")
        if row.period not in class_names:
            problems.append(f"{row.tablet_id}: period '{row.period}' is not one of {class_names}")
        if row.split not in VALID_SPLITS:
            problems.append(f"{row.tablet_id}: split '{row.split}' must be train, val, test or empty")
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ManifestError(f"Manifest has {len(problems)} problem(s):\n  {shown}{more}")


def class_counts(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    """``{period: {split: count, 'total': count}}``."""
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = summary.setdefault(row.period, {"total": 0})
        bucket["total"] += 1
        key = row.split or "unassigned"
        bucket[key] = bucket.get(key, 0) + 1
    return summary
