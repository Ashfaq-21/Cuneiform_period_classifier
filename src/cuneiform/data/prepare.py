"""Build the manifest and render depth maps. Both steps are safe to re-run.

* :func:`build_heicubeda_manifest` picks labelled tablets from the metadata,
  samples up to ``max_tablets_per_class`` per class with a fixed seed and assigns
  tablet-level splits. Existing rows (and their splits) are preserved.
* :func:`prepare_depth_maps` makes sure every manifest row has a cached depth map,
  downloading missing HeiCuBeDa scans on demand. Already-cached tablets and
  tablets in the invalid-scan log are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from cuneiform.config import REPO_ROOT, ProjectConfig
from cuneiform.data import heicubeda
from cuneiform.data.cache import DepthMapCache
from cuneiform.data.manifest import ManifestRow, read_manifest, validate_manifest, write_manifest
from cuneiform.data.splits import assign_splits, check_split_isolation
from cuneiform.geometry.depth_map import scan_to_face_depth_maps
from cuneiform.geometry.mesh_io import InvalidScanError

log = logging.getLogger(__name__)


def resolve_scan_path(local_path: str) -> Path:
    path = Path(local_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def label_summary(config: ProjectConfig) -> dict[str, int]:
    """Count all HeiCuBeDa tablets per raw CDLI period string (before sampling)."""
    records = heicubeda.load_metadata(config.paths.metadata_json)
    return dict(Counter(r.period_raw or "<unknown>" for r in records).most_common())


def build_heicubeda_manifest(config: ProjectConfig) -> list[ManifestRow]:
    records = heicubeda.load_metadata(config.paths.metadata_json)
    index = heicubeda.load_zip_index(config.paths.zip_index)
    existing = read_manifest(config.paths.manifest) if config.paths.manifest.exists() else []
    existing_ids = {r.tablet_id for r in existing}
    scans_dir = config.paths.raw_dir / "heicubeda" / "scans"

    candidates: dict[str, list[str]] = {label: [] for label in config.class_names}
    for record in records:
        label = config.label_for_period(record.period_raw)
        if label and record.tablet_id in index:
            candidates[label].append(record.tablet_id)

    rng = random.Random(config.dataset.seed)
    rows = list(existing)
    for label in config.class_names:
        have = sum(1 for r in existing if r.period == label)
        pool = sorted(set(candidates[label]) - existing_ids)
        rng.shuffle(pool)
        for tablet_id in pool[: max(0, config.dataset.max_tablets_per_class - have)]:
            rows.append(
                ManifestRow(
                    tablet_id=tablet_id,
                    period=label,
                    source=heicubeda.SOURCE_NAME,
                    local_path=_relative_to_repo(scans_dir / f"{tablet_id}.ply"),
                )
            )
    rows = assign_splits(rows, config.dataset.split_fractions, config.dataset.seed)
    validate_manifest(rows, config.class_names)
    check_split_isolation(rows)
    write_manifest(rows, config.paths.manifest)
    return rows


@dataclass
class PrepareReport:
    cached: list[str] = field(default_factory=list)
    already_cached: list[str] = field(default_factory=list)
    invalid: dict[str, str] = field(default_factory=dict)
    skipped_known_invalid: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # transient errors, retried next run

    def summary(self) -> str:
        return (
            f"newly cached: {len(self.cached)}, already cached: {len(self.already_cached)}, "
            f"invalid: {len(self.invalid)}, failed (retry later): {len(self.failed)}, "
            f"previously logged invalid (skipped): {len(self.skipped_known_invalid)}"
        )


def _process_one(
    row: ManifestRow, config: ProjectConfig, cache: DepthMapCache, index: dict | None, keep_raw: bool
) -> str:
    """Download (if needed) and render one tablet. Returns the cache path; raises on failure."""
    scan_path = resolve_scan_path(row.local_path)
    if not scan_path.exists():
        if row.source != heicubeda.SOURCE_NAME or index is None:
            raise InvalidScanError(f"scan file not found: {scan_path}")
        if row.tablet_id not in index:
            raise InvalidScanError(f"{row.tablet_id} is not in the HeiCuBeDa zip index")
        heicubeda.download_scan(index[row.tablet_id], scan_path)
    maps = scan_to_face_depth_maps(
        scan_path, config.preprocessing.resolution, config.preprocessing.margin_fraction
    )
    cache.save(row.tablet_id, maps)
    # Only delete scans we can download again; never delete user-supplied local files.
    if not keep_raw and config.dataset.delete_raw_after_processing and row.source == heicubeda.SOURCE_NAME:
        scan_path.unlink(missing_ok=True)
    return str(cache.path(row.tablet_id))


def prepare_depth_maps(
    config: ProjectConfig,
    workers: int = 3,
    limit: int | None = None,
    retry_invalid: bool = False,
    tablet_ids: list[str] | None = None,
    keep_raw: bool = False,
) -> PrepareReport:
    """Ensure every manifest row has a cached depth map. Safe to interrupt and re-run.

    Test tablets are processed first, then val, then train, so a partial run
    still yields a usable evaluation set.
    """
    rows = read_manifest(config.paths.manifest)
    validate_manifest(rows, config.class_names)
    if tablet_ids:
        rows = [r for r in rows if r.tablet_id in set(tablet_ids)]
    cache = DepthMapCache(config.paths.processed_dir, config.preprocessing)
    known_invalid = set() if retry_invalid else cache.invalid_ids()
    needs_index = any(r.source == heicubeda.SOURCE_NAME for r in rows)
    index = heicubeda.load_zip_index(config.paths.zip_index) if needs_index else None

    report = PrepareReport()
    todo: list[ManifestRow] = []
    order = {"test": 0, "val": 1, "train": 2, "": 3}
    for row in sorted(rows, key=lambda r: (order[r.split], r.tablet_id)):
        if cache.has(row.tablet_id):
            report.already_cached.append(row.tablet_id)
        elif row.tablet_id in known_invalid:
            report.skipped_known_invalid.append(row.tablet_id)
        else:
            todo.append(row)
    if limit is not None:
        todo = todo[:limit]

    log.info("%d tablets to process (%s)", len(todo), report.summary())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_process_one, row, config, cache, index, keep_raw): row for row in todo}
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                future.result()
                report.cached.append(row.tablet_id)
                log.info("[%d/%d] cached %s (%s, %s)", done, len(todo), row.tablet_id, row.period, row.split)
            except InvalidScanError as error:
                # A problem with the scan itself: log it so later runs skip this tablet.
                cache.log_invalid(row.tablet_id, str(error))
                report.invalid[row.tablet_id] = str(error)
                log.warning("[%d/%d] INVALID %s: %s", done, len(todo), row.tablet_id, error)
            except Exception as error:  # noqa: BLE001 - network/disk errors: retry on next run
                report.failed[row.tablet_id] = f"{type(error).__name__}: {error}"
                log.error("[%d/%d] FAILED %s (will retry next run): %s", done, len(todo), row.tablet_id, error)
    return report
