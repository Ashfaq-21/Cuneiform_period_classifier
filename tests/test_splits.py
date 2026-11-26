from collections import Counter

import pytest

from cuneiform.data.manifest import ManifestError, ManifestRow, read_manifest, validate_manifest, write_manifest
from cuneiform.data.splits import assign_splits, check_split_isolation
from cuneiform.modeling.dataset import Sample

FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _rows(n_per_class: int = 40) -> list[ManifestRow]:
    return [
        ManifestRow(f"{prefix}_{i:03d}", period, "test", f"scans/{prefix}_{i:03d}.ply")
        for prefix, period in (("A", "Ur III"), ("B", "Old Babylonian"))
        for i in range(n_per_class)
    ]


def test_every_tablet_gets_exactly_one_split_and_classes_are_stratified():
    rows = assign_splits(_rows(), FRACTIONS, seed=1)
    check_split_isolation(rows)
    per_class = Counter((r.period, r.split) for r in rows)
    for period in ("Ur III", "Old Babylonian"):
        assert per_class[(period, "test")] == 6 and per_class[(period, "val")] == 6
        assert per_class[(period, "train")] == 28


def test_faces_of_one_tablet_never_cross_splits():
    rows = assign_splits(_rows(), FRACTIONS, seed=1)
    split_of = {r.tablet_id: r.split for r in rows}
    # The dataset expands each tablet into one sample per face; all inherit the tablet's split.
    samples = [Sample(r.tablet_id, face, 0) for r in rows for face in ("primary", "secondary")]
    splits_per_tablet: dict[str, set[str]] = {}
    for sample in samples:
        splits_per_tablet.setdefault(sample.tablet_id, set()).add(split_of[sample.tablet_id])
    assert all(len(s) == 1 for s in splits_per_tablet.values())
    train_ids = {r.tablet_id for r in rows if r.split == "train"}
    test_ids = {r.tablet_id for r in rows if r.split == "test"}
    assert train_ids.isdisjoint(test_ids)


def test_check_split_isolation_detects_leaks():
    leaking = [ManifestRow("A_001", "Ur III", "x", "p", "train"), ManifestRow("A_001", "Ur III", "x", "p", "test")]
    with pytest.raises(ValueError, match="more than one split"):
        check_split_isolation(leaking)


def test_splits_are_deterministic_and_independent_of_row_order():
    rows = _rows()
    first = {r.tablet_id: r.split for r in assign_splits(rows, FRACTIONS, seed=7)}
    second = {r.tablet_id: r.split for r in assign_splits(list(reversed(rows)), FRACTIONS, seed=7)}
    assert first == second


def test_adding_tablets_keeps_existing_assignments():
    original = assign_splits(_rows(20), FRACTIONS, seed=7)
    extra = [ManifestRow(f"C_{i}", "Ur III", "test", "p") for i in range(10)]
    grown = assign_splits(original + extra, FRACTIONS, seed=7)
    before = {r.tablet_id: r.split for r in original}
    assert all(r.split == before[r.tablet_id] for r in grown if r.tablet_id in before)
    assert all(r.split for r in grown)


def test_manifest_roundtrip_and_validation(tmp_path):
    rows = assign_splits(_rows(5), FRACTIONS, seed=0)
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    assert read_manifest(path) == rows
    validate_manifest(rows, ["Ur III", "Old Babylonian"])
    bad = rows + [ManifestRow("A_000", "Neo-Assyrian", "x", "p", "later")]
    with pytest.raises(ManifestError) as error:
        validate_manifest(bad, ["Ur III", "Old Babylonian"])
    message = str(error.value)
    assert "duplicate tablet_id" in message and "Neo-Assyrian" in message and "later" in message
