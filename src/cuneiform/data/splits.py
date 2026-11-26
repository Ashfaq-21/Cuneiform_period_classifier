"""Deterministic, stratified train/val/test splits at the tablet level.

Splitting by ``tablet_id`` guarantees that every face or view of one tablet
lands in the same split, so the test set never contains a tablet seen in training.
"""

from __future__ import annotations

import random
from collections import defaultdict

from cuneiform.data.manifest import ManifestRow


def assign_splits(
    rows: list[ManifestRow],
    fractions: dict[str, float],
    seed: int,
    keep_existing: bool = True,
) -> list[ManifestRow]:
    """Return rows with ``split`` set, stratified by period.

    With ``keep_existing`` (the default), rows that already have a split keep it,
    so adding tablets later never moves an existing tablet into a different split.
    """
    rng = random.Random(seed)
    by_period: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        if not (keep_existing and row.split):
            by_period[row.period].append(row)

    assigned: dict[str, str] = {r.tablet_id: r.split for r in rows if keep_existing and r.split}
    for period in sorted(by_period):
        # Sort before shuffling so the result depends only on the IDs and the seed.
        tablets = sorted({r.tablet_id for r in by_period[period]})
        rng.shuffle(tablets)
        n = len(tablets)
        n_test = max(1, round(n * fractions["test"])) if n >= 3 else 0
        n_val = max(1, round(n * fractions["val"])) if n >= 3 else 0
        for index, tablet_id in enumerate(tablets):
            if index < n_test:
                assigned[tablet_id] = "test"
            elif index < n_test + n_val:
                assigned[tablet_id] = "val"
            else:
                assigned[tablet_id] = "train"
    return [row.with_split(assigned[row.tablet_id]) for row in rows]


def check_split_isolation(rows: list[ManifestRow]) -> None:
    """Raise if any tablet_id appears in more than one split."""
    splits_per_tablet: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_per_tablet[row.tablet_id].add(row.split)
    leaking = {tid: sorted(s) for tid, s in splits_per_tablet.items() if len(s) > 1}
    if leaking:
        raise ValueError(f"tablets present in more than one split: {dict(list(leaking.items())[:10])}")
