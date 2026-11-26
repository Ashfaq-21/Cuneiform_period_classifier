"""Final evaluation on the held-out test split, against a majority-class baseline.

Predictions are made per tablet (probabilities averaged over the configured faces).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from cuneiform.config import ProjectConfig
from cuneiform.modeling.bundle import load_bundle, update_bundle_metadata
from cuneiform.modeling.dataset import build_dataset, load_split_rows
from cuneiform.modeling.model import resolve_device
from cuneiform.modeling.train import predict_probabilities

log = logging.getLogger(__name__)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> dict:
    indices = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", labels=indices, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist(),
    }


def _plot_confusion(matrix: list[list[int]], class_names: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.5, 4))
    data = np.array(matrix)
    axis.imshow(data, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted period")
    axis.set_ylabel("True period")
    axis.set_title(title)
    for (i, j), value in np.ndenumerate(data):
        axis.text(j, i, str(value), ha="center", va="center",
                  color="white" if value > data.max() / 2 else "black", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def evaluate_bundle(bundle_dir: Path, config: ProjectConfig, split: str = "test") -> dict:
    """Evaluate a bundle on ``split`` and write ``metrics_<split>.json`` + a confusion-matrix PNG.

    Data paths come from ``config`` (this machine); preprocessing and labels come from the bundle.
    """
    device = resolve_device(config.training.device)
    bundle = load_bundle(bundle_dir, device)
    # Use the bundle's preprocessing/labels with this machine's paths.
    eval_config = bundle.config.model_copy(update={"paths": config.paths})
    splits, missing = load_split_rows(eval_config)
    rows = splits[split]
    if not rows:
        raise RuntimeError(f"No cached tablets in the '{split}' split ({missing[split]} without depth maps).")
    dataset = build_dataset(rows, eval_config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    probabilities, labels = predict_probabilities(bundle.model, loader, device)

    # Average face-level probabilities into one prediction per tablet.
    per_tablet: dict[str, list[np.ndarray]] = {}
    tablet_label: dict[str, int] = {}
    for sample, probs, label in zip(dataset.samples, probabilities, labels):
        per_tablet.setdefault(sample.tablet_id, []).append(probs)
        tablet_label[sample.tablet_id] = int(label)
    tablet_ids = sorted(per_tablet)
    tablet_probs = np.stack([np.mean(per_tablet[t], axis=0) for t in tablet_ids])
    y_true = np.array([tablet_label[t] for t in tablet_ids])
    y_pred = tablet_probs.argmax(axis=1)

    class_names = bundle.class_names
    train_counts = Counter(r.period for r in splits["train"])
    majority_class = max(class_names, key=lambda c: (train_counts.get(c, 0), -class_names.index(c)))
    baseline_pred = np.full_like(y_true, class_names.index(majority_class))

    metrics = {
        "split": split,
        "num_tablets": len(tablet_ids),
        "tablets_without_depth_map": missing[split],
        "class_names": class_names,
        "support": {c: int((y_true == i).sum()) for i, c in enumerate(class_names)},
        "model": classification_metrics(y_true, y_pred, class_names),
        "majority_baseline": {"predicts": majority_class, **classification_metrics(y_true, baseline_pred, class_names)},
        "predictions": [
            {"tablet_id": t, "true": class_names[y_true[i]], "predicted": class_names[y_pred[i]],
             "confidence": float(tablet_probs[i].max())}
            for i, t in enumerate(tablet_ids)
        ],
    }
    (bundle_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
    _plot_confusion(metrics["model"]["confusion_matrix"], class_names, bundle_dir / f"confusion_{split}.png",
                    f"{split} split, n={len(tablet_ids)} tablets")
    update_bundle_metadata(bundle_dir, **{f"metrics_{split}": {k: metrics[k] for k in ("num_tablets", "model", "majority_baseline")}})
    log.info("%s: model acc %.3f macroF1 %.3f | majority baseline (%s) acc %.3f macroF1 %.3f",
             split, metrics["model"]["accuracy"], metrics["model"]["macro_f1"], majority_class,
             metrics["majority_baseline"]["accuracy"], metrics["majority_baseline"]["macro_f1"])
    return metrics
