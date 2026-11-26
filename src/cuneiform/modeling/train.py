"""Fine-tune the classifier. Model selection uses the validation split only."""

from __future__ import annotations

import copy
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from cuneiform.config import ProjectConfig
from cuneiform.modeling.bundle import models_root, save_bundle
from cuneiform.modeling.dataset import build_dataset, load_split_rows
from cuneiform.modeling.model import build_model, resolve_device
from cuneiform.modeling.transforms import augment

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probabilities[N, K], labels[N])`` in dataset order."""
    model.eval()
    probabilities, labels = [], []
    for inputs, targets in loader:
        probabilities.append(torch.softmax(model(inputs.to(device)), dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def train_model(config: ProjectConfig, run_name: str | None = None, epochs: int | None = None) -> Path:
    """Train, keep the epoch with the best validation macro F1, and save a bundle. Returns its directory."""
    settings = config.training
    generator = seed_everything(settings.seed)
    device = resolve_device(settings.device)
    splits, missing = load_split_rows(config)
    for split in ("train", "val"):
        if not splits[split]:
            raise RuntimeError(
                f"No cached '{split}' tablets. Run 'python -m cuneiform.cli prepare' first "
                f"({missing[split]} {split} rows in the manifest have no depth map yet)."
            )
    train_set, val_set = build_dataset(splits["train"], config), build_dataset(splits["val"], config)
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True,
                              num_workers=settings.num_workers, generator=generator)
    val_loader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers)

    num_classes = len(config.class_names)
    counts = np.bincount([s.label_index for s in train_set.samples], minlength=num_classes)
    if (counts == 0).any():
        raise RuntimeError(f"Training split is missing a class: counts {dict(zip(config.class_names, counts))}")
    class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)

    model = build_model(settings.model, num_classes, len(config.preprocessing.input_channels), settings.pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    total_epochs = epochs or settings.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    log.info("device=%s train=%d val=%d samples, class counts %s",
             device, len(train_set), len(val_set), dict(zip(config.class_names, counts.tolist())))
    history, best = [], {"val_macro_f1": -1.0, "val_loss": float("inf"), "epoch": -1}
    best_state = None
    for epoch in range(1, total_epochs + 1):
        started = time.time()
        model.train()
        running_loss, seen = 0.0, 0
        for inputs, targets in train_loader:
            inputs = augment(inputs, generator).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
            seen += len(targets)
        scheduler.step()

        probabilities, labels = predict_probabilities(model, val_loader, device)
        val_loss = float(nn.functional.nll_loss(
            torch.log(torch.from_numpy(probabilities).clamp_min(1e-8)), torch.from_numpy(labels),
            weight=class_weights.cpu()))
        predictions = probabilities.argmax(axis=1)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_accuracy": float((predictions == labels).mean()),
            "val_macro_f1": float(f1_score(labels, predictions, average="macro", labels=list(range(num_classes)))),
            "seconds": round(time.time() - started, 1),
        }
        history.append(record)
        log.info("epoch %2d  train_loss %.4f  val_loss %.4f  val_acc %.3f  val_macroF1 %.3f  (%.0fs)",
                 epoch, record["train_loss"], val_loss, record["val_accuracy"], record["val_macro_f1"], record["seconds"])
        improved = (record["val_macro_f1"], -val_loss) > (best["val_macro_f1"], -best["val_loss"])
        if improved:
            best = {k: record[k] for k in ("epoch", "val_macro_f1", "val_loss", "val_accuracy")}
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    run_name = run_name or datetime.now().strftime("resnet18_%Y%m%d_%H%M%S")
    directory = models_root(config) / run_name
    save_bundle(directory, model, config, {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "best_epoch": best,
        "history": history,
        "data": {split: len(rows) for split, rows in splits.items()},
        "train_tablet_ids": sorted(r.tablet_id for r in splits["train"]),
    })
    log.info("saved model bundle to %s (best epoch %d, val macro F1 %.3f)", directory, best["epoch"], best["val_macro_f1"])
    return directory
