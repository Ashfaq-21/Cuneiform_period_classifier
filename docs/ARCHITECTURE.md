# Architecture: how it works and how it is structured

## The pipeline

```
 heiDATA (HTTP Range)          configs/prototype.yaml  (one validated config)
        │                                   │
        ▼                                   ▼
 fetch-metadata ──► metadata JSON + zip_index.json
        │
 build-manifest ──► data/manifest.csv   (tablet_id, period, source, local_path, split)
        │
 prepare ────────► per tablet: download .ply ─► orient ─► ray-cast both faces ─► cache .npz
        │                                                        (invalid scans → invalid_scans.csv)
 train ──────────► cached depth map ─► 3 input channels ─► ResNet18 ─► best epoch on VAL
        │                                                        └─► artifacts/models/<run>/ (bundle)
 evaluate ───────► bundle + TEST split ─► metrics_test.json, confusion_test.png
        │
 app / predict ──► Predictor (inference.py) ─► Prediction: period, confidence, depth map, Grad-CAM
```

### 1. Data access ([src/cuneiform/data/](../src/cuneiform/data/))

* `remote_zip.py`: a seekable file object backed by HTTP Range requests. Python's `zipfile` reads the remote
  central directory and streams out one member, so a 150 MB scan doesn't require a 20 GB download.
  Writes are atomic (`.part` file, then rename).
* `heicubeda.py`: dataset constants (Dataverse file IDs), metadata parsing, the archive index, and
  `canonical_tablet_id()` so that `HS 768` (metadata) and `HS_0768_GMOCF…ply` (archive) join.
* `manifest.py`: the manifest CSV schema, read/write, and validation that lists *every* problem.
* `splits.py`: stratified, seeded, **tablet-level** splits. Adding tablets later never moves existing ones.
  `check_split_isolation()` raises if a tablet appears in two splits.
* `cache.py`: depth maps stored as `processed/<cache_key>/<tablet_id>.npz`. The cache key hashes the
  settings that change rendering, so changed settings never reuse stale maps.
* `prepare.py`: builds the manifest and fills the cache with a thread pool. Order is test → val → train, so a
  partial run is still evaluable. It **resumes** by skipping cached tablets. Scan problems are logged permanently,
  while network errors are only reported and retried on the next run.

### 2. Geometry ([src/cuneiform/geometry/](../src/cuneiform/geometry/))

* `mesh_io.py`: Open3D loading plus validation (faces present, finite, non-degenerate). Failures raise
  `InvalidScanError` with a reason.
* `orientation.py`: PCA pose. The thinnest axis becomes the view direction (z) and the longest becomes image "up" (y).
* `depth_map.py`: Open3D `RaycastingScene`, one orthographic ray per pixel, for **both** large faces at the
  same scale. Output: float32 heights in mm, with `NaN` where no surface is hit. The face with more fine relief becomes `primary`.
* `representation.py`: the **only** place a depth map becomes model input. Channels: `depth` (normalised
  height), `relief` (height minus a Gaussian-smoothed surface, which captures wedges) and `mask`. It also provides hill-shading for display.
* `synthetic.py`: synthetic tablet meshes, used by tests only.

### 3. Modeling ([src/cuneiform/modeling/](../src/cuneiform/modeling/))

* `transforms.py`: `model_input()` (shared by training, evaluation and inference) and training-only
  augmentation: 180° and ±10° rotation, ±10% scale, never mirroring.
* `dataset.py`: one sample per (tablet, face) from the cache. Tensors are kept in memory.
* `model.py`: `build_model("resnet18")` with ImageNet weights and a new 2-class head, plus device selection.
* `train.py`: seeded training with AdamW, a cosine schedule and class-weighted loss. It keeps the epoch with the best
  **validation** macro F1 and saves a bundle.
* `evaluate.py`: per-tablet predictions (averaged over faces), metrics vs the majority baseline, a confusion-matrix PNG, and per-tablet predictions in JSON.
* `gradcam.py`: Grad-CAM on `layer4` for the predicted class.
* `bundle.py`: saves and loads `model.pt` + `bundle.json` (class order, full config, cache key, history, metrics).
  `artifacts/models/LATEST` names the default bundle.

### 4. Inference and interfaces

* [inference.py](../src/cuneiform/inference.py): `Predictor.from_latest()` loads a bundle and uses **the bundle's**
  preprocessing and labels with this machine's data paths. `predict_tablet(id)` uses the cache, and
  `predict_scan(path)` renders a raw `.ply` with identical settings. Both return a `Prediction` dataclass.
* [cli.py](../src/cuneiform/cli.py): thin argparse commands. Each calls one package function.
* [app/app.py](../app/app.py): Streamlit layout only. Everything goes through `Predictor`. If no model exists,
  it shows the commands to create one.

## Repository structure

```
cuneiform/
├── pyproject.toml            dependencies, `cuneiform` console script, pytest settings
├── README.md                 short intro + quick start
├── .gitignore                scans, caches, weights, venv excluded
├── .streamlit/config.toml    upload size limit (1 GB)
├── configs/prototype.yaml    paths, labels, dataset, preprocessing, training
├── docs/                     these documents
├── src/cuneiform/
│   ├── config.py             pydantic schema, load_config(), cache_key()
│   ├── data/                 remote_zip, heicubeda, manifest, splits, cache, prepare
│   ├── geometry/             mesh_io, orientation, depth_map, representation, synthetic
│   ├── modeling/             transforms, dataset, model, train, evaluate, gradcam, bundle
│   ├── inference.py          Predictor / Prediction
│   ├── visualization.py      colour maps, hill-shade, Grad-CAM overlay, preview PNGs
│   └── cli.py                command-line entry points
├── app/app.py                Streamlit demo
├── tests/                    test_depth_map, test_splits, test_inference
├── data/                     manifest.csv + README (everything else git-ignored)
└── artifacts/                models/ (ignored), previews/, reports/
```

## Design rules

* **One config.** Paths, labels, resolution and training settings are only in `configs/prototype.yaml`, validated
  by pydantic (`extra="forbid"`, so typos fail loudly).
* **One preprocessing path.** The same functions produce cached maps, training tensors and inference tensors.
  The bundle records the settings it was trained with.
* **Splits by tablet.** Faces and views of a tablet are never split apart, and the test split is used only by `evaluate`.
* **Thin edges.** The CLI and app contain no modelling logic.

## Extension points

| To add… | Change | Notes |
|---------|--------|-------|
| More periods | `labels:` in the config | Then `build-manifest`, `prepare`, `train`. Class order = output order. |
| A surface representation (normals, curvature, MSII) | a channel function in `geometry/representation.py` + `InputChannel` in `config.py` | Training and inference both pick it up |
| A different renderer | `geometry/depth_map.py` | Bump `version` in `PreprocessingConfig.cache_key()` |
| Multi-view | `preprocessing.faces: [primary, secondary]` | Per-tablet probability averaging is already implemented |
| Another model | `modeling/model.py:build_model` + `TrainingConfig.model` literal | Grad-CAM layer is in `gradcam_target_layer()` |
| Another dataset | write `data/manifest.csv` with `source: local` | See [data/README.md](../data/README.md) |
