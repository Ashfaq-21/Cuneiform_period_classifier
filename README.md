# 3D Cuneiform Tablet Period Classifier

Predicts the historical period of a cuneiform tablet, **Ur III** (ca. 2100-2000 BC) or
**Old Babylonian** (ca. 1900-1600 BC), from a real 3D scan. Each scan is oriented
automatically, rendered as an orthographic **depth map** of one face, and classified
by a fine-tuned **ResNet18**. A local **Streamlit** app shows the depth map, predicted
period, model confidence, and a **Grad-CAM** overlay of the regions that influenced the prediction.

Data: [HeiCuBeDa Hilprecht](https://doi.org/10.11588/data/IE8CCN) (1,977 scans,
CC BY-SA 4.0), with period labels from CDLI in the dataset's metadata. See [data/README.md](data/README.md).

**Docs:** [progress](docs/PROGRESS.md) · [overview & purpose](docs/OVERVIEW.md) ·
[architecture](docs/ARCHITECTURE.md) · [how to run](docs/RUNNING.md)

<!-- RESULTS -->

## How it works

```
.ply scan ──► PCA pose ──► Open3D ray casting ──► depth map (256², mm) ──► cache (.npz)
                                                                        │
             depth / relief / mask channels  ◄──────────────────────────┘
                         │                         (same function in training and inference)
                         ▼
                 ResNet18 (ImageNet init) ──► P(period)  +  Grad-CAM on layer4
```

1. **Orientation** ([geometry/orientation.py](src/cuneiform/geometry/orientation.py)):
   centre the mesh, then align the thinnest principal axis with the view direction and the longest with image "up".
2. **Depth map** ([geometry/depth_map.py](src/cuneiform/geometry/depth_map.py)): one
   orthographic ray per pixel down onto each large face. Both faces use the same scale.
   The face with more fine relief (usually the more densely inscribed side) is `primary`.
   The prototype uses only `primary`, per `preprocessing.faces` in the config.
3. **Model input** ([geometry/representation.py](src/cuneiform/geometry/representation.py)):
   three channels: normalised height (shape), local relief (height minus a Gaussian-smoothed surface, which captures wedges), and tablet mask.
4. **Training** ([modeling/train.py](src/cuneiform/modeling/train.py)): ResNet18 with ImageNet weights,
   AdamW with a cosine schedule, augmentation by 180° and ±10° rotation and ±10% scale (no mirroring),
   a fixed seed, and the epoch with the best **validation** macro F1 kept.
5. **Evaluation** ([modeling/evaluate.py](src/cuneiform/modeling/evaluate.py)): accuracy, macro F1 and a confusion
   matrix on the **test** split, against a majority-class baseline, one prediction per tablet.

Splits are assigned **per tablet ID**, stratified by period, with a fixed seed. All faces
and views of a tablet share one split. Test tablets are never used for training or model selection.

## Setup (Windows PowerShell)

Requires Python 3.10-3.12 (tested on 3.12.10) and about 2 GB disk for the environment.
Run everything from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

This installs the CPU build of PyTorch, and every step below works on CPU.

### Optional: CUDA (NVIDIA GPU)

Replace the CPU PyTorch with a CUDA build *after* the step above, then check it:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`training.device: auto` in the config uses the GPU when one is visible. Set it to `cpu` to force CPU.

## Run the pipeline

```powershell
# 1. Metadata + remote index of the 20 scan archives (~15 MB of traffic, ~3 min)
.\.venv\Scripts\python.exe -m cuneiform.cli fetch-metadata

# 2. Sample up to 150 tablets per period and assign tablet-level splits -> data/manifest.csv
.\.venv\Scripts\python.exe -m cuneiform.cli build-manifest

# 3. Download each scan (range request into the archive), render depth maps, cache them.
#    Resumable: re-run the same command after an interruption. ~40 GB of traffic, about 2 hours at 3-10 MB/s.
#    Raw scans are deleted after caching (dataset.delete_raw_after_processing).
.\.venv\Scripts\python.exe -m cuneiform.cli prepare --workers 6
.\.venv\Scripts\python.exe -m cuneiform.cli summarize            # class counts per split + cache status
.\.venv\Scripts\python.exe -m cuneiform.cli preview HS_1030      # PNG of one tablet's depth maps

# 4. Train (also writes a validation report), then evaluate once on the held-out test split
.\.venv\Scripts\python.exe -m cuneiform.cli train
.\.venv\Scripts\python.exe -m cuneiform.cli evaluate

# 5. Tests (synthetic meshes, ~30 s, no download needed)
.\.venv\Scripts\python.exe -m pytest

# 6. Demo app at http://localhost:8501
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Other commands: `predict HS_1030` or `predict --ply path\to\scan.ply` print a JSON prediction.
`assign-splits` fills empty splits in a hand-written manifest. `--help` works on every command.

**Using your own scans** instead of downloading: see [data/README.md](data/README.md#using-your-own-scans-no-download).

## Outputs

| Path | Contents |
|------|----------|
| `data/manifest.csv` | tablet_id, period, source, local_path, split |
| `data/processed/<cache_key>/` | cached depth maps and `invalid_scans.csv` (scans that failed, with reasons) |
| `artifacts/models/<run>/model.pt` | weights |
| `artifacts/models/<run>/bundle.json` | class names, full config incl. preprocessing, cache key, training history, metrics |
| `artifacts/models/<run>/metrics_{val,test}.json`, `confusion_*.png` | evaluation reports, incl. per-tablet predictions |
| `artifacts/models/LATEST` | name of the bundle the app and `predict` load |
| `artifacts/reports/data_summary.json` | class counts per split |

Scans, caches and weights are git-ignored.

## Project layout

```
configs/prototype.yaml   one validated config: paths, labels, depth-map and training settings
src/cuneiform/
  config.py              pydantic schema + load_config()
  data/                  HeiCuBeDa access (remote zip), manifest, tablet-level splits, cache, prepare
  geometry/              mesh loading, orientation, depth maps, model-input channels, synthetic meshes
  modeling/              ResNet18, dataset, training, evaluation, Grad-CAM, model bundles
  inference.py           Predictor: bundle -> structured Prediction (used by app and CLI)
  visualization.py       colour maps and overlays shared by app and CLI
  cli.py                 thin argparse wrappers
app/app.py               Streamlit UI; calls cuneiform.inference only
tests/                   depth maps, split isolation, inference contract (synthetic data)
```

### Extension points

* **New period / label set**: edit `labels:` in the config (class name → exact CDLI period strings), then
  `build-manifest`, `prepare` and `train`. The class order in the config is the model's output order and is saved in the bundle.
* **New surface representation** (normals, curvature, MSII values): add a channel function in
  `geometry/representation.py` and its name to `InputChannel` in `config.py`. Training and inference both pick it up.
* **Different rendering**: change `geometry/depth_map.py` and bump `version` in
  `PreprocessingConfig.cache_key()` so old caches are not reused.
* **Both faces / multi-view**: set `preprocessing.faces: [primary, secondary]`. The dataset yields one sample per face,
  splits stay per tablet, and evaluation and inference average probabilities per tablet.
* **Another architecture**: add it to `modeling/model.py:build_model` and to the `model` literal in `TrainingConfig`.

## Limitations

* **Two classes, one collection.** All scans come from the Hilprecht Collection (mostly Nippur). The model may learn
  collection- or site-specific cues, and results do not show that it generalises to other collections.
* **Labels are catalogue metadata** (CDLI `period`), not re-verified.
* **Confounds.** Old Babylonian tablets in this collection are larger on average. Depth maps are scaled to fit the frame,
  so absolute size is removed, but proportions, fragment shapes and wedge size relative to the tablet still carry
  size information. Fragments are included.
* **Face selection is a heuristic.** Obverse and reverse are not known from geometry alone, so "primary" is the face with more fine relief,
  which can be a damaged surface.
* **Grad-CAM** shows where the network's evidence came from. It is not an explanation a scholar would give.
* **Confidence** is a softmax probability, not a calibrated probability.
