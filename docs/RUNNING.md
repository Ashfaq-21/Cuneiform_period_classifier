# How to run (Windows PowerShell)

Run every command from the repository root
(`C:\Users\Admin\Documents\personal projects\cuneiform`). Requires Python 3.10-3.12.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

This gives a CPU-only PyTorch. Everything works on CPU, and training is just slower.

**Optional, NVIDIA GPU:** swap in a CUDA build and check it.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`training.device: auto` in `configs/prototype.yaml` uses the GPU when one is visible. Set it to `cpu` to force CPU.

## 2. Prepare data

```powershell
# Metadata + index of the 20 remote scan archives (~15 MB download, a few minutes)
.\.venv\Scripts\python.exe -m cuneiform.cli fetch-metadata

# Choose up to 150 tablets per period and split them by tablet -> data/manifest.csv
.\.venv\Scripts\python.exe -m cuneiform.cli build-manifest

# Download scans one by one, render depth maps, cache them. ~40 GB of traffic, about 2 h.
# Safe to stop (Ctrl+C) and re-run: finished tablets are skipped.
.\.venv\Scripts\python.exe -m cuneiform.cli prepare --workers 4

# Check class counts per split and how many depth maps exist
.\.venv\Scripts\python.exe -m cuneiform.cli summarize

# Look at one tablet's depth maps (writes artifacts\previews\HS_1030.png)
.\.venv\Scripts\python.exe -m cuneiform.cli preview HS_1030
```

`--workers` sets parallel *downloads* only. Mesh processing runs one scan at a time, because a large scan needs several GB of RAM.
Every download is checked against the archive's CRC before use.

Useful `prepare` options: `--limit 10` (try a few first), `--tablet HS_1030` (one tablet),
`--keep-raw` (keep the downloaded `.ply`), `--retry-invalid` (retry scans in `invalid_scans.csv`).

**Own scans instead of downloading:** write `data/manifest.csv` by hand (format in
[data/README.md](../data/README.md#using-your-own-scans-no-download)), then:

```powershell
.\.venv\Scripts\python.exe -m cuneiform.cli assign-splits
.\.venv\Scripts\python.exe -m cuneiform.cli prepare
```

## 3. Train

```powershell
.\.venv\Scripts\python.exe -m cuneiform.cli train
```

This saves `artifacts\models\<run>\` (weights + `bundle.json`), points `artifacts\models\LATEST` at it, and writes a
validation report. Options: `--epochs 5` (quick run), `--run-name my_run`.

## 4. Evaluate (final numbers, test split)

```powershell
.\.venv\Scripts\python.exe -m cuneiform.cli evaluate
```

This prints accuracy, macro F1 and the confusion matrix next to the majority-class baseline, and writes
`metrics_test.json` and `confusion_test.png` into the bundle folder. Use `--run <name>` for a specific bundle.

## 5. Predict from the command line

```powershell
.\.venv\Scripts\python.exe -m cuneiform.cli predict HS_1030
.\.venv\Scripts\python.exe -m cuneiform.cli predict --ply data\raw\heicubeda\scans\HS_1030.ply
```

## 6. Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The tests use synthetic meshes and need no download (~40 s). They cover depth-map generation, tablet-level split
isolation, and the inference output contract.

## 7. Launch the demo app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Then open <http://localhost:8501>. Pick a held-out tablet, click **Analyze**, and read the depth map, predicted
period, confidence and Grad-CAM overlay. The second tab accepts a `.ply` upload. If no model has been trained yet, the
app says so and shows the commands to fix it.

To use a different config: `$env:CUNEIFORM_CONFIG = "configs\other.yaml"` before launching.

## Troubleshooting

| Message | Fix |
|---------|-----|
| `Manifest not found` | Run `build-manifest` (or write the manifest by hand) |
| `Zip index not found` | Run `fetch-metadata` |
| `No cached 'train' tablets` | Run `prepare`, then check with `summarize` |
| `No trained model found` | Run `train` |
| `training.device is 'cuda' but PyTorch cannot see a GPU` | Install the CUDA build above or set `device: cpu` |
| `FAILED ... (will retry next run)` in the prepare log | Network hiccup: re-run `prepare` |
