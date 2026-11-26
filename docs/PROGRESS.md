# Progress

Update this file at the end of every work session: move **Last accomplishment** into the log,
write the new one, and refresh **Next** and **Blocked**.

## Status at a glance

| Milestone (from the brief) | Status |
|----------------------------|--------|
| 1. Verify data: download access, period labels, class counts, one real depth map | ✅ Done |
| 2. Rerunnable preprocessing: manifest, cache, resume, invalid-scan log | ✅ Done (full download running) |
| 3. Train + evaluate one model (tablet-level splits, baseline, bundle) | 🟡 Pipeline verified on real data (smoke run), full training pending |
| 4. Streamlit demo (held-out tablet, depth map, confidence, Grad-CAM) | ✅ Works end to end with the smoke model; will be re-checked with the final model |
| Stretch: `.ply` upload in the app | 🟡 Code done, untested on a real scan |
| Stretch: public hosting | ⬜ Not started |

## Last accomplishment

**2026-11-26, 21:16. Made data preparation robust after a crash and a corrupt download.**

* **What happened:** the first full `prepare` run (6 workers) crashed natively (exit code 5, the low byte of
  0xC0000005) after 127 tablets. One downloaded scan (HS_1105) had its first 48 MiB zeroed on disk and was wrongly
  logged as an invalid scan. Cause: this machine has 7.7 GB RAM, and 6 threads were loading meshes of up to 1.1 GB in Open3D at the same time.
* **Fixes:**
  1. Every download is re-read from disk and checked against the archive's CRC-32 and size before use.
  2. A damaged raw file is re-downloaded rather than logged as invalid.
  3. Downloads stay parallel, but only one mesh is loaded and ray-cast at a time.
* **Verified:** HS_1105 now downloads, passes the CRC check and renders. Audited all 130 cached maps (tablet extent 21-166 mm,
  thickness 7-36 mm). The only outlier, HS_1446, is a genuinely large 16.5 cm tablet per the dataset's own bounding box.
* Full `prepare` restarted with `--workers 4`.

### Previous: thin end-to-end flow verified on real scans

**2026-11-26, 20:30.**

* Ran a 3-epoch smoke training on the GPU with the tablets cached so far (23 train / 44 val). The bundle
  `artifacts/models/smoke_partial_data` was saved. This is a plumbing check only, *not* a result.
* `predict HS_1030` (a held-out test tablet) returned a structured prediction.
* Headless app test (`scripts/smoke_test_app.py`): selected HS_1030, clicked **Analyze**, and the page showed the
  predicted period, confidence and correctness banner. The depth map, shaded relief and Grad-CAM panels render correctly
  (`artifacts/previews/app_panels_HS_1030.png`).
* Fixed a Windows-specific download failure. Renaming a freshly downloaded scan can hit `WinError 32` while antivirus
  scans it, so the rename now retries. One tablet (HS_1035) hit this before the fix and will be retried on the next `prepare`.

### Previous: real HeiCuBeDa scans flow into cached depth maps

**2026-11-26, day 1.**

* Confirmed the scans can be downloaded *one tablet at a time*. The server honours HTTP Range
  requests, so the pipeline pulls a single `.ply` out of the 5-35 GB archives.
* Built a tablet → archive index for all 1,977 scans and matched it to the period labels (all 1,977 IDs join).
* Class counts in the full dataset: **399 Ur III, 307 Old Babylonian**. The manifest samples 150 per class
  with splits of 106 train / 22 val / 22 test per class.
* Generated clear depth maps from real scans (e.g. `artifacts/previews/HS_1030.png`, where individual wedges are visible).
* All 15 tests pass (depth maps, split isolation, inference output).
* The full 300-tablet `prepare` run is in progress: 36/300 cached at 20:02, about 3 tablets/min, no invalid scans so far.

## Next

1. Let `prepare` finish (about 1.5 h), then run it once more to retry HS_1035 and any other transient failures.
2. Run `summarize` and record the final class counts per split here.
3. `train`, then `evaluate`, and record test accuracy, macro F1 and the confusion matrix vs the majority baseline below.
4. Re-run `scripts/smoke_test_app.py` with the final model, and test the `.ply` upload with a kept scan in `data/raw/heicubeda/scans/`.
5. Optional experiments: both faces (`faces: [primary, secondary]`), resolution 384.
6. Stretch: public hosting (Streamlit Community Cloud needs the model weights hosted somewhere. Decide where.)

## Blocked / risks

* Nothing is blocked.
* Download time is the bottleneck (~40 GB for 300 tablets). Raw scans are deleted after caching to save disk.
* The primary face is chosen by a heuristic (more fine relief). Damaged surfaces sometimes win (e.g. HS_0205).

## Results

_No real-data metrics yet. Historical classification metrics are reported only for real labelled tablets on the test split._

| Run | Test tablets | Accuracy | Macro F1 | Majority baseline acc. | Notes |
|-----|--------------|----------|----------|------------------------|-------|
| – | – | – | – | – | – |

## Log

### 2026-11-26, day 1
* Inspected the workspace: empty, no prior notes. Python 3.12, git, and a GTX 1650 (4 GB) are available.
* Queried the heiDATA API: the 3D data is 20 zip parts (~340 GB) plus an 11 MB metadata JSON with CDLI periods.
* Wrote the remote-zip reader and HeiCuBeDa index, then the config, geometry, data, modeling, inference, CLI and app modules.
* Changed face selection from "flatter face" to "more fine relief" after the first real scan (fragment HS_0270e)
  picked its broken back as the primary face.
* Wrote the tests (synthetic meshes + a tiny CPU training run), docs, and `.gitignore`. Initialised git, no commits yet.
