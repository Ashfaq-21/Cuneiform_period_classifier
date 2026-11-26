# Data

Everything in this folder except this file and `manifest.csv` is git-ignored and can be
recreated with the commands in the main README.

```
data/
├── manifest.csv                  # one row per tablet (committed; small)
├── raw/heicubeda/
│   ├── HeiCuBeDa_B_Hilprecht_Database_240121.json   # metadata incl. CDLI period (11 MB)
│   ├── zip_index.json            # tablet_id -> archive + member (built by fetch-metadata)
│   └── scans/<tablet_id>.ply     # downloaded scans; deleted after caching by default
├── processed/<cache_key>/        # cached depth maps, one .npz per tablet
│   ├── <tablet_id>.npz           # primary + secondary face, float32 heights (mm), NaN background
│   ├── settings.json             # preprocessing settings that produced this cache
│   └── invalid_scans.csv         # tablet_id, reason, logged_at
└── uploads/                      # temporary files from the app's .ply upload
```

`<cache_key>` (e.g. `r256_c8a5742f7a`) encodes the resolution, margin and a renderer
version. Changing any of these writes a fresh cache instead of reusing stale maps.

## Source: HeiCuBeDa Hilprecht

Mara, H. & Bogacz, B. (2019). *HeiCuBeDa Hilprecht – Heidelberg Cuneiform Benchmark Dataset
for the Hilprecht Collection*. heiDATA, [doi:10.11588/data/IE8CCN](https://doi.org/10.11588/data/IE8CCN).
License: CC BY-SA 4.0.

* 1,977 tablet scans in 20 zip archives (~340 GB total, 5-35 GB each).
* The period label is the CDLI `period` field in the metadata JSON. Of 1,977 tablets,
  399 are `Ur III (ca. 2100-2000 BC)` and 307 are `Old Babylonian (...)`. 818 have no CDLI record.
* The server supports HTTP Range requests, so `prepare` extracts single scans from the
  archives without downloading a whole archive (see `src/cuneiform/data/remote_zip.py`).
* Scans are in millimetres, with 1-2 million vertices and 50-500 MB compressed each.

Tablet IDs are normalised to `HS_<4-digit number><suffix>`, e.g. `HS 768` → `HS_0768`
and `HS 0282A` → `HS_0282a`, so the metadata and the archive file names join on all 1,977 tablets.

## Using your own scans (no download)

1. Put triangle meshes (`.ply`; `.obj`/`.stl`/`.off` also load) anywhere, e.g. `data/raw/local/`.
   Point clouds without faces are rejected.
2. Write `data/manifest.csv` by hand:

   ```csv
   tablet_id,period,source,local_path,split
   MY_0001,Ur III,local,data/raw/local/MY_0001.ply,
   MY_0002,Old Babylonian,local,data/raw/local/MY_0002.ply,
   ```

   | column       | required | meaning |
   |--------------|----------|---------|
   | `tablet_id`  | yes      | unique per physical tablet (all views/faces share it) |
   | `period`     | yes      | must be a class name from `labels:` in `configs/prototype.yaml` |
   | `source`     | yes      | free text; `HeiCuBeDa` rows can be downloaded on demand |
   | `local_path` | yes      | path to the scan, absolute or relative to the repository root |
   | `split`      | no       | `train`/`val`/`test`; leave empty and it is assigned by tablet |

3. Assign splits and render depth maps. Local scans are never deleted.

   ```powershell
   .\.venv\Scripts\python.exe -m cuneiform.cli assign-splits
   .\.venv\Scripts\python.exe -m cuneiform.cli prepare
   ```
