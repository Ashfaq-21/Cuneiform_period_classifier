"""Access to the HeiCuBeDa Hilprecht dataset (heiDATA, doi:10.11588/data/IE8CCN).

The dataset has three pieces we use:

* ``HeiCuBeDa_B_Hilprecht_Database_240121.json``: per-tablet metadata, including
  the CDLI ``period`` string (our label source).
* ``HeiCuBeDa_C_3DData_with_MSII_and_FunctionValue_partNN.zip``: twenty archives
  holding one ``.ply`` mesh per tablet.
* A zip index we build ourselves (``zip_index.json``) mapping tablet IDs to the
  archive and member that contain the scan. Building it reads only the central
  directories, so it costs a few MB of traffic.

License: CC BY-SA 4.0. Cite Mara, Hubert & Bogacz, Bartosz (2019), heiDATA.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cuneiform.data.remote_zip import USER_AGENT, extract_remote_member, list_remote_zip

DATAVERSE_ACCESS_URL = "https://heidata.uni-heidelberg.de/api/access/datafile/{file_id}"
METADATA_FILE_ID = 4753
# Dataverse file IDs of the 3D archives, in part order 01..20.
MESH_ARCHIVE_FILE_IDS: dict[str, int] = {
    "part01": 2658, "part02": 2660, "part03": 2661, "part04": 2662, "part05": 2663,
    "part06": 2664, "part07": 2665, "part08": 2666, "part09": 2667, "part10": 2668,
    "part11": 2669, "part12": 2673, "part13": 2674, "part14": 2672, "part15": 2675,
    "part16": 2676, "part17": 2677, "part18": 2678, "part19": 2679, "part20": 2680,
}
SOURCE_NAME = "HeiCuBeDa"


@dataclass(frozen=True)
class TabletRecord:
    """One tablet from the HeiCuBeDa metadata."""

    tablet_id: str  # filesystem-safe, e.g. "HS_768"
    hs_number: str  # museum number as written in the metadata, e.g. "HS 768"
    period_raw: str  # CDLI period string, "" if unknown
    cdli_number: str


_HS_PATTERN = re.compile(r"HS[ _]0*(\d+)(.*)", re.IGNORECASE)


def canonical_tablet_id(name: str) -> str:
    """Normalise museum numbers / file stems to one ID: ``"HS 768"``, ``"HS_0768"`` -> ``"HS_0768"``.

    Suffixes are kept, lower-cased: ``"HS 0282A"`` -> ``"HS_0282a"``, ``"HS_0246_1"`` -> ``"HS_0246_1"``.
    """
    match = _HS_PATTERN.fullmatch(name.strip())
    if not match:
        return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", match.group(2)).lower()
    return f"HS_{int(match.group(1)):04d}{suffix}"


def _stem_before_gigamesh_suffix(filename: str) -> str:
    """``HS_768_GMOCF_r1.50_n4_v512.ply`` -> ``HS_768``."""
    return Path(filename.replace("\\", "/")).name.split("_GMO")[0]


def archive_url(part: str) -> str:
    return DATAVERSE_ACCESS_URL.format(file_id=MESH_ARCHIVE_FILE_IDS[part])


def download_metadata(destination: Path) -> Path:
    """Download the metadata JSON (11 MB) unless it already exists."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATAVERSE_ACCESS_URL.format(file_id=METADATA_FILE_ID), headers={"User-Agent": USER_AGENT}
    )
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=300) as response, open(partial, "wb") as sink:
        sink.write(response.read())
    partial.replace(destination)
    return destination


def load_metadata(path: Path) -> list[TabletRecord]:
    """Parse the metadata JSON into :class:`TabletRecord` objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for hs_number, entry in raw.items():
        cdli = entry.get("from_cdli_search_view") or entry.get("from_cdli_archival_view") or {}
        # The PLY file stem is the most reliable join key with the archives.
        ply_name = (entry.get("from_ply_filename") or {}).get("filename") or hs_number
        records.append(
            TabletRecord(
                tablet_id=canonical_tablet_id(_stem_before_gigamesh_suffix(ply_name)),
                hs_number=hs_number,
                period_raw=(cdli.get("period") or "").strip(),
                cdli_number=str((entry.get("from_hilprecht3d") or {}).get("cdli_number") or ""),
            )
        )
    return records


def tablet_id_from_member(member_name: str) -> str | None:
    """Extract ``HS_0768`` from e.g. ``.../HS_0768_GMOCF_r1.50_n4_v512.ply``."""
    stem = _stem_before_gigamesh_suffix(member_name)
    return canonical_tablet_id(stem) if stem.upper().startswith("HS") else None


def build_zip_index(index_path: Path, parts: list[str] | None = None) -> dict[str, dict]:
    """List every archive's central directory and save ``{tablet_id: {part, member, ...}}``.

    Resumable: parts already present in the index file are skipped.
    """
    parts = parts or list(MESH_ARCHIVE_FILE_IDS)
    state = json.loads(index_path.read_text()) if index_path.exists() else {"parts": {}, "tablets": {}}
    for part in parts:
        if part in state["parts"]:
            continue
        members = list_remote_zip(archive_url(part))
        state["parts"][part] = len(members)
        for member in members:
            if not member["name"].lower().endswith(".ply"):
                continue
            tablet_id = tablet_id_from_member(member["name"])
            if tablet_id:
                state["tablets"][tablet_id] = {"part": part, **member}
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(state, indent=1))  # checkpoint after each part
        print(f"indexed {part}: {len(members)} members", flush=True)
    return state["tablets"]


def load_zip_index(index_path: Path) -> dict[str, dict]:
    """Load the index as ``{canonical tablet_id: {part, name, compressed_size, size}}``."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"Zip index not found at {index_path}. Run: python -m cuneiform.cli fetch-metadata"
        )
    tablets = json.loads(index_path.read_text())["tablets"]
    return {tablet_id_from_member(entry["name"]) or key: entry for key, entry in tablets.items()}


def download_scan(entry: dict, destination: Path) -> Path:
    """Extract one tablet's .ply from its remote archive (skips if already present)."""
    if destination.exists():
        return destination
    return extract_remote_member(archive_url(entry["part"]), entry["name"], destination)
