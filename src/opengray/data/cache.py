"""On-disk cache for cases: one uncompressed ``.npz`` plus a ``.json`` sidecar per case.

Layout::

    <cache_dir>/<cohort>/<case_id>.npz    arrays (feasible_idx, D as CSR parts, masks, dose)
    <cache_dir>/<cohort>/<case_id>.json   everything else (names, volumes, prescriptions, provenance)
    <cache_dir>/<cohort>/manifest.json    per-case stats and source checksums written by ingest

Uncompressed npz is chosen for load speed (a 100-case cohort loads in well under a second per
case from SSD); the compression that the source archive used is not worth repeating.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from opengray.data.base import BeamletTable, Case, Structure

CACHE_FORMAT_VERSION = 1


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def case_paths(cache_dir: Path, cohort: str, case_id: str) -> tuple[Path, Path]:
    d = Path(cache_dir) / cohort
    return d / f"{case_id}.npz", d / f"{case_id}.json"


def write_case(case: Case, cache_dir: Path) -> tuple[Path, Path]:
    npz_path, json_path = case_paths(cache_dir, case.cohort, case.case_id)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    D = case.D
    arrays: dict[str, np.ndarray] = {
        "feasible_idx": np.asarray(case.feasible_idx, dtype=np.int64),
        "voxel_volume_cc": np.asarray(case.voxel_volume_cc, dtype=np.float64),
        "D_data": D.data.astype(np.float64, copy=False),
        "D_indices": D.indices.astype(np.int32, copy=False),
        "D_indptr": D.indptr.astype(np.int64, copy=False),
        "D_shape": np.asarray(D.shape, dtype=np.int64),
    }
    if case.reference_dose is not None:
        arrays["reference_dose"] = np.asarray(case.reference_dose, dtype=np.float64)
    if case.beamlets is not None:
        arrays["beamlet_row"] = np.asarray(case.beamlets.row, dtype=np.int32)
        arrays["beamlet_col"] = np.asarray(case.beamlets.col, dtype=np.int32)
        arrays["beamlet_beam"] = np.asarray(case.beamlets.beam, dtype=np.int32)
    for name, s in case.structures.items():
        arrays[f"struct__{name}"] = np.asarray(s.mask_idx, dtype=np.int32)

    with open(npz_path, "wb") as f:
        np.savez(f, **arrays)

    meta: dict[str, Any] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "case_id": case.case_id,
        "cohort": case.cohort,
        "grid_shape": list(case.grid_shape) if case.grid_shape else None,
        "voxel_size_mm": list(case.voxel_size_mm) if case.voxel_size_mm else None,
        "structures": {
            name: {"raw_name": s.raw_name, "n_outside": s.n_outside, "volume_cc": s.volume_cc}
            for name, s in case.structures.items()
        },
        "prescriptions": case.prescriptions,
        "beam_geometry": case.beam_geometry,
        "provenance": case.provenance,
    }
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return npz_path, json_path


def read_case(cache_dir: Path, cohort: str, case_id: str) -> Case:
    npz_path, json_path = case_paths(cache_dir, cohort, case_id)
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"case {case_id} not in cache at {npz_path.parent}")
    meta = json.loads(json_path.read_text())
    if meta.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"cache format {meta.get('cache_format_version')} != {CACHE_FORMAT_VERSION}; re-run ingest"
        )
    with np.load(npz_path) as z:
        shape = tuple(int(x) for x in z["D_shape"])
        D = sp.csr_matrix((z["D_data"], z["D_indices"], z["D_indptr"]), shape=shape)
        feasible_idx = z["feasible_idx"]
        voxel_volume_cc = z["voxel_volume_cc"]
        reference_dose = z["reference_dose"] if "reference_dose" in z.files else None
        beamlets = None
        if "beamlet_row" in z.files:
            beamlets = BeamletTable(row=z["beamlet_row"], col=z["beamlet_col"], beam=z["beamlet_beam"])
        structures: dict[str, Structure] = {}
        for name, info in meta["structures"].items():
            structures[name] = Structure(
                name=name,
                raw_name=info["raw_name"],
                mask_idx=z[f"struct__{name}"],
                n_outside=int(info["n_outside"]),
                volume_cc=float(info["volume_cc"]),
            )
    return Case(
        case_id=meta["case_id"],
        cohort=meta["cohort"],
        grid_shape=tuple(meta["grid_shape"]) if meta.get("grid_shape") else None,
        voxel_size_mm=tuple(meta["voxel_size_mm"]) if meta.get("voxel_size_mm") else None,
        feasible_idx=feasible_idx,
        voxel_volume_cc=voxel_volume_cc,
        D=D,
        structures=structures,
        prescriptions={k: float(v) for k, v in meta["prescriptions"].items()},
        reference_dose=reference_dose,
        beamlets=beamlets,
        beam_geometry=meta.get("beam_geometry", {}),
        provenance=meta.get("provenance", {}),
    )


def list_cached_cases(cache_dir: Path, cohort: str) -> list[str]:
    d = Path(cache_dir) / cohort
    if not d.exists():
        return []
    ids = [p.stem for p in d.glob("*.npz") if (d / f"{p.stem}.json").exists()]
    return sorted(ids, key=_natural_key)


def _natural_key(s: str) -> tuple:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def read_manifest(cache_dir: Path, cohort: str) -> dict[str, Any]:
    p = Path(cache_dir) / cohort / "manifest.json"
    if not p.exists():
        return {"cohort": cohort, "cases": {}}
    return json.loads(p.read_text())


def write_manifest(cache_dir: Path, cohort: str, manifest: dict[str, Any]) -> Path:
    p = Path(cache_dir) / cohort / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return p
