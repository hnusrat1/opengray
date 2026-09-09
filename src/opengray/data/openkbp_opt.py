"Import user-supplied OpenKBP-Opt archives into a checksummed sparse-matrix cache.\n\nThe importer retains source indexing, voxel dimensions, outside-mask structure voxels, and reference-dose provenance. Source data are obtained from the upstream authors rather than distributed with this package."

from __future__ import annotations

import re
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from opengray.data.base import BeamletTable, Case, CohortLoader, Structure
from opengray.data.cache import (
    list_cached_cases,
    read_case,
    read_manifest,
    sha256_file,
    write_case,
    write_manifest,
)
from opengray.physics.influence import restrict_rows
from opengray.physics.nomenclature import TARGET_PRESCRIPTION_GY, is_target_name, resolve

COHORT = "openkbp_opt"
GRID_SHAPE: tuple[int, int, int] = (128, 128, 128)
ARCHIVE_PREFIX = "open-kbp-opt-data/reference-plans/"
PATIENT_RE = re.compile(r"^pt_(\d+)$")

# Raw structure names in the source, in the reference code's channel order.
RAW_STRUCTURES: tuple[str, ...] = (
    "Brainstem",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Esophagus",
    "Larynx",
    "Mandible",
    "PTV56",
    "PTV63",
    "PTV70",
)
REQUIRED_FILES: tuple[str, ...] = (
    "dij.npz",
    "beamlet_indices.csv",
    "dose.csv",
    "possible_dose_mask.csv",
    "voxel_dimensions.csv",
)

BEAM_GEOMETRY: dict[str, Any] = {
    "n_beams": 9,
    "gantry_deg": [0, 40, 80, 120, 160, 200, 240, 280, 320],
    "gantry_assignment": "assumed beam k (0-based) at 40*k degrees; the archive stores only an angle index 1..9 (unverified against the source)",
    "beamlet_size_mm": [5.0, 5.0],
    "fluence_grid": [64, 64],
    "energy": "6 MV",
    "technique": "step-and-shoot IMRT, coplanar, equispaced",
    "dose_engine": "CERR IMRTP (pencil-beam class)",
    "units": "Gy per unit beamlet intensity, full course",
    "source": "Babier et al. PMB 2022 sec 2.1; Babier et al. Med Phys 2021 sec 2 (OpenKBP)",
}


def read_sparse_csv(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (indices, values) for an OpenKBP sparse CSV; values is None for a mask file."""
    df = pd.read_csv(path, index_col=0)
    indices = np.asarray(df.index.values, dtype=np.int64)
    if df.shape[1] == 0 or df.isnull().values.any():
        return indices, None
    return indices, np.asarray(df.iloc[:, 0].values, dtype=np.float64)


def read_voxel_dimensions(path: Path) -> tuple[float, float, float]:
    v = np.loadtxt(path, dtype=np.float64).ravel()
    if v.size != 3:
        raise ValueError(f"{path}: expected three voxel dimensions, got {v.size}")
    return float(v[0]), float(v[1]), float(v[2])


def read_beamlet_indices(path: Path) -> BeamletTable:
    df = pd.read_csv(path)
    expected = ["row", "column", "angle"]
    if list(df.columns) != expected:
        raise ValueError(f"{path}: expected columns {expected}, got {list(df.columns)}")
    angle = df["angle"].to_numpy(dtype=np.int64)
    if angle.min() < 1:
        raise ValueError(f"{path}: angle index must be 1-based")
    return BeamletTable(
        row=df["row"].to_numpy(dtype=np.int32),
        col=df["column"].to_numpy(dtype=np.int32),
        beam=(angle - 1).astype(np.int32),
    )


@dataclass
class LoadReport:
    case_id: str
    warnings: list[str] = field(default_factory=list)
    conversions: list[str] = field(default_factory=list)
    missing_structures: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def ensure_external(case: Case) -> Case:
    "Add External from dose-feasible voxels when absent.\n\nThe possible-dose mask is a proxy for a body contour. External carries the visible maximum-dose goal. It is derived at load time rather than stored in the source cache."
    if "External" not in case.structures:
        n = case.n_feasible
        case.structures["External"] = Structure(
            name="External",
            raw_name="possible_dose_mask",
            mask_idx=np.arange(n, dtype=np.int32),
            n_outside=0,
            volume_cc=float(np.sum(case.voxel_volume_cc)),
        )
        case.provenance.setdefault("conversions", []).append("External structure derived from the possible-dose mask at load time")
    return case


def load_patient_dir(
    pdir: Path, case_id: str | None = None, grid_shape: tuple[int, int, int] = GRID_SHAPE
) -> tuple[Case, LoadReport]:
    """Build a :class:`Case` from one extracted ``pt_<n>`` folder.

    ``grid_shape`` is the source grid (128^3 for OpenKBP); tests pass a small grid.
    """
    pdir = Path(pdir)
    case_id = case_id or pdir.name
    rep = LoadReport(case_id=case_id)
    for f in REQUIRED_FILES:
        if not (pdir / f).exists():
            raise FileNotFoundError(f"{pdir}: missing {f}")

    t0 = time.perf_counter()
    voxel_size = read_voxel_dimensions(pdir / "voxel_dimensions.csv")
    voxel_cc = float(np.prod(voxel_size)) / 1000.0

    D_full = sp.load_npz(pdir / "dij.npz")
    if D_full.shape[0] != int(np.prod(grid_shape)):
        raise ValueError(f"{case_id}: dij has {D_full.shape[0]} rows, expected {np.prod(grid_shape)}")
    D_csr = sp.csr_matrix(D_full)
    nz_rows = np.flatnonzero(np.diff(D_csr.indptr))

    mask_idx, _ = read_sparse_csv(pdir / "possible_dose_mask.csv")
    mask_idx = np.unique(mask_idx)
    if not np.array_equal(mask_idx, nz_rows):
        extra_D = np.setdiff1d(nz_rows, mask_idx).size
        extra_mask = np.setdiff1d(mask_idx, nz_rows).size
        rep.warnings.append(
            f"possible_dose_mask and nonzero dij rows differ: {extra_D} dij rows outside mask "
            f"(dropped, dose forced to zero), {extra_mask} mask voxels with no dij entries (kept)"
        )
    feasible_idx = mask_idx.astype(np.int64)
    D = restrict_rows(D_csr, feasible_idx)
    rep.conversions.append(
        f"dij restricted from {D_full.shape[0]} grid rows to {feasible_idx.size} feasible rows (COO -> CSR, int32 indices)"
    )

    # Reference dose in feasible space; voxels in the mask without a dose entry are zero.
    dose_idx, dose_val = read_sparse_csv(pdir / "dose.csv")
    if dose_val is None:
        raise ValueError(f"{case_id}: dose.csv has no data column")
    pos = np.searchsorted(feasible_idx, dose_idx)
    inside = (pos < feasible_idx.size) & (feasible_idx[np.minimum(pos, feasible_idx.size - 1)] == dose_idx)
    if not inside.all():
        rep.warnings.append(f"{(~inside).sum()} reference-dose voxels lie outside the feasible mask and were dropped")
    reference_dose = np.zeros(feasible_idx.size, dtype=np.float64)
    reference_dose[pos[inside]] = dose_val[inside]

    beamlets = read_beamlet_indices(pdir / "beamlet_indices.csv")
    if beamlets.n_beamlets != D.shape[1]:
        raise ValueError(
            f"{case_id}: beamlet_indices has {beamlets.n_beamlets} rows but dij has {D.shape[1]} columns"
        )
    rep.conversions.append("beamlet angle index 1..9 mapped to zero-based beam index 0..8")

    structures: dict[str, Structure] = {}
    prescriptions: dict[str, float] = {}
    for raw in RAW_STRUCTURES:
        f = pdir / f"{raw}.csv"
        if not f.exists():
            rep.missing_structures.append(raw)
            continue
        s_idx, _ = read_sparse_csv(f)
        s_idx = np.unique(s_idx)
        p = np.searchsorted(feasible_idx, s_idx)
        ok = (p < feasible_idx.size) & (feasible_idx[np.minimum(p, feasible_idx.size - 1)] == s_idx)
        n_outside = int((~ok).sum())
        res = resolve(raw)
        if not res.known:
            rep.warnings.append(f"structure {raw!r} is not in the TG-263 alias table; kept raw name")
        if n_outside:
            rep.warnings.append(f"{res.canonical}: {n_outside} of {s_idx.size} voxels outside the feasible mask (zero dose)")
        structures[res.canonical] = Structure(
            name=res.canonical,
            raw_name=raw,
            mask_idx=p[ok].astype(np.int32),
            n_outside=n_outside,
            volume_cc=float(s_idx.size * voxel_cc),
        )
        if is_target_name(res.canonical):
            rx = TARGET_PRESCRIPTION_GY.get(res.canonical)
            if rx is None:
                rep.warnings.append(f"no prescription known for target {res.canonical}")
            else:
                prescriptions[res.canonical] = rx

    # Any structure file we did not expect is reported, not silently ignored.
    known = set(RAW_STRUCTURES) | set(REQUIRED_FILES) | {"ct.csv"}
    for f in sorted(pdir.glob("*.csv")):
        if f.stem not in known and f.name not in known:
            rep.warnings.append(f"unexpected file {f.name} ignored")

    case = Case(
        case_id=case_id,
        cohort=COHORT,
        grid_shape=grid_shape,
        voxel_size_mm=voxel_size,
        feasible_idx=feasible_idx,
        voxel_volume_cc=np.full(feasible_idx.size, voxel_cc, dtype=np.float64),
        D=D,
        structures=structures,
        prescriptions=prescriptions,
        reference_dose=reference_dose,
        beamlets=beamlets,
        beam_geometry=dict(BEAM_GEOMETRY),
        provenance={
            "source": "OpenKBP-Opt core-data.zip, reference-plans/" + pdir.name,
            "warnings": rep.warnings,
            "conversions": rep.conversions,
            "missing_structures": rep.missing_structures,
        },
    )
    ensure_external(case)
    rep.stats = {
        "n_feasible": int(feasible_idx.size),
        "n_beamlets": int(D.shape[1]),
        "nnz": int(D.nnz),
        "n_beams": int(beamlets.n_beams),
        "voxel_size_mm": list(voxel_size),
        "voxel_cc": voxel_cc,
        "n_structures": len([n for n in structures if n != "External"]),
        "n_targets": len(prescriptions),
        "structures": sorted(n for n in structures if n != "External"),
        "missing_structures": rep.missing_structures,
        "D_mem_mb": round((D.data.nbytes + D.indices.nbytes + D.indptr.nbytes) / 2**20, 1),
        "ref_dose_max_gy": float(reference_dose.max()),
        "load_s": round(time.perf_counter() - t0, 2),
    }
    return case, rep


def list_patients_in_zip(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    pts: set[str] = set()
    for n in names:
        if n.startswith(ARCHIVE_PREFIX):
            rest = n[len(ARCHIVE_PREFIX):]
            top = rest.split("/", 1)[0]
            if PATIENT_RE.match(top):
                pts.add(top)
    return sorted(pts, key=lambda s: int(PATIENT_RE.match(s).group(1)))  # type: ignore[union-attr]


def extract_patient(zip_path: Path, patient: str, dest: Path) -> Path:
    prefix = f"{ARCHIVE_PREFIX}{patient}/"
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith(prefix) and not m.endswith("/")]
        if not members:
            raise FileNotFoundError(f"{patient} not found in {zip_path}")
        out = Path(dest) / patient
        out.mkdir(parents=True, exist_ok=True)
        for m in members:
            target = out / Path(m).name
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
    return out


def ingest(
    zip_path: Path,
    cache_dir: Path,
    patients: list[str] | None = None,
    resume: bool = True,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
    grid_shape: tuple[int, int, int] = GRID_SHAPE,
) -> dict[str, Any]:
    """Convert patients from ``core-data.zip`` into the package cache. Returns the manifest.

    Ingest is resumable: cases already in the manifest are skipped when ``resume`` is true, so a
    long run can be split across calls or restarted after an interruption.
    """
    zip_path, cache_dir = Path(zip_path), Path(cache_dir)
    manifest = read_manifest(cache_dir, COHORT)
    manifest.setdefault("cohort", COHORT)
    manifest.setdefault("source_archive", {"name": zip_path.name, "size_bytes": zip_path.stat().st_size})
    manifest.setdefault("cases", {})
    manifest["beam_geometry"] = BEAM_GEOMETRY

    todo = patients or list_patients_in_zip(zip_path)
    for pt in todo:
        if resume and pt in manifest["cases"]:
            if progress:
                progress(pt, {"skipped": True})
            continue
        with tempfile.TemporaryDirectory(prefix="opengray_") as tmp:
            pdir = extract_patient(zip_path, pt, Path(tmp))
            checksums = {f.name: sha256_file(f) for f in sorted(pdir.iterdir())}
            case, rep = load_patient_dir(pdir, case_id=pt, grid_shape=grid_shape)
            case.provenance["source_sha256"] = checksums
            write_case(case, cache_dir)
        entry = {"stats": rep.stats, "warnings": rep.warnings, "source_sha256": checksums}
        manifest["cases"][pt] = entry
        write_manifest(cache_dir, COHORT, manifest)
        if progress:
            progress(pt, entry)
    return manifest


class OpenKBPOptLoader(CohortLoader):
    """Loads cached OpenKBP-Opt cases. Run ``opengray ingest`` first."""

    cohort = COHORT

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def list_cases(self) -> list[str]:
        return list_cached_cases(self.cache_dir, COHORT)

    def load(self, case_id: str) -> Case:
        return ensure_external(read_case(self.cache_dir, COHORT, case_id))

    def manifest(self) -> dict[str, Any]:
        return read_manifest(self.cache_dir, COHORT)
