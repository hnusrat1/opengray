"""Synthetic OpenKBP-Opt style patients so the loader and cache are tested without real data."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

GRID = (8, 8, 8)


def write_sparse_csv(path: Path, indices: np.ndarray, values: np.ndarray | None) -> None:
    with open(path, "w") as f:
        f.write(",data\n")
        if values is None:
            for i in indices:
                f.write(f"{int(i)},\n")
        else:
            for i, v in zip(indices, values, strict=True):
                f.write(f"{int(i)},{float(v):.3f}\n")


def make_synthetic_patient(
    root: Path,
    name: str = "pt_1",
    n_beamlets: int = 12,
    seed: int = 0,
    grid: tuple[int, int, int] = GRID,
    structures: dict[str, np.ndarray] | None = None,
) -> dict:
    """Write one patient folder in the verified OpenKBP-Opt format and return the ground truth."""
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(grid))
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)

    # Feasible mask: a central block. Everything outside must get zero dose.
    feasible = np.arange(n_vox).reshape(grid)[1:7, 1:7, 1:7].ravel()
    feasible = np.sort(feasible)

    # Full-grid COO dij with entries only on feasible rows, positive values.
    rows, cols, vals = [], [], []
    for b in range(n_beamlets):
        hit = rng.choice(feasible, size=rng.integers(20, 60), replace=False)
        rows.append(hit)
        cols.append(np.full(hit.size, b))
        vals.append(rng.uniform(0.01, 1.0, size=hit.size))
    D_full = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32))),
        shape=(n_vox, n_beamlets),
    )
    D_full.sum_duplicates()
    sp.save_npz(pdir / "dij.npz", D_full)

    # Reference plan: dose = D w_ref, written sparse with 3 decimals like the source.
    w_ref = rng.uniform(1.0, 5.0, size=n_beamlets)
    dose_full = np.asarray(sp.csr_matrix(D_full) @ w_ref).ravel()
    nz = np.flatnonzero(dose_full)
    write_sparse_csv(pdir / "dose.csv", nz, dose_full[nz])

    # Mask equals the nonzero rows of dij by construction of the source data.
    nz_rows = np.unique(D_full.row)
    write_sparse_csv(pdir / "possible_dose_mask.csv", nz_rows, None)

    voxel_size = (3.9, 3.9, 2.5)
    (pdir / "voxel_dimensions.csv").write_text("\n".join(f"{v:.18e}" for v in voxel_size) + "\n")

    angles = (np.arange(n_beamlets) % 9) + 1
    with open(pdir / "beamlet_indices.csv", "w") as f:
        f.write("row,column,angle\n")
        for b in range(n_beamlets):
            f.write(f"{b // 4 + 1},{b % 4 + 1},{int(angles[b])}\n")

    if structures is None:
        all_idx = np.arange(n_vox)
        inside = nz_rows
        outside = np.setdiff1d(all_idx, feasible)
        structures = {
            "PTV70": inside[:30],
            "PTV56": inside[30:70],
            "SpinalCord": np.concatenate([inside[70:80], outside[:5]]),  # 5 voxels outside the mask
            "LeftParotid": inside[80:100],
            # Brainstem deliberately absent.
        }
    for sname, idx in structures.items():
        write_sparse_csv(pdir / f"{sname}.csv", np.sort(idx), None)

    (pdir / "ct.csv").write_text(",data\n0,1000.0\n")
    return {
        "dir": pdir,
        "grid": grid,
        "D_full": sp.csr_matrix(D_full),
        "w_ref": w_ref,
        "dose_full": dose_full,
        "feasible": nz_rows,
        "voxel_size": voxel_size,
        "structures": structures,
        "n_beamlets": n_beamlets,
    }


def make_synthetic_archive(root: Path, patients: list[str], seed: int = 0) -> Path:
    """Zip synthetic patients under the real archive prefix."""
    src = root / "src"
    for i, p in enumerate(patients):
        make_synthetic_patient(src, p, seed=seed + i)
    zpath = root / "core-data.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in patients:
            for f in sorted((src / p).iterdir()):
                z.write(f, f"open-kbp-opt-data/reference-plans/{p}/{f.name}")
    return zpath


@pytest.fixture
def synthetic_patient(tmp_path: Path) -> dict:
    return make_synthetic_patient(tmp_path)


@pytest.fixture
def synthetic_archive(tmp_path: Path) -> Path:
    return make_synthetic_archive(tmp_path, ["pt_1", "pt_2", "pt_3", "pt_4"])
