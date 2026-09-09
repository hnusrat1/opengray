"""Sparse dose influence matrix helpers: dose, column norms, pruning."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def compute_dose(D: sp.csr_matrix, w: np.ndarray) -> np.ndarray:
    """dose = D @ w in Gy; ``w`` is clipped to be nonnegative first, and the clip is reported."""
    w = np.asarray(w, dtype=np.float64)
    if (w < 0).any():
        raise ValueError("beamlet intensities must be nonnegative")
    return np.asarray(D @ w).ravel()


def column_norms(D: sp.csr_matrix) -> np.ndarray:
    """Euclidean norm of every column (beamlet)."""
    Dc = D.tocsc(copy=False)
    return np.sqrt(np.asarray(Dc.multiply(Dc).sum(axis=0)).ravel())


def restrict_rows(D: sp.spmatrix, rows: np.ndarray) -> sp.csr_matrix:
    """Keep only the given rows (sorted unique int64) and return CSR with int32 indices."""
    rows = np.asarray(rows, dtype=np.int64)
    out = sp.csr_matrix(D)[rows, :]
    out.sort_indices()
    out.indices = out.indices.astype(np.int32, copy=False)
    out.indptr = out.indptr.astype(np.int32, copy=False)
    return out


def prune_columns(D: sp.csr_matrix, norm_threshold: float) -> tuple[sp.csr_matrix, np.ndarray]:
    "Explicitly prune beamlets below a column-norm threshold; return the matrix and retained column indices. Never applied silently."
    norms = column_norms(D)
    keep = np.flatnonzero(norms >= norm_threshold)
    return sp.csr_matrix(D[:, keep]), keep
