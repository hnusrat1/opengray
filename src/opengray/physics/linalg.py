"Threaded sparse matrix-vector products for the fluence optimizer."

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools

_POOL: ThreadPoolExecutor | None = None


def n_threads() -> int:
    env = os.environ.get("OPENGRAY_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(4, os.cpu_count() or 1))


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=n_threads(), thread_name_prefix="opengray-matvec")
    return _POOL


class BlockedCSR:
    """A CSR matrix split into row blocks for threaded matvec; ``matvec(x)`` equals ``M @ x``."""

    def __init__(self, M: sp.spmatrix, blocks: int | None = None):
        M = sp.csr_matrix(M)
        self.shape = M.shape
        self.dtype = M.dtype
        self.M = M
        k = max(1, min(blocks or n_threads(), M.shape[0]))
        bounds = np.linspace(0, M.shape[0], k + 1).astype(int)
        self.blocks = []
        for a, c in zip(bounds[:-1], bounds[1:], strict=True):
            if c <= a:
                continue
            s, e = int(M.indptr[a]), int(M.indptr[c])
            self.blocks.append((int(a), int(c), np.ascontiguousarray(M.indptr[a : c + 1] - M.indptr[a]), M.indices[s:e], M.data[s:e]))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=self.dtype)
        if x.shape != (self.shape[1],):
            raise ValueError(f"expected a vector of length {self.shape[1]}, got {x.shape}")
        y = np.zeros(self.shape[0], dtype=self.dtype)
        if len(self.blocks) == 1:
            a, c, ip, ind, dat = self.blocks[0]
            _sparsetools.csr_matvec(c - a, self.shape[1], ip, ind, dat, x, y)
            return y

        def run(b: tuple) -> None:
            a, c, ip, ind, dat = b
            _sparsetools.csr_matvec(c - a, self.shape[1], ip, ind, dat, x, y[a:c])

        list(_pool().map(run, self.blocks))
        return y
