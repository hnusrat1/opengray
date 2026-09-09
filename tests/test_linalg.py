"""Threaded CSR products (physics/linalg) are bit-identical to scipy's single-threaded ones."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from opengray.physics.linalg import BlockedCSR
from opengray.physics.objectives import AssembledObjective, Objective
from tests.test_solver import ALL_TERMS, make_case


def test_blocked_matvec_is_bit_identical_for_any_block_count() -> None:
    rng = np.random.default_rng(3)
    M = sp.random(1000, 300, density=0.05, random_state=rng, format="csr", dtype=np.float64)
    M.indices = M.indices.astype(np.int32)
    M.indptr = M.indptr.astype(np.int32)
    x = rng.standard_normal(300)
    ref = M @ x
    for k in (1, 2, 3, 4, 7):
        y = BlockedCSR(M, blocks=k).matvec(x)
        assert y.shape == ref.shape and np.array_equal(y, ref), k
    MT = M.T.tocsr()
    v = rng.standard_normal(1000)
    assert np.array_equal(BlockedCSR(MT, blocks=4).matvec(v), MT @ v)


def test_objective_value_matches_value_and_grad_and_scipy_products() -> None:
    case = make_case(n_vox=120, n_beamlets=24, seed=5)
    obj = AssembledObjective(case, Objective(terms=ALL_TERMS, smoothness_lambda=1e-3))
    w = np.random.default_rng(1).uniform(0, 30, case.n_beamlets)
    f, g = obj.value_and_grad(w)
    assert obj.value(w) == f
    assert np.array_equal(obj.dose(w), np.asarray(case.D @ w).ravel())
    # The gradient's D^T product equals scipy's.
    d = obj.dose(w)
    assert np.array_equal(obj._DTb.matvec(d), np.asarray(obj.DT @ d).ravel())
    assert g.shape == (case.n_beamlets,)
