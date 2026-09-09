from __future__ import annotations

import numpy as np

from opengray.physics.validation import (
    default_objective,
    independent_solver_check,
    matrix_read_check,
    subsample_case,
    timing_run,
)
from tests.test_solver import make_case


def test_matrix_read_check_reproduces_a_known_plan() -> None:
    case = make_case(n_vox=80, n_beamlets=20, seed=2)
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 5, case.n_beamlets)
    w[3] = 0.0
    dose = case.dose(w)
    fl = np.stack([np.flatnonzero(w), w[w > 0]], axis=1)
    nz = np.flatnonzero(dose)
    r = matrix_read_check(case, fl, case.feasible_idx[nz], dose[nz])
    assert r["max_abs_diff_gy"] < 1e-9 and r["n_pub_outside_mask"] == 0 and r["w_nonzero"] == 19
    # A wrong index space shows up as a large difference.
    r_bad = matrix_read_check(case, fl, case.feasible_idx[nz], np.roll(dose[nz], 1))
    assert r_bad["max_abs_diff_gy"] > 0.1


def test_subsample_keeps_structures_consistent() -> None:
    case = make_case(n_vox=120, n_beamlets=16, seed=4)
    sub = subsample_case(case, n_per_structure=10)
    assert sub.n_feasible <= 30
    for name, s in sub.structures.items():
        assert s.mask_idx.size <= 10 and s.n_outside == case.structures[name].n_outside
        assert s.mask_idx.max() < sub.n_feasible
    assert sub.D.shape == (sub.n_feasible, case.n_beamlets)


def test_independent_solver_agrees_and_timing_runs() -> None:
    case = make_case(n_vox=120, n_beamlets=16, seed=6)
    objective = default_objective(case)
    r = independent_solver_check(case, objective, n_per_structure=25)
    assert abs(r["rel_diff"]) < 1e-2
    # With the weight box and a fixed SPG penalty both solvers still agree.
    r2 = independent_solver_check(case, objective, n_per_structure=25, w_max=15.0, complexity_lambda=50.0, complexity_limit=5.0)
    assert abs(r2["rel_diff"]) < 2e-2, r2
    t = timing_run(case, objective)
    assert t["cold_converged"] and t["warm_converged"]
    # Iteration counts include the complexity continuation rounds, so a warm start is not
    # guaranteed fewer iterations than a cold one; both must converge.
    assert t["warm_iters"] > 0 and t["cold_iters"] > 0
