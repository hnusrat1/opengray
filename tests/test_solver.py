from __future__ import annotations

import numpy as np
import pytest
import scipy.optimize as sopt
import scipy.sparse as sp

from opengray.data.base import BeamletTable, Case, Structure
from opengray.physics.objectives import (
    AssembledObjective,
    Objective,
    ObjectiveTerm,
    smoothness_operator,
)
from opengray.physics.solver import CaseSolverContext, SolverConfig, operator_norm, solve


def make_case(n_vox: int = 60, n_beamlets: int = 16, seed: int = 0, density: float = 0.3, n_outside: int = 0) -> Case:
    rng = np.random.default_rng(seed)
    D = sp.random(n_vox, n_beamlets, density=density, random_state=rng, data_rvs=lambda size: rng.uniform(0.05, 1.0, size))
    D = sp.csr_matrix(D)
    # Guarantee every voxel and every beamlet has at least one entry.
    D = D + sp.csr_matrix((np.full(n_vox, 0.05), (np.arange(n_vox), np.arange(n_vox) % n_beamlets)), shape=D.shape)
    ptv = np.arange(0, n_vox // 3, dtype=np.int32)
    oar = np.arange(n_vox // 3, 2 * n_vox // 3, dtype=np.int32)
    oar2 = np.arange(2 * n_vox // 3, n_vox, dtype=np.int32)
    side = int(np.ceil(np.sqrt(n_beamlets)))
    j = np.arange(n_beamlets)
    beamlets = BeamletTable(row=(j // side).astype(np.int32), col=(j % side).astype(np.int32), beam=np.zeros(n_beamlets, np.int32))
    return Case(
        case_id="synthetic",
        cohort="test",
        feasible_idx=np.arange(n_vox, dtype=np.int64),
        voxel_volume_cc=np.full(n_vox, 0.03),
        D=D,
        structures={
            "PTV_7000": Structure(name="PTV_7000", raw_name="PTV70", mask_idx=ptv, volume_cc=ptv.size * 0.03),
            "SpinalCord": Structure(name="SpinalCord", raw_name="SpinalCord", mask_idx=oar, n_outside=n_outside, volume_cc=(oar.size + n_outside) * 0.03),
            "Parotid_L": Structure(name="Parotid_L", raw_name="LeftParotid", mask_idx=oar2, volume_cc=oar2.size * 0.03),
        },
        prescriptions={"PTV_7000": 70.0},
        beamlets=beamlets,
    )


def diagonal_case(a: np.ndarray) -> Case:
    n = a.size
    D = sp.csr_matrix(sp.diags(a))
    idx = np.arange(n, dtype=np.int32)
    return Case(
        case_id="diag",
        cohort="test",
        feasible_idx=np.arange(n, dtype=np.int64),
        voxel_volume_cc=np.ones(n),
        D=D,
        structures={"PTV_7000": Structure(name="PTV_7000", raw_name="PTV70", mask_idx=idx, volume_cc=float(n))},
        prescriptions={"PTV_7000": 70.0},
    )


ALL_TERMS = [
    ObjectiveTerm(structure="PTV_7000", term="min_dose", level=66.5, priority=10),
    ObjectiveTerm(structure="PTV_7000", term="uniform_dose", level=70.0, priority=2),
    ObjectiveTerm(structure="PTV_7000", term="max_dose", level=74.9, priority=5),
    ObjectiveTerm(structure="SpinalCord", term="max_dose", level=45.0, priority=20),
    ObjectiveTerm(structure="SpinalCord", term="dvh_max", level=30.0, priority=3, volume_fraction=0.3),
    ObjectiveTerm(structure="Parotid_L", term="mean_dose", level=26.0, priority=4),
]


@pytest.mark.parametrize("n_outside", [0, 7])
def test_gradient_matches_finite_differences(n_outside: int) -> None:
    case = make_case(n_outside=n_outside)
    obj = AssembledObjective(case, Objective(terms=ALL_TERMS, smoothness_lambda=1e-2))
    rng = np.random.default_rng(1)
    w = rng.uniform(20, 120, case.n_beamlets)  # doses straddle every level so all branches are active
    f, g = obj.value_and_grad(w)
    assert f > 0
    eps = 1e-5
    for j in rng.choice(case.n_beamlets, size=6, replace=False):
        e = np.zeros_like(w)
        e[j] = eps
        fd = (obj.value(w + e) - obj.value(w - e)) / (2 * eps)
        assert fd == pytest.approx(g[j], rel=1e-4, abs=1e-6)


def test_smoothness_operator_counts_neighbours() -> None:
    b = BeamletTable(row=np.array([0, 0, 1, 1], np.int32), col=np.array([0, 1, 0, 1], np.int32), beam=np.zeros(4, np.int32))
    L = smoothness_operator(b)
    assert L.shape == (4, 4)  # two horizontal and two vertical pairs
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert np.allclose(L @ w, 0)
    w = np.array([0.0, 1.0, 0.0, 1.0])
    assert float((L @ w) @ (L @ w)) == 2.0
    # Different beams are never coupled.
    b2 = BeamletTable(row=np.array([0, 0], np.int32), col=np.array([0, 1], np.int32), beam=np.array([0, 1], np.int32))
    assert smoothness_operator(b2).shape[0] == 0


def test_uniform_dose_on_diagonal_matrix_recovers_exact_solution() -> None:
    a = np.array([0.5, 1.0, 2.0, 4.0])
    case = diagonal_case(a)
    obj = Objective(terms=[ObjectiveTerm(structure="PTV_7000", term="uniform_dose", level=60.0, priority=1)], smoothness_lambda=0.0)
    res = solve(case, obj, config=SolverConfig(rel_tol=1e-12, max_iter=5000, w_max=None, complexity_limit=None))
    assert res.converged
    assert np.allclose(res.w, 60.0 / a, rtol=1e-4)
    assert np.allclose(res.dose, 60.0, atol=1e-2)


def test_min_dose_from_zero_reaches_the_level() -> None:
    a = np.array([0.5, 1.0, 2.0, 4.0])
    case = diagonal_case(a)
    obj = Objective(terms=[ObjectiveTerm(structure="PTV_7000", term="min_dose", level=66.5, priority=1)], smoothness_lambda=0.0)
    res = solve(case, obj, config=SolverConfig(rel_tol=1e-10, w_max=None, complexity_limit=None))
    assert res.dose.min() >= 66.5 - 1e-2
    assert res.objective_value < 1e-3


def test_max_dose_from_hot_start_comes_down() -> None:
    a = np.array([0.5, 1.0, 2.0, 4.0])
    case = diagonal_case(a)
    obj = Objective(terms=[ObjectiveTerm(structure="PTV_7000", term="max_dose", level=45.0, priority=1)], smoothness_lambda=0.0)
    res = solve(case, obj, w0=np.full(4, 200.0), config=SolverConfig(rel_tol=1e-10, w_max=None, complexity_limit=None))
    assert res.dose.max() <= 45.0 + 1e-2
    assert (res.w >= 0).all()


def test_mean_dose_is_soft_and_monotone_in_priority() -> None:
    # Alone, from a hot start, the mean term brings the mean down to the level.
    a = np.array([0.5, 1.0, 2.0, 4.0])
    case = diagonal_case(a)
    obj = Objective(terms=[ObjectiveTerm(structure="PTV_7000", term="mean_dose", level=20.0, priority=1)], smoothness_lambda=0.0)
    res = solve(case, obj, w0=np.full(4, 100.0), config=SolverConfig(rel_tol=1e-10, w_max=None, complexity_limit=None))
    assert case.structure_dose(res.dose, "PTV_7000").mean() <= 20.0 + 1e-2

    # Against a competing target term it is a trade-off, and a higher priority means a smaller violation.
    case = make_case()
    violations = []
    for p in (5.0, 50.0, 500.0):
        obj = Objective(
            terms=[
                ObjectiveTerm(structure="PTV_7000", term="min_dose", level=66.5, priority=10),
                ObjectiveTerm(structure="Parotid_L", term="mean_dose", level=20.0, priority=p),
            ],
            smoothness_lambda=0.0,
        )
        res = solve(case, obj, config=SolverConfig(rel_tol=1e-9, w_max=None, complexity_limit=None))
        violations.append(max(0.0, case.structure_dose(res.dose, "Parotid_L").mean() - 20.0))
    assert violations[0] > violations[1] > violations[2]


def test_dvh_max_penalizes_only_the_excess_fraction() -> None:
    # Uncoupled voxels, all hot: the term must push exactly 1 - vf of them to the level and
    # leave the allowed top fraction alone.
    n = 20
    a = np.linspace(0.5, 2.0, n)
    case = diagonal_case(a)
    obj = Objective(
        terms=[ObjectiveTerm(structure="PTV_7000", term="dvh_max", level=30.0, priority=1, volume_fraction=0.25)],
        smoothness_lambda=0.0,
    )
    res = solve(case, obj, w0=50.0 / a, config=SolverConfig(rel_tol=1e-10, max_iter=4000, w_max=None, complexity_limit=None))
    d = case.structure_dose(res.dose, "PTV_7000")
    above = d > 30.0 + 1e-2
    assert above.sum() == int(0.25 * n)
    # The allowed fraction is not driven to the level (ties at the start push it down a little,
    # which is the known behaviour of the sort-and-select approximation).
    assert (d[above] > 40.0).all()
    assert res.objective_value < 1e-6


def test_matches_independent_solver_within_one_percent() -> None:
    case = make_case(n_vox=90, n_beamlets=24, seed=5)
    objective = Objective(
        terms=[
            ObjectiveTerm(structure="PTV_7000", term="min_dose", level=66.5, priority=10),
            ObjectiveTerm(structure="PTV_7000", term="max_dose", level=74.9, priority=5),
            ObjectiveTerm(structure="SpinalCord", term="max_dose", level=45.0, priority=20),
            ObjectiveTerm(structure="Parotid_L", term="mean_dose", level=26.0, priority=4),
        ],
        smoothness_lambda=1e-3,
    )
    obj = AssembledObjective(case, objective)
    ref = sopt.minimize(
        lambda w: obj.value_and_grad(w),
        np.zeros(case.n_beamlets),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0, None)] * case.n_beamlets,
        options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10},
    )
    res = solve(case, objective, config=SolverConfig(rel_tol=1e-9, max_iter=5000, w_max=None, complexity_limit=None))
    assert res.objective_value == pytest.approx(ref.fun, rel=1e-2, abs=1e-6)
    assert (res.w >= 0).all()


def test_deterministic_and_warm_start_is_faster() -> None:
    case = make_case(seed=8)
    objective = Objective(terms=ALL_TERMS, smoothness_lambda=1e-3)
    ctx = CaseSolverContext(case)
    r1 = solve(case, objective, context=ctx)
    r2 = solve(case, objective, context=ctx)
    assert np.array_equal(r1.w, r2.w) and r1.iterations == r2.iterations
    r3 = solve(case, objective, w0=r1.w, context=ctx)
    assert r3.iterations < r1.iterations
    assert r3.objective_value <= r1.objective_value * (1 + 1e-6)


def test_operator_norm_matches_dense_svd() -> None:
    rng = np.random.default_rng(0)
    M = sp.csr_matrix(rng.uniform(0, 1, (30, 12)))
    assert operator_norm(M) == pytest.approx(np.linalg.norm(M.toarray(), 2), rel=1e-6)


def test_objective_validation_rejects_unknown_structure_and_bad_terms() -> None:
    case = make_case()
    bad = Objective(terms=[ObjectiveTerm(structure="Lens_L", term="max_dose", level=5, priority=1)])
    assert bad.validate_against(case) == ["term 0: structure 'Lens_L' not in case"]
    with pytest.raises(ValueError):
        ObjectiveTerm(structure="SpinalCord", term="dvh_max", level=30, priority=1)
    with pytest.raises(ValueError):
        ObjectiveTerm(structure="SpinalCord", term="max_dose", level=30, priority=2000)
    assert Objective(terms=ALL_TERMS).hash() == Objective(terms=list(ALL_TERMS)).hash()
