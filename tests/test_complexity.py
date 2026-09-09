"""Sum of positive gradients as OpenKBP-Opt's optimizer defines it (physics/complexity)."""

from __future__ import annotations

import numpy as np
import pytest

from opengray.data.base import BeamletTable
from opengray.physics.complexity import complexity_report, row_spg_terms, spg


def table() -> BeamletTable:
    # Beam 0: row 0 has columns 0, 1, 2; row 1 has columns 0, 2 (column 1 inactive). Beam 1: row 0, columns 0, 1.
    return BeamletTable(row=np.array([0, 0, 0, 1, 1, 0, 0]), col=np.array([0, 1, 2, 0, 2, 0, 1]), beam=np.array([0, 0, 0, 0, 0, 1, 1]))


def test_spg_matches_the_hand_computation() -> None:
    w = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 1.0, 1.0])
    rows = row_spg_terms(table(), w)
    # Beam 0 row 0: (3-1)+ + (1-2)+ + (2-0)+ = 2 + 0 + 2 = 4. Row 1: column 0's right neighbour
    # (column 1) is inactive so it contributes 4; column 2 has no neighbour, contributes 5: 9.
    # Beam 1 row 0: (1-1)+ + (1-0)+ = 1.
    assert rows == {(0, 0): 4.0, (0, 1): 9.0, (1, 0): 1.0}
    assert spg(table(), w) == 9.0 + 1.0  # per beam the largest row, summed over beams
    rep = complexity_report(table(), w, dose=np.array([10.0, 90.0]))
    assert rep["spg"] == 10.0 and rep["spg_per_beam"] == [9.0, 1.0] and rep["spg_within_limit"]
    assert rep["w_max"] == 5.0 and rep["w_within_limit"] and not rep["dose_within_limit"]


def test_spg_is_zero_for_a_flat_map_with_no_edges_and_grows_with_edges() -> None:
    flat = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    # A beamlet whose right neighbour is inactive contributes its full value: beam 0 row 0 gives
    # 2 (last column), row 1 gives 4 (the gap at column 1 and the last column), beam 1 gives 2.
    assert spg(table(), flat) == 4.0 + 2.0
    spiky = np.array([9.0, 0.0, 9.0, 9.0, 9.0, 9.0, 0.0])
    assert spg(table(), spiky) > spg(table(), flat)
    assert spg(table(), np.zeros(7)) == 0.0


def test_complexity_term_matches_spg_and_finite_differences() -> None:
    from opengray.physics.complexity import ComplexityTerm

    t = ComplexityTerm(table(), limit=2.0, tau=0.5)
    w = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 1.0, 1.0])
    assert t.spg(w) == spg(table(), w) == 10.0
    w = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 1.5, 1.0])  # no ties: the penalty is differentiable here
    rs, _ = t.row_sums(w)
    sm, _ = t.smooth_spg(rs)
    assert sm >= t.spg(w) and sm <= t.spg(w) + 0.5 * np.log(2) * 2 + 1e-9  # LSE at most max + tau log(rows) per beam
    f, g = t.value_and_grad(w, lam=1.5)
    assert f > 0
    eps = 1e-6
    for j in range(w.size):
        e = np.zeros_like(w)
        e[j] = eps
        num = (t.value_and_grad(w + e, 1.5)[0] - t.value_and_grad(w - e, 1.5)[0]) / (2 * eps)
        assert abs(num - g[j]) < 1e-4 * max(1.0, abs(g[j])), (j, num, g[j])
    f0, g0 = t.value_and_grad(np.zeros(7), 1.5)
    assert f0 == 0.0 and not g0.any()


def test_solver_continuation_brings_spg_under_the_limit_and_the_cap_binds() -> None:
    from opengray.physics.objectives import Objective, ObjectiveTerm
    from opengray.physics.solver import SolverConfig, solve
    from tests.test_solver import make_case

    case = make_case(n_vox=120, n_beamlets=24, seed=21)
    obj = Objective(terms=[ObjectiveTerm(structure="PTV_7000", term="min_dose", level=70.0, priority=1000), ObjectiveTerm(structure="SpinalCord", term="max_dose", level=20.0, priority=1000)], smoothness_lambda=1e-4)
    free = solve(case, obj, config=SolverConfig(w_max=None, complexity_limit=None))
    from opengray.physics.complexity import spg as spg_of

    s_free = spg_of(case.beamlets, free.w)
    # The unconstrained solve reports the plan's SPG too (the census reads it), with no rounds.
    assert free.spg == pytest.approx(s_free) and free.complexity_rounds == 0
    limit = 0.5 * s_free
    res = solve(case, obj, config=SolverConfig(w_max=None, complexity_limit=limit, complexity_rounds=6))
    assert res.spg is not None and res.spg == pytest.approx(spg_of(case.beamlets, res.w))
    assert res.spg <= limit * 1.02 + 1e-9 and res.complexity_rounds >= 1 and res.complexity_lambda > 0
    assert res.penalty_value >= 0 and res.objective_value >= free.objective_value - 1e-9
    capped = solve(case, obj, config=SolverConfig(w_max=3.0, complexity_limit=None))
    assert capped.w.max() <= 3.0 + 1e-12
