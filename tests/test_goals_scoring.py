from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from opengray.data.base import Case, Structure
from opengray.goals.defaults import openkbp_default_goals
from opengray.goals.schema import Goal, GoalList
from opengray.goals.scoring import ScoreWeights, normalized_violation, plan_score
from opengray.physics.dvh import (
    d_cc,
    d_percent,
    dvh_curve,
    evaluate,
    merged_target_indices,
    parse_metric,
)


def build_case(n: int = 200, voxel_cc: float = 0.038) -> Case:
    """A case whose dose we set directly (D is identity so dose == w)."""
    idx = np.arange(n, dtype=np.int32)
    ptv70 = idx[:40]
    ptv56 = idx[40:100]
    cord = idx[100:130]
    par_l = idx[130:160]
    par_r = idx[160:190]
    return Case(
        case_id="score",
        cohort="test",
        feasible_idx=np.arange(n, dtype=np.int64),
        voxel_volume_cc=np.full(n, voxel_cc),
        D=sp.identity(n, format="csr"),
        structures={
            "PTV_7000": Structure(name="PTV_7000", raw_name="PTV70", mask_idx=ptv70, volume_cc=ptv70.size * voxel_cc),
            "PTV_5600": Structure(name="PTV_5600", raw_name="PTV56", mask_idx=ptv56, volume_cc=ptv56.size * voxel_cc),
            "SpinalCord": Structure(name="SpinalCord", raw_name="SpinalCord", mask_idx=cord, n_outside=10, volume_cc=40 * voxel_cc),
            "Parotid_L": Structure(name="Parotid_L", raw_name="LeftParotid", mask_idx=par_l, volume_cc=par_l.size * voxel_cc),
            "Parotid_R": Structure(name="Parotid_R", raw_name="RightParotid", mask_idx=par_r, volume_cc=par_r.size * voxel_cc),
        },
        prescriptions={"PTV_7000": 70.0, "PTV_5600": 56.0},
    )


def good_dose(case: Case) -> np.ndarray:
    d = np.zeros(case.n_feasible)
    d[:40] = 70.0
    d[40:100] = 56.5
    d[100:130] = 30.0
    d[130:160] = 20.0
    d[160:190] = 24.0
    return d


def test_metric_grammar() -> None:
    assert parse_metric("Dmean").kind == "Dmean"
    assert parse_metric("D99").param == 99
    assert parse_metric("D0.1cc").param == pytest.approx(0.1)
    assert parse_metric("V20Gy").param == 20
    assert parse_metric("CI").targets_only and parse_metric("HI").targets_only
    for bad in ("D0", "D101", "Dx", "V-1Gy", "D0cc", "mean"):
        with pytest.raises(ValueError):
            parse_metric(bad)
    assert parse_metric("D95").name == "D95" and parse_metric("D0.1cc").name == "D0.1cc"


def test_metrics_match_reference_formulas() -> None:
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 70, 500)
    assert d_percent(d, 99) == np.percentile(d, 1)
    assert d_percent(d, 95) == np.percentile(d, 5)
    assert d_percent(d, 1) == np.percentile(d, 99)
    # OpenKBP: voxels_in_tenth_of_cc = max(1, round(100 / voxel_mm3)); 38 mm3 voxels -> 3 voxels.
    voxel_cc = 0.038
    n_vox = max(1, round(0.1 / voxel_cc))
    assert n_vox == 3
    assert d_cc(d, 0.1, voxel_cc) == np.percentile(d, 100 - n_vox / d.size * 100)
    # Tiny voxels still use at least one voxel.
    assert d_cc(d, 0.1, 10.0) == np.percentile(d, 100 - 1 / d.size * 100)
    levels, frac = dvh_curve(d, 11)
    assert frac[0] == 1.0 and frac[-1] <= 1 / d.size + 1e-12 and np.all(np.diff(frac) <= 0)


def test_structure_metrics_include_outside_voxels_and_merge_targets() -> None:
    case = build_case()
    dose = good_dose(case)
    # Cord has 30 voxels at 30 Gy plus 10 zero-dose voxels outside the mask: mean is 22.5, max 30.
    assert evaluate(case, dose, "SpinalCord", "Dmean") == pytest.approx(22.5)
    assert evaluate(case, dose, "SpinalCord", "Dmax") == 30.0
    assert evaluate(case, dose, "SpinalCord", "V25Gy") == pytest.approx(0.75)
    # Merged PTV56 = PTV56 + PTV70 (100 voxels); D99 sits in the 56.5 Gy part either way here.
    idx, n_out = merged_target_indices(case, "PTV_5600")
    assert idx.size == 100 and n_out == 0
    assert evaluate(case, dose, "PTV_5600", "D99", merge_targets=True) == pytest.approx(56.5)
    dose2 = dose.copy()
    dose2[40:100] = 50.0  # PTV56 cold: merged D99 differs from unmerged only through the union
    assert evaluate(case, dose2, "PTV_5600", "D99", merge_targets=False) == 50.0
    assert evaluate(case, dose2, "PTV_5600", "D99", merge_targets=True) == 50.0
    assert evaluate(case, dose, "PTV_7000", "HI") == pytest.approx(1.0)
    assert evaluate(case, dose, "PTV_7000", "CI") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        evaluate(case, dose, "SpinalCord", "CI")


def test_goal_list_units_and_case_filtering() -> None:
    with pytest.raises(ValueError):
        GoalList(goals=[Goal(structure="A", metric="Dmean", op="<=", value=26, unit="Gy"), Goal(structure="B", metric="Dmean", op="<=", value=2600, unit="cGy")])
    trap = GoalList(goals=[Goal(structure="A", metric="Dmean", op="<=", value=26, unit="Gy"), Goal(structure="B", metric="Dmean", op="<=", value=2600, unit="cGy")], allow_mixed_units=True)
    gy = trap.to_gy()
    assert [g.value for g in gy.goals] == [26.0, 26.0] and all(g.unit == "Gy" for g in gy.goals)
    assert gy.log == ["converted B Dmean <= 2600 cGy to Gy"]
    case = build_case()
    default = openkbp_default_goals()
    assert len(default.goals) == 11
    applied = default.for_case(case)
    assert applied.structures() == ["PTV_7000", "PTV_5600", "SpinalCord", "Parotid_L", "Parotid_R"]
    assert len(applied.log) == 6 and all("dropped" in line for line in applied.log)
    assert Goal(structure="X", metric="D0.1cc", op="<=", value=4500, unit="cGy").is_met(44.0)
    assert not Goal(structure="X", metric="D99", op=">=", value=66.5, unit="Gy").is_met(66.0)


def test_plan_score_breakdown_on_a_good_plan() -> None:
    case = build_case()
    dose = good_dose(case)
    ref = dose.copy()
    ref[130:160] = 25.0  # reference parotid_L mean 25 vs ours 20; parotid_R equal
    bd = plan_score(case, dose, openkbp_default_goals(), reference_dose=ref)
    assert not bd.gated and bd.n_hard == 3 and bd.n_soft == 2
    assert bd.H == 1.0 and bd.V == 0.0
    # r values: (25 - 20) / 25 = 0.2 and 0 -> mean 0.1 -> R = 0.55
    assert bd.R == pytest.approx(0.55)
    assert bd.plan_score == pytest.approx(0.5 + 0.3 + 0.2 * 0.55)
    names = {(g.structure, g.metric) for g in bd.goals}
    assert ("SpinalCord", "D0.1cc") in names and ("PTV_5600", "D99") in names
    row = bd.as_row()
    assert row["Parotid_L.Dmean"] == pytest.approx(20.0)
    # Same plan scored with cGy goals gives the same result.
    cgy = GoalList(goals=[g.model_copy(update={"value": g.value * 100, "unit": "cGy"}) for g in openkbp_default_goals().goals])
    bd2 = plan_score(case, dose, cgy, reference_dose=ref)
    assert bd2.plan_score == pytest.approx(bd.plan_score)


def test_violations_and_gates() -> None:
    case = build_case()
    dose = good_dose(case)
    dose[130:160] = 39.0  # parotid_L mean 39 vs 26: violation 0.5
    dose[100:130] = 50.0  # cord D0.1cc 50 > 45: hard goal fails
    bd = plan_score(case, dose, openkbp_default_goals())
    assert bd.H == pytest.approx(2 / 3)
    assert bd.V == pytest.approx(((39 - 26.05) / 26 + 0) / 2)  # violation beyond the tolerated limit, mean with 0
    assert bd.R == 0.5 and "no reference plan" in bd.log[-1]
    assert bd.plan_score == pytest.approx(0.5 * 2 / 3 + 0.3 * (1 - bd.V) + 0.2 * 0.5)

    hot = good_dose(case)
    hot[0] = 81.0  # 1.157 x 70, above the 1.15 gate
    bd = plan_score(case, hot, openkbp_default_goals())
    assert bd.gated and bd.plan_score == 0.0 and "global max" in bd.gate_reason

    cold = good_dose(case)
    cold[40:100] = 40.0  # PTV56 D99 40 < 0.8 x 56
    bd = plan_score(case, cold, openkbp_default_goals())
    assert bd.gated and "PTV_5600 D99" in bd.gate_reason

    hard_only = plan_score(case, good_dose(case), openkbp_default_goals(), weights=ScoreWeights.hard_only())
    assert hard_only.plan_score == 1.0


@pytest.mark.parametrize("gate", ["maximum", "coverage"])
def test_gate_boundary_is_stable_to_fluence_replay_roundoff(gate):
    case = build_case()
    dose = good_dose(case)
    if gate == "maximum":
        dose[0] = 80.5 + 6e-14  # observed from D @ normalized_w on a saved real plan
    else:
        dose[:40] = 56.0 - 6e-14
    assert not plan_score(case, dose, openkbp_default_goals()).gated
    if gate == "maximum":
        dose[0] = 80.5 + 1e-6
    else:
        dose[:40] = 56.0 - 1e-6
    assert plan_score(case, dose, openkbp_default_goals()).gated


@settings(max_examples=60, deadline=None)
@given(a=st.floats(0, 200, allow_nan=False), b=st.floats(0, 200, allow_nan=False))
def test_violation_is_monotone_and_clipped(a: float, b: float) -> None:
    upper = Goal(structure="X", metric="Dmean", op="<=", value=26, unit="Gy", kind="soft")
    lower = Goal(structure="X", metric="D99", op=">=", value=66.5, unit="Gy", kind="soft")
    lo, hi = min(a, b), max(a, b)
    assert 0.0 <= normalized_violation(upper, lo) <= normalized_violation(upper, hi) <= 1.0
    assert 0.0 <= normalized_violation(lower, hi) <= normalized_violation(lower, lo) <= 1.0
    assert normalized_violation(upper, 26.0) == 0.0 and normalized_violation(lower, 66.5) == 0.0
    assert normalized_violation(upper, 26.04) == 0.0 and normalized_violation(lower, 66.46) == 0.0


@settings(max_examples=40, deadline=None)
@given(cord=st.floats(0, 60, allow_nan=False), par=st.floats(0, 60, allow_nan=False))
def test_score_is_monotone_in_oar_dose(cord: float, par: float) -> None:
    case = build_case()
    d1 = good_dose(case)
    d1[100:130] = cord
    d1[130:160] = par
    d2 = d1.copy()
    d2[100:130] = cord + 1.0
    d2[130:160] = par + 1.0
    s1 = plan_score(case, d1, openkbp_default_goals())
    s2 = plan_score(case, d2, openkbp_default_goals())
    assert s2.plan_score <= s1.plan_score + 1e-12
    assert s2.H <= s1.H and s2.V >= s1.V


def test_goals_are_compared_at_reporting_precision() -> None:
    cord = Goal(structure="SpinalCord", metric="D0.1cc", op="<=", value=45, unit="Gy")
    assert cord.is_met(45.0003) and cord.is_met(45.05) and not cord.is_met(45.06)
    assert not cord.is_met(45.0003, tol=0.0)
    cover = Goal(structure="PTV_7000", metric="D99", op=">=", value=6650, unit="cGy")
    assert cover.is_met(66.46) and not cover.is_met(66.44)
    v = Goal(structure="Lung", metric="V20Gy", op="<=", value=0.30, unit="Gy")
    assert v.is_met(0.3004) and not v.is_met(0.301)
    hi = Goal(structure="PTV_7000", metric="HI", op="<=", value=1.10, unit="Gy")
    assert hi.is_met(1.104) and not hi.is_met(1.106)
    # Scoring agrees with is_met: a cord at 45.0003 Gy keeps the hard goal.
    case = build_case()
    dose = good_dose(case)
    dose[100:130] = 45.0003
    bd = plan_score(case, dose, openkbp_default_goals())
    assert bd.H == 1.0 and bd.V == 0.0
