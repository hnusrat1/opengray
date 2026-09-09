"""LP feasibility certificates (physics/certificate): a lower bound on the cord D<v>cc that every
plan meeting the other hard goals must respect."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from opengray.data.base import BeamletTable, Case, Structure
from opengray.goals.defaults import openkbp_default_goals
from opengray.goals.schema import GoalList
from opengray.physics.certificate import (
    ROBUST_EXTRA_VOXELS,
    build_relaxation,
    certify,
    cold_voxel_allowance,
    d_cc_voxels,
)
from opengray.physics.feasibility import coverage_floor, feasibility_status, infeasible_limit
from tests.test_solver import make_case


def with_external(case: Case) -> Case:
    idx = np.arange(case.n_feasible, dtype=np.int32)
    case.structures["External"] = Structure(name="External", raw_name="External", mask_idx=idx, volume_cc=float(case.voxel_volume_cc.sum()))
    return case


def hand_case(n_cord: int = 40, n_target: int = 3, target_coef: float = 1.0, cord_coef: float = 0.5, voxel_cc: float = 0.1) -> Case:
    """One beamlet; every target voxel receives target_coef x w, every cord voxel cord_coef x w."""
    n = n_target + n_cord
    D = sp.csr_matrix(np.concatenate([np.full(n_target, target_coef), np.full(n_cord, cord_coef)])[:, None])
    t_idx = np.arange(n_target, dtype=np.int32)
    c_idx = np.arange(n_target, n, dtype=np.int32)
    case = Case(case_id="hand", cohort="test", feasible_idx=np.arange(n, dtype=np.int64), voxel_volume_cc=np.full(n, voxel_cc), D=D, structures={"PTV_7000": Structure(name="PTV_7000", raw_name="PTV70", mask_idx=t_idx, volume_cc=n_target * voxel_cc), "SpinalCord": Structure(name="SpinalCord", raw_name="SpinalCord", mask_idx=c_idx, volume_cc=n_cord * voxel_cc)}, prescriptions={"PTV_7000": 70.0})
    return with_external(case)


def structured_case(seed: int = 0) -> Case:
    """Sixteen beamlets cover the target (coefficients 0.9 to 1.1) and graze the cord (0.1 to 0.3);
    eight more hit the cord (0.8 to 1.2) and barely the target. Coverage forces cord dose."""
    rng = np.random.default_rng(seed)
    n_t, n_c, n_p, n_a, n_b = 40, 40, 40, 16, 8
    D = np.zeros((n_t + n_c + n_p, n_a + n_b))
    D[:n_t, :n_a] = rng.uniform(0.9, 1.1, (n_t, n_a))
    D[n_t : n_t + n_c, :n_a] = rng.uniform(0.1, 0.3, (n_c, n_a))
    D[n_t + n_c :, :n_a] = rng.uniform(0.2, 0.5, (n_p, n_a))
    D[n_t : n_t + n_c, n_a:] = rng.uniform(0.8, 1.2, (n_c, n_b))
    D[:n_t, n_a:] = rng.uniform(0.0, 0.1, (n_t, n_b))
    n = D.shape[0]
    j = np.arange(n_a + n_b)
    beamlets = BeamletTable(row=(j // 5).astype(np.int32), col=(j % 5).astype(np.int32), beam=np.zeros(j.size, np.int32))
    case = Case(case_id="structured", cohort="test", feasible_idx=np.arange(n, dtype=np.int64), voxel_volume_cc=np.full(n, 0.03), D=sp.csr_matrix(D), structures={"PTV_7000": Structure(name="PTV_7000", raw_name="PTV70", mask_idx=np.arange(0, n_t, dtype=np.int32), volume_cc=n_t * 0.03), "SpinalCord": Structure(name="SpinalCord", raw_name="SpinalCord", mask_idx=np.arange(n_t, n_t + n_c, dtype=np.int32), volume_cc=n_c * 0.03), "Parotid_L": Structure(name="Parotid_L", raw_name="LeftParotid", mask_idx=np.arange(n_t + n_c, n, dtype=np.int32), volume_cc=n_p * 0.03)}, prescriptions={"PTV_7000": 70.0}, beamlets=beamlets)
    return with_external(case)


GOALS = openkbp_default_goals()


def test_voxel_counts_follow_the_dvh_conventions() -> None:
    case = with_external(make_case())
    assert d_cc_voxels(case, "SpinalCord", 0.1) == 3  # 0.1 cc over 0.03 cc voxels
    # D99 on N voxels: numpy's index is 0.01 (N - 1); the allowance is its ceiling plus one.
    assert cold_voxel_allowance(20, 99.0) == 1 + ROBUST_EXTRA_VOXELS
    assert cold_voxel_allowance(101, 99.0) == 1 + ROBUST_EXTRA_VOXELS
    assert cold_voxel_allowance(30001, 99.0) == 300 + ROBUST_EXTRA_VOXELS
    assert cold_voxel_allowance(1000, 95.0) == 50 + ROBUST_EXTRA_VOXELS


def test_hand_case_bound_matches_the_arithmetic() -> None:
    # Three target voxels at 1 x w and forty cord voxels at 0.5 x w. The relaxation allows two
    # cold target voxels (m = ceil(0.02) + 1 = 2), so its only target constraint is the mean row
    # 3 w >= (3 - 2) x 66.45, w >= 22.15, and every tail mean of the cord is 0.5 w = 11.075.
    # With n = round(0.1 / 0.1) + 1 = 2 top voxels capped at 77.05 and k in {4, 8, 20}, the
    # best D0.1cc bound is at k = 20: (20 x 11.075 - 2 x 77.05) / 18 = 3.7444.
    case = hand_case()
    rec = certify(case, GOALS)
    assert "reason" not in rec and rec["n_top"] == 2 and rec["e_eff_gy"] == pytest.approx(77.05)
    assert [t["k"] for t in rec["tails"]] == [4, 8, 20]
    for t in rec["tails"]:
        assert t["status"] == 0 and t["tail_mean_gy"] == pytest.approx(11.075, abs=1e-4)
    assert rec["lower_bound_gy"] == pytest.approx(3.7444 - 1e-3, abs=2e-3)
    assert rec["certified_below_gy"] == pytest.approx(rec["lower_bound_gy"] - 0.05, abs=1e-6)
    # The true minimum (D99 of three equal voxels at 66.45 needs w = 66.45, cord 33.2 Gy) is
    # above the bound, as it must be; a limit below the certificate is certified infeasible.
    assert rec["lower_bound_gy"] < 33.2
    assert feasibility_status(3.0, 33.2, rec["certified_below_gy"]) == "certified_infeasible"
    assert feasibility_status(3.7, 33.2, rec["certified_below_gy"]) == "uncertified"
    assert feasibility_status(33.2, 33.2, rec["certified_below_gy"]) == "feasible_witness"


def test_certificate_needs_an_external_cap_and_reports_an_infeasible_relaxation() -> None:
    case = hand_case()
    no_ext = GoalList(name="x", goals=[g for g in GOALS.goals if g.structure != "External"])
    rec = certify(case, no_ext)
    assert rec["certified_below_gy"] is None and "External" in rec["reason"]
    # Target voxels at 0.5 x w need w >= 44.3 for the mean row; a cord voxel at 2 x w then sits
    # at 88.6 Gy, above the 77.05 Gy cap: no plan meets the other hard goals at all.
    bad = hand_case(n_cord=3, target_coef=0.5, cord_coef=2.0)
    rec = certify(bad, GOALS)
    assert rec["relaxation_infeasible"] and rec["certified_below_gy"] is None and rec["lower_bound_gy"] is None
    # A goal with more merged voxels outside the mask than it allows below its level.
    case = hand_case()
    case.structures["PTV_7000"] = case.structures["PTV_7000"].model_copy(update={"n_outside": 5})
    assert "unmeetable" in build_relaxation(case, GOALS)["reason"]


def test_certificate_sits_below_the_witness_on_the_structured_case() -> None:
    case = structured_case()
    floor = coverage_floor(case, GOALS)
    assert floor["others_met"] and floor["best_achieved_gy"] > 5.0
    rec = certify(case, GOALS, keep_w=True)
    assert "reason" not in rec and rec["lp_rows"] > 0 and rec["tails"] and not rec["relaxation_infeasible"]
    # A positive certificate, below the witness; every LP tail mean at most the witness plan's.
    assert rec["lower_bound_gy"] is not None and 0 < rec["lower_bound_gy"] <= floor["best_achieved_gy"] + 1e-6
    assert rec["certified_below_gy"] == pytest.approx(rec["lower_bound_gy"] - 0.05, abs=1e-6)
    cord = case.structure_dose(case.dose(floor["witness_w"]), "SpinalCord")
    for t in rec["tails"]:
        assert t["status"] == 0 and t["tail_mean_gy"] <= np.sort(cord)[-t["k"] :].mean() + 1e-6
    # The LP's own plans meet the relaxed constraints: no cord or target voxel above the cap,
    # the whole-target mean at least (N - m) c' / N, and the lower tails at their levels.
    tinfo = rec["targets"][0]
    idx = case.structures["PTV_7000"].mask_idx
    for w in rec["tail_w"].values():
        dose = case.dose(w)
        assert dose[case.structures["SpinalCord"].mask_idx].max() <= rec["e_eff_gy"] + 1e-6
        assert dose[idx].max() <= rec["e_eff_gy"] + 1e-6
        assert dose[idx].sum() >= (tinfo["n_merged"] - tinfo["cold_allowance"]) * tinfo["c_eff_gy"] - 1e-6
        for lv in tinfo["levels"]:
            if lv["scope"] == "all":
                assert np.sort(dose[idx])[: lv["j"]].sum() >= (lv["j"] - tinfo["cold_allowance"]) * tinfo["c_eff_gy"] - 1e-6
    assert rec["best_lp_w"] is not None and (rec["best_lp_w"][case.n_beamlets - 8 :] >= 0).all()
    # Every LP fluence was checked against the other hard goals.
    assert len(rec["lp_plans"]) == len(rec["tails"]) and all("others_met" in p and "organ_gy" in p for p in rec["lp_plans"])


def test_random_synthetic_case_builds_and_reports() -> None:
    case = with_external(make_case(n_vox=120, n_beamlets=24, seed=21))
    rec = certify(case, GOALS)
    assert "reason" not in rec and rec["lp_rows"] > 0 and rec["tails"]
    assert isinstance(rec["relaxation_infeasible"], bool) and "lower_bound_gy" in rec


def test_infeasible_limit_uses_the_certificate_when_the_fraction_is_not_certified() -> None:
    assert infeasible_limit(20.0) == 14.0
    assert infeasible_limit(20.0, certified_below_gy=15.0) == 14.0  # 0.7 x best is already certified
    lim = infeasible_limit(20.0, certified_below_gy=10.03)
    assert lim < 10.03 and lim == pytest.approx(9.9) and feasibility_status(lim, 20.0, 10.03) == "certified_infeasible"
    assert infeasible_limit(20.0, certified_below_gy=None) == 14.0


def test_default_goals_relax_every_hard_target_and_keep_only_one_external_row_set() -> None:
    case = with_external(make_case())
    case.structures["PTV_6300"] = Structure(name="PTV_6300", raw_name="PTV63", mask_idx=np.arange(0, 30, dtype=np.int32), volume_cc=0.9)
    case.prescriptions["PTV_6300"] = 63.0
    relax = build_relaxation(case, GOALS)  # the default list carries PTV_6300 D99 >= 59.85
    names = [t["structure"] for t in relax["targets"]]
    assert names == ["PTV_7000", "PTV_6300"] and relax["targets"][1]["n_merged"] == 30  # merged with PTV_7000
    assert relax["n_rows"] == relax["A"].shape[0] and relax["A"].shape[1] == relax["n_var"] + relax["n_aux"]


def test_witness_check_and_repair_agree_with_the_sweep() -> None:
    from opengray.physics.certificate import repair_witness, witness_check

    case = structured_case()
    floor = coverage_floor(case, GOALS)
    assert floor["others_met"]
    chk = witness_check(case, GOALS, floor["witness_w"])
    assert chk["others_met"] and chk["organ_gy"] == pytest.approx(floor["best_achieved_gy"], abs=2e-3)
    assert set(chk["goals"]) == {g.label() for g in GOALS.for_case(case).hard if g.structure != "SpinalCord"}
    rep = repair_witness(case, GOALS, floor["witness_w"], target_gy=floor["best_achieved_gy"], priorities=(1000.0, 10.0))
    assert len(rep["sweep"]) == 4 and all(set(s) >= {"level", "target_priority", "organ_gy", "others_met"} for s in rep["sweep"])
    if rep["best"] is not None:
        again = witness_check(case, GOALS, rep["best"]["w"])
        assert again["others_met"] and again["organ_gy"] == pytest.approx(rep["best"]["organ_gy"], abs=1e-6)
    # The repair from the LP fluence finds a witness no worse than the sweep's on this case.
    rec = certify(case, GOALS, keep_w=True)
    rep2 = repair_witness(case, GOALS, rec["best_lp_w"], target_gy=floor["best_achieved_gy"])
    assert rep2["best"] is not None and rep2["best"]["organ_gy"] >= rec["lower_bound_gy"]
