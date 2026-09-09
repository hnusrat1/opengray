import numpy as np
import pytest

from opengray.agents.base import AgentSpec
from opengray.env.escalation import NotConstructible, present_tight
from opengray.env.tracks import EpisodeSpec, track_config
from opengray.goals.defaults import openkbp_default_goals
from opengray.physics.feasibility import check_tight_witness
from opengray.runner.run import RunConfig, build_presentation
from tests.test_solver import make_case


def test_missing_or_invalid_witness_is_not_a_feasible_control():
    case = make_case(n_vox=120, n_beamlets=24, seed=21)
    goals = openkbp_default_goals()
    rec = {"best_achieved_gy": 20.0, "others_met": True}
    assert not check_tight_witness(case, goals, rec, None)["valid"]
    rec["witness_w"] = np.zeros(case.n_beamlets)
    audit = check_tight_witness(case, goals, rec, None)
    assert not audit["valid"] and any(not g["met"] for g in audit["goals"])
    with pytest.raises(NotConstructible, match="saved fluence"):
        present_tight(case, goals, {"SpinalCord": {"best_achieved_gy":20.0,"witness_verified":False}})


def test_certificate_only_construction_does_not_launch_a_feasibility_solve(tmp_path, monkeypatch):
    from tests.test_track3_track5 import overlap_case
    def unexpected(*args, **kwargs):
        raise AssertionError("unnecessary feasibility solve")
    monkeypatch.setattr("opengray.runner.run.floor_for", unexpected)
    case = overlap_case()
    cfg = RunConfig(track=track_config("T5"), agent=AgentSpec("controller"), split="validation",
                    seeds=[0], out_dir=tmp_path, split_file=tmp_path/"split.json")
    spec = EpisodeSpec(episode_id="x", track="T5", k=3, case_id=case.case_id,
                       split="validation", seed=0, transform="coverage_cap")
    pres = build_presentation(spec, case, openkbp_default_goals(), cfg)
    assert pres.meta["label_status"] == "certified_infeasible"
