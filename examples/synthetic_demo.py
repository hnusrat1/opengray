"""Run a tiny fictional planning episode without downloaded data or model calls."""

from __future__ import annotations

import json

import numpy as np
import scipy.sparse as sp

from opengray.agents.heuristic import HeuristicAgent
from opengray.data.base import Case, Structure
from opengray.env.core import PlanningSession
from opengray.env.tools import InProcessClient
from opengray.env.tracks import track_config
from opengray.goals.defaults import openkbp_default_goals


def synthetic_case() -> Case:
    """An invented two-region matrix, with no patient anatomy or reference plan."""
    target = np.arange(12, dtype=np.int32)
    organ = np.arange(12, 24, dtype=np.int32)
    matrix = np.vstack([np.tile([1.0, 0.2], (12, 1)), np.tile([0.1, 0.3], (12, 1))])
    return Case(
        case_id="synthetic-demo",
        cohort="synthetic",
        feasible_idx=np.arange(24, dtype=np.int64),
        voxel_volume_cc=np.full(24, 0.1),
        D=sp.csr_matrix(matrix),
        structures={
            "PTV_7000": Structure(name="PTV_7000", raw_name="PTV70", mask_idx=target, volume_cc=1.2),
            "SpinalCord": Structure(name="SpinalCord", raw_name="SpinalCord", mask_idx=organ, volume_cc=1.2),
        },
        prescriptions={"PTV_7000": 70.0},
        provenance={"source": "constructed numerical example; no patient data"},
    )


def main() -> None:
    session = PlanningSession(case=synthetic_case(), goals=openkbp_default_goals(), track=track_config("T2", 3), episode_id="synthetic-demo")
    HeuristicAgent().run(InProcessClient(session), seed=0)
    terminal = session.terminal
    assert session.status == "submitted", terminal
    assert 1 <= session.optimize_calls <= 3
    score = terminal["score"]
    print(json.dumps({
        "example": "Synthetic software demonstration; not a clinical plan or patient result",
        "status": session.status,
        "optimization_calls": session.optimize_calls,
        "optimization_budget": 3,
        "plan_score": score["plan_score"],
        "goals": [{k: g[k] for k in ("structure", "metric", "limit", "achieved", "met")} for g in score["goals"]],
        "recorded_events": len(session.events),
    }, indent=2))


if __name__ == "__main__":
    main()
