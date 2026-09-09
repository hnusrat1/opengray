"Default research goals based on OpenKBP-Opt criteria.\n\nThe list includes target coverage, organ limits, and a visible External maximum-dose goal. The scalar score and its rejection gates are defined separately in scoring.py."

from __future__ import annotations

from opengray.goals.schema import Goal, GoalList


def openkbp_default_goals() -> GoalList:
    g = [
        Goal(structure="External", metric="Dmax", op="<=", value=77.0, unit="Gy", kind="hard", tier=1, note="hot-spot rule: 110% of the 70 Gy prescription, evaluated over the dose-feasible region"),
        Goal(structure="PTV_7000", metric="D99", op=">=", value=66.5, unit="Gy", kind="hard", tier=1, note="95% of 70 Gy"),
        Goal(structure="PTV_6300", metric="D99", op=">=", value=59.85, unit="Gy", kind="hard", tier=1, note="95% of 63 Gy"),
        Goal(structure="PTV_5600", metric="D99", op=">=", value=53.2, unit="Gy", kind="hard", tier=1, note="95% of 56 Gy"),
        Goal(structure="Brainstem", metric="D0.1cc", op="<=", value=50.0, unit="Gy", kind="hard", tier=1, note="serial organ, absolute limit"),
        Goal(structure="SpinalCord", metric="D0.1cc", op="<=", value=45.0, unit="Gy", kind="hard", tier=1, note="serial organ, absolute limit"),
        Goal(structure="Bone_Mandible", metric="D0.1cc", op="<=", value=73.5, unit="Gy", kind="hard", tier=1, note="serial organ, absolute limit"),
        Goal(structure="Parotid_L", metric="Dmean", op="<=", value=26.0, unit="Gy", kind="soft", tier=2, note="parallel organ"),
        Goal(structure="Parotid_R", metric="Dmean", op="<=", value=26.0, unit="Gy", kind="soft", tier=2, note="parallel organ"),
        Goal(structure="Esophagus", metric="Dmean", op="<=", value=45.0, unit="Gy", kind="soft", tier=2, note="parallel organ"),
        Goal(structure="Larynx", metric="Dmean", op="<=", value=45.0, unit="Gy", kind="soft", tier=2, note="parallel organ"),
    ]
    return GoalList(name="openkbp_v1", goals=g)
