"Protocol identities covering solver, score, goals, grading, schemas, presentation, and prompts."

from __future__ import annotations

import hashlib
import json
from typing import Any

from opengray.env import contract as c
from opengray.env.tracks import TrackConfig
from opengray.goals.schema import GoalList
from opengray.goals.scoring import GATE_ATOL_GY, SCORING_VERSION, ScoreWeights
from opengray.physics.solver import SolverConfig, fingerprint

LEGACY = "legacy"
GRADER_VERSION_T4 = "2026-09-06.1"  # name_injection graded on the submitted plan's objective
GRADER_VERSION_T5 = "2026-09-08.1"  # escalation credited with infeasible or contradictory_instructions; refutation needs both goals met
PRESENTATION_VERSION = "2026-09-08.1"  # a changed goal's note is recomputed with its value (Track 3 relax, Track 5 coverage arm)


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def contract_hash() -> str:
    schemas = {name.value: {"request": c.REQUEST_MODELS[name].model_json_schema(), "response": c.RESPONSE_MODELS[name].model_json_schema()} for name in c.ToolName}
    return _digest(schemas)


def describe(track: TrackConfig, goals: GoalList, weights: ScoreWeights, solver: SolverConfig | None = None, prompt: str | None = None, merge_targets: bool = True) -> dict[str, Any]:
    return {
        "scoring_version": SCORING_VERSION,
        "gate_numeric_tolerance_gy": GATE_ATOL_GY,
        "weights": weights.model_dump(),
        "merge_targets": merge_targets,
        "goals_hash": _digest([g.model_dump() for g in goals.goals]),
        "goals_name": goals.name,
        "solver_hash": _digest(fingerprint(solver)),
        "contract_hash": contract_hash(),
        "grader_t4": GRADER_VERSION_T4,
        "grader_t5": GRADER_VERSION_T5,
        "presentation": PRESENTATION_VERSION,
        "track": track.model_dump(),
        "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:16] if prompt else None,
    }


def protocol_id(components: dict[str, Any]) -> str:
    return _digest(components)[:12]


def protocol_for(track: TrackConfig, goals: GoalList, weights: ScoreWeights, solver: SolverConfig | None = None, prompt: str | None = None, merge_targets: bool = True) -> dict[str, Any]:
    comp = describe(track, goals, weights, solver, prompt, merge_targets)
    return {"protocol_id": protocol_id(comp), **comp}
