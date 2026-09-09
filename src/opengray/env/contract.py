"Typed requests and responses for the planning tool interface."

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from opengray.physics.objectives import PRIORITY_MAX, TermType


class ToolName(StrEnum):
    get_case_summary = "get_case_summary"
    get_metrics = "get_metrics"
    get_dvh = "get_dvh"
    set_objectives = "set_objectives"
    optimize = "optimize"
    normalize = "normalize"
    compare_to_goals = "compare_to_goals"
    submit = "submit"
    escalate = "escalate"


class EscalationReason(StrEnum):
    infeasible = "infeasible"
    missing_structure = "missing_structure"
    contradictory_instructions = "contradictory_instructions"
    unit_ambiguity = "unit_ambiguity"
    other = "other"


# Shared views -------------------------------------------------------------------------------


class GoalView(BaseModel):
    structure: str
    metric: str
    op: Literal["<=", ">="]
    value: float
    unit: str
    kind: Literal["hard", "soft"]
    tier: int
    note: str = ""


class GoalStatus(GoalView):
    achieved: float | None = Field(default=None, description="In the goal's unit; None if not computable")
    met: bool | None = None


class BudgetView(BaseModel):
    optimize_total: int
    optimize_remaining: int
    tool_calls_made: int
    tool_calls_max: int | None = None


class StructureSummary(BaseModel):
    name: str
    volume_cc: float
    n_voxels: int
    is_target: bool
    prescription_gy: float | None = None
    overlap_fraction_with_targets: dict[str, float] = Field(default_factory=dict, description="fraction of this structure's voxels inside each target")
    min_distance_mm_to_targets: dict[str, float] = Field(default_factory=dict)


class SolverStats(BaseModel):
    iterations: int
    converged: bool
    wall_s: float
    cached: bool = False
    warm_start: bool = Field(default=False, description="True when the solve started from the episode's previous plan rather than from zero fluence")
    spg: float | None = Field(default=None, description="Fluence complexity of the plan (OpenKBP-Opt's sum of positive gradients; its reference plans were limited to 65). Reported for every plan; only the deliverable solver limits it")
    complexity_rounds: int = Field(default=0, description="Rounds the deliverable solver needed to bring the plan's SPG within its limit (0 under the default solver)")


# Requests and responses --------------------------------------------------------------------


class GetCaseSummaryRequest(BaseModel):
    pass


class CaseSummaryResponse(BaseModel):
    case_id: str
    track: str
    note: str = Field(default="", description="Free-text case note from the physician (may be empty)")
    structures: list[StructureSummary]
    prescriptions: dict[str, float]
    beam_geometry: dict[str, Any]
    goals: list[GoalView]
    rules: list[str] = Field(default_factory=list, description="Acceptability rules of the task, stated to every agent (empty when the track withholds them)")
    objective_terms: list[str] = Field(description="Term types the optimizer accepts")
    budget: BudgetView


class MetricQuery(BaseModel):
    structure: str
    metric: str


class GetMetricsRequest(BaseModel):
    plan_id: str
    metrics: list[MetricQuery] = Field(min_length=1, max_length=64)


class MetricValue(MetricQuery):
    value: float | None
    unit: str


class GetMetricsResponse(BaseModel):
    plan_id: str
    values: list[MetricValue]


class GetDvhRequest(BaseModel):
    plan_id: str
    structure: str
    n_points: int = Field(default=50, ge=5, le=500)


class GetDvhResponse(BaseModel):
    plan_id: str
    structure: str
    dose_gy: list[float]
    volume_fraction: list[float]


class ObjectiveTermSpec(BaseModel):
    structure: str
    term: TermType
    level: float = Field(ge=0.0, description="Gy")
    priority: float = Field(ge=0.0, le=PRIORITY_MAX)
    volume_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)


class SetObjectivesRequest(BaseModel):
    objectives: list[ObjectiveTermSpec] = Field(min_length=1, max_length=64)


class SetObjectivesResponse(BaseModel):
    objective_id: str
    valid: bool
    problems: list[str] = Field(default_factory=list)
    n_terms: int


class OptimizeRequest(BaseModel):
    objective_id: str


class OptimizeResponse(BaseModel):
    plan_id: str
    goals: list[GoalStatus]
    hard_met: int
    hard_total: int
    soft_met: int
    soft_total: int
    solver: SolverStats
    budget: BudgetView
    score: float | None = Field(default=None, description="Only present when the track exposes the score")
    notice: str | None = Field(default=None, description="Track 3: set once, when the goal list changed after this optimize; the goals above are already evaluated against the new list")


class NormalizeRequest(BaseModel):
    plan_id: str
    structure: str
    metric: str
    value: float = Field(gt=0.0, description="Target value for the metric, Gy")


class NormalizeResponse(BaseModel):
    plan_id: str
    source_plan_id: str
    scale_factor: float
    goals: list[GoalStatus]
    hard_met: int
    hard_total: int
    budget: BudgetView
    score: float | None = Field(default=None, description="Only present when the track exposes the score")


class CompareToGoalsRequest(BaseModel):
    plan_id: str


class CompareToGoalsResponse(BaseModel):
    plan_id: str
    goals: list[GoalStatus]
    hard_met: int
    hard_total: int
    soft_met: int
    soft_total: int


class SubmitRequest(BaseModel):
    plan_id: str
    note: str = Field(default="", max_length=4000)


class SubmitResponse(BaseModel):
    episode_id: str
    status: Literal["submitted"]
    plan_id: str


class EscalateRequest(BaseModel):
    reason: EscalationReason
    explanation: str = Field(default="", max_length=4000)


class EscalateResponse(BaseModel):
    episode_id: str
    status: Literal["escalated"]
    reason: EscalationReason


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


REQUEST_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.get_case_summary: GetCaseSummaryRequest,
    ToolName.get_metrics: GetMetricsRequest,
    ToolName.get_dvh: GetDvhRequest,
    ToolName.set_objectives: SetObjectivesRequest,
    ToolName.optimize: OptimizeRequest,
    ToolName.normalize: NormalizeRequest,
    ToolName.compare_to_goals: CompareToGoalsRequest,
    ToolName.submit: SubmitRequest,
    ToolName.escalate: EscalateRequest,
}

RESPONSE_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.get_case_summary: CaseSummaryResponse,
    ToolName.get_metrics: GetMetricsResponse,
    ToolName.get_dvh: GetDvhResponse,
    ToolName.set_objectives: SetObjectivesResponse,
    ToolName.optimize: OptimizeResponse,
    ToolName.normalize: NormalizeResponse,
    ToolName.compare_to_goals: CompareToGoalsResponse,
    ToolName.submit: SubmitResponse,
    ToolName.escalate: EscalateResponse,
}

TOOL_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.get_case_summary: "Describe the case: structures with volumes, overlaps and distances to targets, prescriptions, beam geometry, the goal list, and the remaining budget.",
    ToolName.get_metrics: "Achieved values of the requested (structure, metric) pairs for a plan.",
    ToolName.get_dvh: "Cumulative dose-volume histogram of one structure for a plan.",
    ToolName.set_objectives: "Register an optimization objective: a list of (structure, term, level, priority) terms. Returns an objective_id; does not optimize.",
    ToolName.optimize: "Run the optimizer for an objective_id. Consumes one unit of the optimize budget. Returns a plan_id and every goal's achieved value.",
    ToolName.normalize: "Rescale a plan so that a structure metric equals a value (linear rescale of all beamlets). Returns a new plan_id; free of budget.",
    ToolName.compare_to_goals: "Goal-by-goal table for a plan.",
    ToolName.submit: "Submit a plan as the final answer and end the episode.",
    ToolName.escalate: "Declare that the case cannot or should not be planned as given, with a reason, and end the episode.",
}
