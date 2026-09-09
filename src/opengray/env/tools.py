"Validated in-process tool dispatch for planning sessions."

from __future__ import annotations

from typing import Any

from opengray.env import contract as c
from opengray.env.core import PlanningSession


def dispatch(session: PlanningSession, tool: str, args: dict[str, Any] | None = None, door: str = "in_process") -> dict[str, Any]:
    resp = session.call(tool, args or {}, door=door)
    return resp.model_dump(mode="json")


class ToolError(Exception):
    def __init__(self, error: c.ErrorResponse):
        super().__init__(f"{error.error}: {error.detail}")
        self.error = error


class InProcessClient:
    """Typed access to the nine tools of one session; raises ToolError on error responses.

    ``call_json`` is the untyped path used by LLM agents: any tool name and a dict of
    arguments in, the serialized response (or error) out, never raising. ``note`` and
    ``record_usage`` let an agent put its transcript and token counts into the episode log.
    The MCP client implements the same three methods so the LLM agent runs unchanged
    through either door.
    """

    door = "in_process"

    def __init__(self, session: PlanningSession, raise_on_error: bool = True):
        self.session = session
        self.raise_on_error = raise_on_error

    def _call(self, tool: c.ToolName, args: dict[str, Any]):
        resp = self.session.call(tool.value, args, door=self.door)
        if isinstance(resp, c.ErrorResponse) and self.raise_on_error:
            raise ToolError(resp)
        return resp

    def call_json(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return dispatch(self.session, tool, args, door=self.door)

    def note(self, event: str, **payload: Any) -> None:
        self.session.note(event, door=self.door, **payload)

    def record_usage(self, tokens_in: int = 0, tokens_out: int = 0, model_calls: int = 0) -> None:
        self.session.record_usage(tokens_in, tokens_out, model_calls)

    def get_case_summary(self) -> c.CaseSummaryResponse:
        return self._call(c.ToolName.get_case_summary, {})

    def get_metrics(self, plan_id: str, metrics: list[dict[str, str]]) -> c.GetMetricsResponse:
        return self._call(c.ToolName.get_metrics, {"plan_id": plan_id, "metrics": metrics})

    def get_dvh(self, plan_id: str, structure: str, n_points: int = 50) -> c.GetDvhResponse:
        return self._call(c.ToolName.get_dvh, {"plan_id": plan_id, "structure": structure, "n_points": n_points})

    def set_objectives(self, objectives: list[dict[str, Any]]) -> c.SetObjectivesResponse:
        return self._call(c.ToolName.set_objectives, {"objectives": objectives})

    def optimize(self, objective_id: str) -> c.OptimizeResponse:
        return self._call(c.ToolName.optimize, {"objective_id": objective_id})

    def normalize(self, plan_id: str, structure: str, metric: str, value: float) -> c.NormalizeResponse:
        return self._call(c.ToolName.normalize, {"plan_id": plan_id, "structure": structure, "metric": metric, "value": value})

    def compare_to_goals(self, plan_id: str) -> c.CompareToGoalsResponse:
        return self._call(c.ToolName.compare_to_goals, {"plan_id": plan_id})

    def submit(self, plan_id: str, note: str = "") -> c.SubmitResponse:
        return self._call(c.ToolName.submit, {"plan_id": plan_id, "note": note})

    def escalate(self, reason: str, explanation: str = "") -> c.EscalateResponse:
        return self._call(c.ToolName.escalate, {"reason": reason, "explanation": explanation})
