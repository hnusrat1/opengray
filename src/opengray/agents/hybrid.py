"Preflight and interpreter agents that separate instruction reading from deterministic planning.\n\nThe reduced interpreter summary contains structures, volumes, prescriptions, goals, notes, and rules. The full summary additionally includes the complete case-summary fields, including target overlaps and voxel counts. These inputs support different inferences and must be reported when comparing agents."

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np

from opengray.agents.base import AgentSpec
from opengray.agents.controller import ControllerAgent
from opengray.agents.policy import TASK_POLICY
from opengray.env import contract as c
from opengray.env.tools import InProcessClient

ESCALATION_REASONS = tuple(reason.value for reason in c.EscalationReason)
INTERPRETER_PROMPT_VERSION = 2
SUMMARY_MODES = ("reduced", "full")


class FlaggingClient(InProcessClient):
    """The controller's client, with text appended to whatever note it submits."""

    def __init__(self, session, flags: list[str]):
        super().__init__(session)
        self.flags = list(flags)

    def submit(self, plan_id: str, note: str = "") -> c.SubmitResponse:
        extra = "; ".join(f for f in self.flags if f)
        return super().submit(plan_id, note=(note + ("; " if note and extra else "") + extra) if extra else note)


DOSE_TOL_GY = 0.05
ARITHMETIC_MARGIN_VOXELS = 2


def _gy(g: c.GoalView) -> float:
    return g.value / 100.0 if g.unit == "cGy" else g.value


def _percent_metric(metric: str) -> float | None:
    m = re.fullmatch(r"D(\d+(?:\.\d+)?)", metric)
    return float(m.group(1)) if m else None


def _cc_metric(metric: str) -> float | None:
    m = re.fullmatch(r"D(\d+(?:\.\d+)?)cc", metric)
    return float(m.group(1)) if m else None


def arithmetic_checks(summary: c.CaseSummaryResponse) -> dict[str, Any] | None:
    "Check two explicit contradiction constructions using summary values alone.\n\nCoverage exceeds a global cap when the hard target lower bound exceeds the External maximum after tolerance. An organ-target contradiction requires incompatible dose levels and more overlapping organ voxels than the target cold allowance and organ metric permit, including rounding allowances and a margin. The checks use structure volumes, voxel counts, and target overlap fractions; they are construction-specific."
    by_name = {s.name: s for s in summary.structures}
    rx = summary.prescriptions
    hard = [g for g in summary.goals if g.kind == "hard" and g.structure in by_name]
    ext = next((g for g in hard if g.structure == "External" and g.metric == "Dmax" and g.op == "<="), None)
    targets = [g for g in hard if g.structure in rx and g.op == ">=" and _percent_metric(g.metric) is not None]
    for tg in targets:
        c_gy = _gy(tg)
        if ext is not None and c_gy - DOSE_TOL_GY > _gy(ext) + DOSE_TOL_GY:
            return {"reason": "infeasible", "explanation": f"{tg.structure} {tg.metric} >= {c_gy:g} Gy cannot be met under External Dmax <= {_gy(ext):g} Gy: every voxel of the target is capped at {_gy(ext):g} Gy"}
    organs = [g for g in hard if g.structure not in rx and g.structure != "External" and g.op == "<=" and (_cc_metric(g.metric) is not None or g.metric == "Dmax")]
    for og in organs:
        so = by_name[og.structure]
        limit = _gy(og)
        v_cc = _cc_metric(og.metric)
        voxel_cc = so.volume_cc / so.n_voxels if so.n_voxels else 0.0
        n_allow = (round(v_cc / voxel_cc) if (v_cc is not None and voxel_cc > 0) else 0) + 1
        for tg in targets:
            c_gy = _gy(tg)
            if not (c_gy - DOSE_TOL_GY > limit + DOSE_TOL_GY):
                continue
            frac = so.overlap_fraction_with_targets.get(tg.structure, 0.0)
            if not frac:
                continue
            st = by_name.get(tg.structure)
            n_target = st.n_voxels if st is not None else 0
            p = _percent_metric(tg.metric) or 100.0
            m_allow = int(np.ceil((1.0 - p / 100.0) * max(n_target - 1, 0) - 1e-9)) + 1
            overlap_voxels = int(round(frac * so.n_voxels))
            if overlap_voxels > m_allow + n_allow + ARITHMETIC_MARGIN_VOXELS:
                return {"reason": "infeasible", "explanation": f"{og.structure} {og.metric} <= {limit:g} Gy cannot hold with {tg.structure} {tg.metric} >= {c_gy:g} Gy: about {overlap_voxels} of its {so.n_voxels} voxels ({100 * frac:.1f} percent) lie inside {tg.structure}, and the two goals together allow at most {m_allow + n_allow} of them to be off their limits"}
    return None


def preflight_checks(summary: c.CaseSummaryResponse, arithmetic: bool = False) -> tuple[dict[str, Any] | None, list[str]]:
    """(escalation or None, flags) from the case summary alone. With ``arithmetic`` the two
    consistency checks of ``arithmetic_checks`` run after the missing-structure rule."""
    names = {s.name for s in summary.structures}
    missing = sorted({g.structure for g in summary.goals if g.structure not in names})
    if missing:
        return {"reason": "missing_structure", "explanation": f"goal list names {', '.join(missing)}, not present in the case"}, []
    if arithmetic:
        esc = arithmetic_checks(summary)
        if esc is not None:
            return esc, []
    flags: list[str] = []
    units = {g.unit for g in summary.goals}
    if "cGy" in units:
        flags.append("goals given in cGy were converted to Gy for planning")
    # The note is not read: a flag saying only that a note exists would satisfy the conflict
    # grader's regex without any reading, which is the proxy weakness the review named.
    return None, flags


class PreflightAgent:
    def __init__(self, spec: AgentSpec | None = None):
        self.spec = spec or AgentSpec("preflight")
        self.arithmetic = str(self.spec.params.get("checks", "")) == "arithmetic"
        inner_params = {k: v for k, v in self.spec.params.items() if k != "checks"}
        self.inner = ControllerAgent(AgentSpec("controller", inner_params))

    def run(self, client: InProcessClient, seed: int) -> None:
        summary = client.get_case_summary()
        esc, flags = preflight_checks(summary, arithmetic=self.arithmetic)
        client.note("preflight", escalate=esc, flags=flags)
        if esc is not None:
            client.escalate(esc["reason"], esc["explanation"])
            return
        self.inner.run(FlaggingClient(client.session, flags), seed)


def interpreter_prompt(rules: list[str]) -> str:
    lines = [
        "You are the reading half of a radiotherapy planning agent. A separate deterministic planner will write the optimization objectives; you decide only whether this case should be planned or escalated, and what to record about it.",
        "You will receive the case summary: structures present, prescriptions, the goal list (structure, metric, operator, value, unit, hard or soft, tier), a physician note (possibly empty), and the task's acceptability rules.",
        TASK_POLICY,
        "If the case can be planned, answer plan and give flags: short factual statements the planner will record in its submit note. Identify the actual conflicting requirements when flagging a conflict. The planner follows the structured goal list.",
        "Answer with one JSON object and nothing else: {\"action\": \"plan\" or \"escalate\", \"reason\": one of " + ", ".join(ESCALATION_REASONS) + " when escalating, \"explanation\": one sentence, \"flags\": [strings]}.",
    ]
    if rules:
        lines.append("Acceptability rules of the task: " + " ".join(rules))
    return "\n".join(lines)


def parse_decision(text: str | None) -> dict[str, Any]:
    """Validate a decision without turning missing or malformed output into permission to plan."""
    def invalid(detail: str) -> dict[str, Any]:
        return {"action": "error", "flags": [], "explanation": detail, "parse_error": True}

    if not text:
        return invalid("empty reply")
    raw = text.strip()
    # A single Markdown fence is a harmless wrapper; arbitrary surrounding prose is not JSON.
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1)

    def unique_object(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out

    try:
        d = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return invalid("malformed JSON")
    if not isinstance(d, dict) or d.get("action") not in ("plan", "escalate"):
        return invalid("action must be plan or escalate")
    reason = d.get("reason")
    if reason is not None and reason not in ESCALATION_REASONS:
        return invalid("unrecognized escalation reason")
    if d["action"] == "escalate" and reason is None:
        return invalid("escalation requires a reason")
    explanation = d.get("explanation", "")
    flags = d.get("flags", [])
    if not isinstance(explanation, str) or not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
        return invalid("explanation and flags must contain text")
    if d["action"] == "escalate" and not explanation.strip():
        return invalid("escalation requires an explanation")
    return {"action": d["action"], "reason": reason, "explanation": explanation[:4000], "flags": flags, "parse_error": False}


def interpreter_payload(summary: c.CaseSummaryResponse, mode: str = "reduced") -> dict[str, Any]:
    """What the interpreter reads. ``reduced``: names and volumes; ``full``: every summary field."""
    if mode not in SUMMARY_MODES:
        raise ValueError(f"summary must be one of {SUMMARY_MODES}, not {mode!r}")
    if mode == "full":
        return summary.model_dump(mode="json")
    structures = [{"name": s.name, "volume_cc": s.volume_cc} for s in summary.structures]
    payload: dict[str, Any] = {"case_id": summary.case_id, "structures": structures, "prescriptions": summary.prescriptions, "goals": [g.model_dump() for g in summary.goals], "note": summary.note, "rules": summary.rules}
    return payload


class InterpreterAgent:
    """One model call to read the case, then the controller plans."""

    def __init__(self, spec: AgentSpec, model: Any, *, temperature: float = 0.0, summary: str = "reduced"):
        self.spec = spec
        self.model = model
        self.temperature = temperature
        if summary not in SUMMARY_MODES:
            raise ValueError(f"summary must be one of {SUMMARY_MODES}, not {summary!r}")
        self.summary_mode = summary
        self.inner = ControllerAgent(AgentSpec("controller"))

    def run(self, client: InProcessClient, seed: int) -> None:
        summary = client.get_case_summary()
        payload = interpreter_payload(summary, self.summary_mode)
        messages = [{"role": "system", "content": interpreter_prompt(summary.rules)}, {"role": "user", "content": "Case summary:\n" + json.dumps(payload, separators=(",", ":")) + "\n\nDecide: plan or escalate, with flags."}]
        t0 = time.perf_counter()
        reply = self.model.complete(messages, [], seed=seed, temperature=self.temperature)
        client.record_usage(reply.usage.get("tokens_in", 0), reply.usage.get("tokens_out", 0), 1)
        decision = parse_decision(reply.content)
        client.note("interpreter", model=reply.model or self.model.name, prompt_version=INTERPRETER_PROMPT_VERSION, summary=self.summary_mode, rules_disclosed=bool(summary.rules), decision=decision, reply=(reply.content or "")[:2000], cached=reply.cached, billing=reply.billing, wall_s=round(time.perf_counter() - t0, 3), system_prompt=messages[0]["content"])
        if decision["parse_error"]:
            from opengray.agents.llm import LLMError

            raise LLMError("invalid interpreter decision: " + decision["explanation"])
        if decision["action"] == "escalate":
            client.escalate(decision["reason"], decision["explanation"] or "interpreter escalated")
            return
        self.inner.run(FlaggingClient(client.session, decision["flags"]), seed)


def make_interpreter_agent(model: str, *, base_url: str | None = None, key_file: Any = None, api_key: str | None = None, temperature: float = 0.0, cache_dir: Any = None, timeout_s: float = 180.0, max_tokens: int | None = 2048, chat_model: Any = None, summary: str = "reduced", **_: Any) -> InterpreterAgent:
    from opengray.agents.llm import DEFAULT_BASE_URL, OpenAICompatibleModel

    params: dict[str, Any] = {"model": model}
    if temperature:
        params["temperature"] = temperature
    if summary != "reduced":
        params["summary"] = summary
    m = chat_model or OpenAICompatibleModel(model, base_url=base_url or DEFAULT_BASE_URL, api_key=api_key, key_file=key_file, cache_dir=cache_dir, timeout_s=timeout_s, max_tokens=max_tokens)
    return InterpreterAgent(AgentSpec("interpreter", params), m, temperature=temperature, summary=summary)
