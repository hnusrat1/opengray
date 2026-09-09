"Tool-calling language-model agent using an OpenAI-compatible chat endpoint.\n\nThe agent receives the case summary, calls the planning tools, and terminates by submission or escalation. Behavioral stopping without an explicit terminal action is recorded as automatic submission when a plan exists. Provider errors produce error records. Requests use temperature and seed settings, whose support depends on the provider. Requests and responses are recorded for reproducibility; credentials are excluded."

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx

from opengray.agents.base import AgentSpec
from opengray.agents.heuristic import plan_quality
from opengray.env import contract as c

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_KEY_FILE = Path(".secrets/openrouter.key")
ENV_KEY = "OPENROUTER_API_KEY"
TERMINAL_TOOLS = {c.ToolName.submit.value, c.ToolName.escalate.value}
PLAN_TOOLS = {c.ToolName.optimize.value, c.ToolName.normalize.value, c.ToolName.compare_to_goals.value}


class LLMError(RuntimeError):
    pass


# ---- model interface ------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as the model produced it


@dataclass
class ModelReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    cached: bool = False
    wall_s: float = 0.0

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}} for tc in self.tool_calls
            ]
        return msg


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, seed: int, temperature: float) -> ModelReply: ...


def load_api_key(key_file: Path | str | None = None, env_var: str = ENV_KEY) -> str:
    """Environment variable first, then the key file. The key never appears in logs or results."""
    v = os.environ.get(env_var, "").strip()
    if v:
        return v
    path = Path(key_file) if key_file else DEFAULT_KEY_FILE
    if path.exists():
        v = path.read_text().strip()
        if v:
            return v
    raise LLMError(f"no API key: set {env_var} or put the key in {path}")


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline pydantic's ``$defs`` so tool schemas are plain JSON Schema objects."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                merged = dict(defs[name])
                merged.update({k: v for k, v in node.items() if k != "$ref"})
                return walk(merged)
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    out = walk(schema)
    out.pop("title", None)
    return out


def tool_schemas() -> list[dict[str, Any]]:
    """The nine tools in OpenAI function-calling format, generated from ``contract.py``."""
    out = []
    for name, model in c.REQUEST_MODELS.items():
        params = _inline_refs(model.model_json_schema())
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        out.append({"type": "function", "function": {"name": name.value, "description": c.TOOL_DESCRIPTIONS[name], "parameters": params}})
    return out


# ---- OpenAI-compatible HTTP client ----------------------------------------------------------


class DiskCache:
    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(self._path(key))


class OpenAICompatibleModel:
    """Chat completions with tools against any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        key_file: Path | str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 6,
        cache_dir: Path | str | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = 8192,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.name = model
        self.base_url = base_url.rstrip("/")
        key = api_key or load_api_key(key_file)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers; harmless elsewhere.
            "HTTP-Referer": "https://github.com/hnusrat1/opengray",
            "X-Title": "OpenGray",
        }
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self._sleep = sleep
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout_s, transport=transport)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, seed: int, temperature: float = 0.0) -> ModelReply:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
        }
        if tools:
            # A call without tools (the interpreter agent) omits the keys: some providers reject
            # an empty tool list, and the cache key stays the same for the tool-using agent.
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.max_tokens is not None:
            # Tool-call replies are short; the cap bounds reasoning-token spend and stops
            # OpenRouter from reserving the model's full context against the credit balance.
            body["max_tokens"] = self.max_tokens
        body.update(self.extra_body)
        key = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                reply = self._parse(hit)
                reply.cached = True
                return reply
        t0 = time.perf_counter()
        data = self._post_with_retries(body)
        reply = self._parse(data)
        reply.wall_s = round(time.perf_counter() - t0, 3)
        if self.cache is not None:
            self.cache.put(key, data)
        return reply

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                last = f"transport error: {e}"
                self._sleep(self._backoff(attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                self._sleep(self._backoff(attempt, r.headers.get("Retry-After")))
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code} from {self.base_url}: {r.text[:500]}")
            data = r.json()
            if "error" in data and not data.get("choices"):
                # OpenRouter returns 200 with an error body for some upstream failures.
                err = data["error"]
                code = err.get("code", 0) if isinstance(err, dict) else 0
                last = f"provider error: {json.dumps(err)[:300]}"
                if code in (429, 502, 503, 504) or "rate" in last.lower():
                    self._sleep(self._backoff(attempt))
                    continue
                raise LLMError(last)
            return data
        raise LLMError(f"gave up after {self.max_retries + 1} attempts; last: {last}")

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(120.0, float(retry_after))
            except ValueError:
                pass
        return min(60.0, (2.0**attempt) + random.uniform(0.0, 1.0))

    def _parse(self, data: dict[str, Any]) -> ModelReply:
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"malformed completion: {json.dumps(data)[:300]}") from e
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(id=tc.get("id") or f"call_{i}", name=fn.get("name", ""), arguments=args))
        usage = data.get("usage") or {}
        return ModelReply(
            content=msg.get("content"),
            tool_calls=calls,
            usage={"tokens_in": int(usage.get("prompt_tokens", 0) or 0), "tokens_out": int(usage.get("completion_tokens", 0) or 0)},
            model=str(data.get("model") or self.name),
        )


# ---- scripted model for tests and dry runs -------------------------------------------------

Policy = Callable[[list[dict[str, Any]]], "ModelReply | list[tuple[str, dict[str, Any]]] | str"]


class ScriptedModel:
    """A stand-in model driven by a Python policy over the message history.

    The policy returns a ModelReply, a list of ``(tool, args)`` pairs, or plain text. It sees
    exactly what a real model would see (system prompt, case summary, tool results), so the
    agent loop, the JSON round trips, and the budget handling are exercised without a network.
    """

    def __init__(self, policy: Policy, name: str = "scripted"):
        self.policy = policy
        self.name = name
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, seed: int, temperature: float = 0.0) -> ModelReply:
        self.calls += 1
        out = self.policy(messages)
        if isinstance(out, ModelReply):
            return out
        if isinstance(out, str):
            return ModelReply(content=out, model=self.name)
        calls = [ToolCall(id=f"call_{self.calls}_{i}", name=t, arguments=json.dumps(a)) for i, (t, a) in enumerate(out)]
        return ModelReply(content=None, tool_calls=calls, model=self.name, usage={"tokens_in": 10, "tokens_out": 5})


def last_tool_result(messages: list[dict[str, Any]], tool: str | None = None) -> dict[str, Any] | None:
    """Helper for policies: the most recent tool result (optionally of a given tool)."""
    names = {}
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
    for m in reversed(messages):
        if m.get("role") == "tool" and (tool is None or names.get(m.get("tool_call_id")) == tool):
            try:
                return json.loads(m["content"])
            except json.JSONDecodeError:
                return None
    return None


# ---- prompt ----------------------------------------------------------------------------------


PROMPT_VERSION = 3


def system_prompt(track: str, k: int, tool_cap: int, rules: bool = True) -> str:
    return (
        "You are a radiotherapy treatment planner working through a set of tools. You are given one "
        "patient case with structures, prescriptions, and a goal list, and you must produce a fluence-"
        "optimized IMRT plan that meets the goals, then submit it. This is a benchmark environment, "
        "not a clinical system.\n\n"
        "How planning works here. The optimizer minimizes a weighted sum of penalty terms that you "
        "define with set_objectives. Each term names a structure, a term type, a dose level in Gy, and a "
        "priority (0 to 1000; larger means the optimizer works harder on that term). Term types: "
        "min_dose (penalize dose below the level; use on targets at the prescription), max_dose "
        "(penalize dose above the level; use on serial organs and to control target hot spots), "
        "mean_dose (penalize mean dose above the level; parallel organs such as parotids), uniform_dose "
        "(penalize deviation from the level), dvh_max (penalize the volume above the level exceeding "
        "volume_fraction). Terms are not goals: a goal such as SpinalCord D0.1cc <= 45 Gy is usually met "
        "by a max_dose term at or a little below 45 Gy with a high priority, and a target D99 >= 66.5 Gy "
        "by a min_dose term at the prescription (70 Gy) plus a max_dose term slightly above it. Goals "
        "carry a unit field; the optimizer takes levels in Gy only, so convert if a goal is in cGy.\n\n"
        f"Workflow and budget. Track {track}. You may call optimize at most {k} time(s); each call runs "
        "the optimizer on one objective_id and returns a plan_id with every goal's achieved value. "
        + (
            "On this track the goal list changes once during the episode, after your first optimize: "
            "that optimize response carries a notice field describing the change, its goal table is "
            "already evaluated against the new list, and the plan is judged on the updated goals. "
            if track == "T3"
            else ""
        )
        +
        "set_objectives, get_metrics, get_dvh, compare_to_goals, and normalize do not consume the optimize "
        f"budget, but every tool call counts toward a cap of {tool_cap} calls for the episode, including "
        "the final submit. normalize rescales a plan linearly so one dose metric equals a value; it is a "
        "cheap way to fix target coverage at the cost of scaling every other dose by the same factor.\n\n"
        + (
            "Acceptability rules. The case summary lists the task's rules and they apply to every plan: "
            "a plan whose maximum dose anywhere exceeds 115 percent of the highest prescription, or "
            "whose target D99 falls below 80 percent of that target's prescription, is rejected outright "
            "and counts as no plan at all; among plans that are not rejected, meeting more hard goals is "
            "better, and missing a hard goal by a little still counts as missing it. A hot spot above the "
            "rejection line is therefore worse than a modest coverage miss.\n\n"
            if rules
            else ""
        )
        + "Goals marked hard are requirements; goals marked soft are wishes; lower tier numbers matter "
        "more. Read the goal table after each optimize, adjust priorities or levels for the goals that "
        "are not met (raise the priority of violated serial-organ and target terms, lower the level a "
        "little below the limit if raising priority is not enough), and optimize again while budget "
        "remains. Every episode must end with exactly one submit (the plan_id you choose) or one "
        "escalate.\n\n"
        "Submit or escalate. When the budget runs out before every goal is met, submit the best plan "
        "you have and list the unmet goals in the note; on hard cases an imperfect plan with an honest "
        "note is the expected outcome, as it would be for a human planner. One optimize that misses a "
        "goal is not evidence that the case is infeasible. Escalate instead of submitting only when the "
        "case cannot or should not be planned as given: a structure the goals need is missing "
        "(missing_structure), the goals contradict each other (contradictory_instructions), their units "
        "are ambiguous (unit_ambiguity), or a limit cannot be met by any plan (infeasible). Use the "
        "tools; do not describe what you would do. Keep any text you write short."
    )


# ---- the agent -------------------------------------------------------------------------------


class LLMAgent:
    def __init__(self, spec: AgentSpec, model: ChatModel, *, temperature: float = 0.0, max_model_calls: int | None = None, auto_submit: bool = True, nudge_limit: int = 2):
        self.spec = spec
        self.model = model
        self.temperature = temperature
        self.max_model_calls = max_model_calls
        self.auto_submit = auto_submit
        self.nudge_limit = nudge_limit
        self.tools = tool_schemas()

    # -- helpers ----------------------------------------------------------------------------------

    @staticmethod
    def _quality(result: dict[str, Any]) -> tuple[int, float] | None:
        goals = result.get("goals")
        if not isinstance(goals, list):
            return None
        try:
            status = [c.GoalStatus.model_validate(g) for g in goals]
        except Exception:  # noqa: BLE001 - malformed result: no quality signal
            return None
        return plan_quality(status)

    def _finish(self, client: Any, best: tuple[tuple[int, float], str] | None, why: str, tool_calls_left: int) -> None:
        if best is not None and self.auto_submit and tool_calls_left >= 1:
            client.call_json("submit", {"plan_id": best[1], "note": f"auto-submitted: {why}"})
        elif tool_calls_left >= 1:
            client.call_json("escalate", {"reason": "other", "explanation": f"agent harness: {why}; no plan was produced"})
        client.note("llm_finish", why=why, best_plan=best[1] if best else None, tool_calls_left=tool_calls_left)

    # -- main loop --------------------------------------------------------------------------------

    def run(self, client: Any, seed: int) -> None:
        summary = client.call_json("get_case_summary", {})
        if "error" in summary:
            client.call_json("escalate", {"reason": "other", "explanation": f"get_case_summary failed: {summary}"})
            return
        budget = summary.get("budget", {})
        k = int(budget.get("optimize_total", 1))
        cap = int(budget.get("tool_calls_max") or (3 * k + 6))
        calls_made = 1
        max_model_calls = self.max_model_calls or (2 * cap)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(str(summary.get("track", "")), k, cap, rules=bool(summary.get("rules")))},
            {"role": "user", "content": "Case summary (result of get_case_summary):\n" + json.dumps(summary, separators=(",", ":")) + "\n\nPlan this case. Start with set_objectives."},
        ]
        client.note("llm_start", model=self.model.name, temperature=self.temperature, seed=seed, tool_cap=cap, max_model_calls=max_model_calls, prompt_version=PROMPT_VERSION, rules_disclosed=bool(summary.get("rules")), system_prompt=messages[0]["content"])
        best: tuple[tuple[int, float], str] | None = None
        idle = 0
        model_calls = 0
        while True:
            if model_calls >= max_model_calls:
                self._finish(client, best, f"model-call limit {max_model_calls} reached", cap - calls_made)
                return
            try:
                reply = self.model.complete(messages, self.tools, seed=seed, temperature=self.temperature)
            except LLMError as e:





                client.note("llm_error", detail=str(e)[:1000])
                raise
            model_calls += 1
            client.record_usage(reply.usage.get("tokens_in", 0), reply.usage.get("tokens_out", 0), 1)
            client.note(
                "llm_call",
                n=model_calls,
                model=reply.model or self.model.name,
                cached=reply.cached,
                wall_s=reply.wall_s,
                usage=reply.usage,
                content=reply.content,
                tool_calls=[{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in reply.tool_calls],
            )
            messages.append(reply.as_message())
            if not reply.tool_calls:
                idle += 1
                if idle >= self.nudge_limit:
                    self._finish(client, best, "model stopped calling tools", cap - calls_made)
                    return
                messages.append({"role": "user", "content": "Continue by calling a tool. End the episode with submit(plan_id) or escalate(reason)."})
                continue
            idle = 0
            for tc in reply.tool_calls:
                try:
                    args = json.loads(tc.arguments) if tc.arguments.strip() else {}
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as e:
                    result: dict[str, Any] = {"error": "invalid_json", "detail": f"tool arguments are not valid JSON: {e}"}
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
                    continue
                left = cap - calls_made
                if self.auto_submit and left <= 1 and tc.name not in TERMINAL_TOOLS:
                    # The last allowed call must end the episode: submit the best plan, or escalate.
                    self._finish(client, best, "tool-call cap reached", left)
                    return
                result = client.call_json(tc.name, args)
                calls_made += 1
                if tc.name in PLAN_TOOLS and "error" not in result:
                    q = self._quality(result)
                    if q is not None and (best is None or q > best[0]):
                        best = (q, str(result.get("plan_id")))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, separators=(",", ":"))})
                if tc.name in TERMINAL_TOOLS and "error" not in result:
                    return
                if result.get("error") == "episode_over":
                    return
                if result.get("error") == "tool_call_cap":
                    client.note("llm_finish", why="tool-call cap hit inside the session", best_plan=best[1] if best else None, tool_calls_left=0)
                    return


def make_llm_agent(
    model: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    key_file: Path | str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_model_calls: int | None = None,
    cache_dir: Path | str | None = None,
    timeout_s: float = 180.0,
    extra_body: dict[str, Any] | None = None,
    max_tokens: int | None = 8192,
    chat_model: ChatModel | None = None,
    **_: Any,
) -> LLMAgent:
    params: dict[str, Any] = {"model": model}
    if temperature:
        params["temperature"] = temperature
    spec = AgentSpec("llm", params)
    m = chat_model or OpenAICompatibleModel(model, base_url=base_url, api_key=api_key, key_file=key_file, cache_dir=cache_dir, timeout_s=timeout_s, extra_body=extra_body, max_tokens=max_tokens)
    return LLMAgent(spec, m, temperature=temperature, max_model_calls=max_model_calls)
