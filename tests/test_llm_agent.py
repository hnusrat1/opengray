from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from opengray.agents.base import AgentSpec, make_agent
from opengray.agents.llm import (
    LLMAgent,
    LLMError,
    ModelReply,
    OpenAICompatibleModel,
    ScriptedModel,
    ToolCall,
    last_tool_result,
    load_api_key,
    make_llm_agent,
    system_prompt,
    tool_schemas,
)
from opengray.env.tools import InProcessClient
from tests.test_session import new_session


def template_from_summary(summary: dict[str, Any], priority: float = 10.0) -> list[dict[str, Any]]:
    """What a competent model would do with the goal table: one term per goal."""
    terms = []
    for g in summary["goals"]:
        value = g["value"] / 100 if g["unit"] == "cGy" else g["value"]
        if g["op"] == ">=":
            rx = summary["prescriptions"][g["structure"]]
            terms.append({"structure": g["structure"], "term": "min_dose", "level": rx, "priority": priority})
            terms.append({"structure": g["structure"], "term": "max_dose", "level": rx * 1.07, "priority": priority})
        elif g["metric"] == "Dmean":
            terms.append({"structure": g["structure"], "term": "mean_dose", "level": value, "priority": priority})
        else:
            terms.append({"structure": g["structure"], "term": "max_dose", "level": value, "priority": priority})
    return terms


def planner_policy(messages: list[dict[str, Any]]):
    """Plan like a careful model: objectives, optimize, escalate priorities once, submit the best."""
    summary = json.loads(messages[1]["content"].split("get_case_summary):\n", 1)[1].split("\n\nPlan this case", 1)[0])
    n_opt = sum(1 for m in messages if m.get("role") == "assistant" for tc in m.get("tool_calls") or [] if tc["function"]["name"] == "optimize")
    last = last_tool_result(messages)
    if last is None:
        return [("set_objectives", {"objectives": template_from_summary(summary)})]
    if "objective_id" in last and "plan_id" not in last:
        return [("optimize", {"objective_id": last["objective_id"]})]
    if "plan_id" in last and "goals" in last:
        if last["hard_met"] == last["hard_total"] or n_opt >= 2:
            return [("submit", {"plan_id": last["plan_id"], "note": "done"})]
        return [("set_objectives", {"objectives": template_from_summary(summary, priority=100.0)})]
    return "I am not sure what to do."


def run_agent(policy, track="T2", k=None, **agent_kw) -> tuple[Any, ScriptedModel, LLMAgent]:
    s = new_session(track, k)
    model = ScriptedModel(policy)
    agent = LLMAgent(AgentSpec("llm", {"model": "scripted"}), model, **agent_kw)
    agent.run(InProcessClient(s), seed=0)
    return s, model, agent


def test_tool_schemas_cover_the_contract() -> None:
    schemas = tool_schemas()
    names = [t["function"]["name"] for t in schemas]
    assert names == ["get_case_summary", "get_metrics", "get_dvh", "set_objectives", "optimize", "normalize", "compare_to_goals", "submit", "escalate"]
    for t in schemas:
        p = t["function"]["parameters"]
        assert p["type"] == "object" and "$defs" not in json.dumps(p) and "$ref" not in json.dumps(p)
    so = next(t for t in schemas if t["function"]["name"] == "set_objectives")["function"]["parameters"]
    assert so["properties"]["objectives"]["items"]["properties"]["term"]["enum"] == ["min_dose", "max_dose", "mean_dose", "uniform_dose", "dvh_max"]
    prompt = system_prompt("T2", 3, 15)
    for banned in ("PlanScore", "0.5", "reference plan", "bootstrap", "gate", "score", "worth"):
        assert banned not in prompt
    # v3 states the acceptability rules (the two rejections) and nothing about how goals are weighed.
    assert "115 percent" in prompt and "80 percent" in prompt and "rejected outright" in prompt
    bare = system_prompt("T2", 3, 15, rules=False)
    assert "rejected outright" not in bare and "115 percent" not in bare


def test_scripted_model_plans_and_submits() -> None:
    s, model, _ = run_agent(planner_policy)
    assert s.status == "submitted" and s.terminal["outcome"] == "submitted"
    assert s.terminal["note"] == "done"
    assert s.optimize_calls >= 1 and model.calls >= 3
    events = [e["event"] for e in s.events]
    assert events[0] == "tool_call" and "llm_start" in events and events.count("llm_call") == model.calls
    assert s.usage["model_calls"] == model.calls and s.usage["tokens_in"] == 10 * model.calls
    # The transcript keeps the tool calls the model made, in order.
    calls = [tc["name"] for e in s.events if e["event"] == "llm_call" for tc in e["tool_calls"]]
    assert calls[0] == "set_objectives" and calls[-1] == "submit"


def test_model_that_stops_calling_tools_gets_nudged_then_auto_submitted() -> None:
    def policy(messages):
        last = last_tool_result(messages)
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        if n_assistant == 0:
            summary = json.loads(messages[1]["content"].split("get_case_summary):\n", 1)[1].split("\n\nPlan this case", 1)[0])
            return [("set_objectives", {"objectives": template_from_summary(summary)}), ("optimize", {"objective_id": "obj_1"})]
        assert last is not None
        return "Looks good to me."

    s, model, _ = run_agent(policy)
    assert s.status == "submitted"
    assert s.terminal["note"].startswith("auto-submitted: model stopped calling tools")
    assert model.calls == 3  # plan, text, nudge -> text again
    nudges = [m for e in s.events if e["event"] == "llm_finish" for m in [e]]
    assert nudges and nudges[0]["best_plan"] == "plan_1"


def test_no_plan_and_no_tools_escalates() -> None:
    s, _, _ = run_agent(lambda messages: "I refuse.")
    assert s.status == "escalated" and s.terminal["reason"] == "other"


def test_bad_json_arguments_are_returned_as_tool_errors_without_spending_calls() -> None:
    seen: list[str] = []

    def policy(messages):
        last = messages[-1]
        if last.get("role") == "tool":
            seen.append(last["content"])
        if not seen:
            return ModelReply(content=None, tool_calls=[ToolCall(id="x1", name="set_objectives", arguments="{not json")])
        return [("escalate", {"reason": "other", "explanation": "stop"})]

    s, _, _ = run_agent(policy)
    assert json.loads(seen[0])["error"] == "invalid_json"
    assert s.tool_calls == 2  # summary + escalate; the bad call never reached the session
    assert s.status == "escalated"


def test_tool_call_cap_triggers_auto_submit_of_best_plan() -> None:
    def policy(messages):
        last = last_tool_result(messages)
        if last is None:
            summary = json.loads(messages[1]["content"].split("get_case_summary):\n", 1)[1].split("\n\nPlan this case", 1)[0])
            return [("set_objectives", {"objectives": template_from_summary(summary)}), ("optimize", {"objective_id": "obj_1"})]
        return [("compare_to_goals", {"plan_id": "plan_1"})]  # burn the cap without ever submitting

    s, _, _ = run_agent(policy, track="T1")  # T1: cap 3 x 1 + 6 = 9
    assert s.status == "submitted"
    assert s.terminal["note"].startswith("auto-submitted: tool-call cap reached")
    assert s.tool_calls == s.track.tool_call_cap


def test_unknown_tool_and_session_errors_flow_back_to_the_model() -> None:
    results: list[dict[str, Any]] = []

    def policy(messages):
        if messages[-1].get("role") == "tool":
            results.append(json.loads(messages[-1]["content"]))
        n = len(results)
        if n == 0:
            return [("teleport", {})]
        if n == 1:
            return [("optimize", {"objective_id": "obj_99"})]
        return [("escalate", {"reason": "infeasible", "explanation": "cannot"})]

    s, _, _ = run_agent(policy)
    assert results[0]["error"] == "unknown_tool" and results[1]["error"] == "session_error"
    assert s.status == "escalated" and s.terminal["reason"] == "infeasible"


def test_make_agent_wires_llm_with_injected_model() -> None:
    agent = make_llm_agent("scripted/x", chat_model=ScriptedModel(lambda m: "hi"))
    assert agent.spec.label == "llm:model=scripted/x"
    with pytest.raises(ValueError):
        make_agent("llm")
    with pytest.raises(LLMError):
        make_agent("llm", model="x/y", key_file=Path("/nonexistent/key"))


def test_load_api_key_prefers_env_then_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    f = tmp_path / "k"
    f.write_text("sk-file\n")
    assert load_api_key(f) == "sk-file"
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    assert load_api_key(f) == "sk-env"


def completion(content: str | None = None, calls: list[tuple[str, dict]] | None = None, model: str = "m") -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return {"id": "x", "model": model, "choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 7}}


def test_http_client_retries_429_then_parses_and_caches(tmp_path: Path) -> None:
    hits: list[dict[str, Any]] = []
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"message": "slow down"}}),
        httpx.Response(200, json=completion(calls=[("optimize", {"objective_id": "obj_1"})], model="provider/m:free")),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append({"auth": request.headers.get("Authorization"), "body": json.loads(request.content)})
        return responses.pop(0)

    slept: list[float] = []
    m = OpenAICompatibleModel("provider/m:free", base_url="https://example.test/v1", api_key="sk-test", cache_dir=tmp_path / "cache", transport=httpx.MockTransport(handler), sleep=slept.append)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    reply = m.complete(msgs, tool_schemas(), seed=3)
    assert len(hits) == 2 and slept == [0.0]
    assert hits[0]["auth"] == "Bearer sk-test"
    assert hits[1]["body"]["seed"] == 3 and hits[1]["body"]["temperature"] == 0.0 and hits[1]["body"]["tool_choice"] == "auto"
    assert reply.tool_calls[0].name == "optimize" and json.loads(reply.tool_calls[0].arguments) == {"objective_id": "obj_1"}
    assert reply.usage == {"tokens_in": 100, "tokens_out": 7} and reply.model == "provider/m:free" and not reply.cached
    # Second identical request is served from the cache without touching the transport.
    again = m.complete(msgs, tool_schemas(), seed=3)
    assert again.cached and len(hits) == 2 and again.tool_calls[0].name == "optimize"
    # A different seed is a different request.
    responses.append(httpx.Response(200, json=completion(content="ok")))
    other = m.complete(msgs, tool_schemas(), seed=4)
    assert not other.cached and other.content == "ok" and len(hits) == 3


def test_http_client_gives_up_and_raises_on_4xx() -> None:
    m = OpenAICompatibleModel("m", base_url="https://example.test/v1", api_key="k", max_retries=1, transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down")), sleep=lambda s: None)
    with pytest.raises(LLMError, match="gave up"):
        m.complete([{"role": "user", "content": "u"}], [], seed=0)
    m2 = OpenAICompatibleModel("m", base_url="https://example.test/v1", api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad request")))
    with pytest.raises(LLMError, match="HTTP 400"):
        m2.complete([{"role": "user", "content": "u"}], [], seed=0)
    m3 = OpenAICompatibleModel("m", base_url="https://example.test/v1", api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"error": {"message": "no such model", "code": 404}})))
    with pytest.raises(LLMError, match="provider error"):
        m3.complete([{"role": "user", "content": "u"}], [], seed=0)


def test_runner_records_llm_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the runner with an injected scripted model."""
    import json as _json

    from opengray.agents import base as agent_base
    from opengray.env.tracks import track_config
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.results import leaderboard, load_runs
    from opengray.runner.run import RunConfig, run
    from tests.test_solver import make_case

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    original = agent_base.make_agent

    def fake_make_agent(name, **params):
        if name == "llm":
            return make_llm_agent(params["model"], chat_model=ScriptedModel(planner_policy))
        return original(name, **params)

    monkeypatch.setattr("opengray.runner.run.make_agent", fake_make_agent)
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["synthetic"]}))
    cfg = RunConfig(track=track_config("T2", 3), agent=AgentSpec("llm", {"model": "scripted/planner"}), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split)
    res = run(cfg, Loader(), openkbp_default_goals())
    assert len(res.rows) == 2 and all(r["outcome"] == "submitted" for r in res.rows)
    assert res.rows[0]["model"] == "scripted/planner" and res.rows[0]["tokens_in"] > 0 and res.rows[0]["model_calls"] >= 3
    assert res.rows[0]["auto_submitted"] is False
    df = load_runs(tmp_path / "runs")
    lb = leaderboard(df)
    assert lb[0]["agent"] == "llm:model=scripted/planner" and lb[0]["model"] == "scripted/planner" and lb[0]["tokens_in_mean"] > 0
    log = (res.run_dir / "episodes.jsonl").read_text().splitlines()
    kinds = {_json.loads(line)["event"] for line in log}
    assert {"llm_start", "llm_call", "tool_call"} <= kinds


def test_model_failure_without_a_plan_is_an_error_not_an_escalation() -> None:
    class Dead:
        name = "dead"

        def complete(self, messages, tools, *, seed, temperature=0.0):
            raise LLMError("HTTP 429: rate limited")

    s = new_session()
    agent = LLMAgent(AgentSpec("llm", {"model": "dead"}), Dead())
    with pytest.raises(LLMError):
        agent.run(InProcessClient(s), seed=0)
    assert s.status == "active" and s.terminal is None
    assert any(e["event"] == "llm_error" for e in s.events)


def test_model_failure_after_a_plan_is_an_error_not_a_submission() -> None:
    class DiesAfterPlan:
        name = "flaky"
        calls = 0

        def complete(self, messages, tools, *, seed, temperature=0.0):
            self.calls += 1
            if self.calls == 1:
                summary = json.loads(messages[1]["content"].split("get_case_summary):\n", 1)[1].split("\n\nPlan this case", 1)[0])
                calls = [ToolCall("a", "set_objectives", json.dumps({"objectives": template_from_summary(summary)})), ToolCall("b", "optimize", json.dumps({"objective_id": "obj_1"}))]
                return ModelReply(content=None, tool_calls=calls, model="flaky")
            raise LLMError("HTTP 503")

    s = new_session()
    with pytest.raises(LLMError):
        LLMAgent(AgentSpec("llm", {"model": "flaky"}), DiesAfterPlan()).run(InProcessClient(s), seed=0)
    assert s.status == "active" and s.terminal is None and any(e["event"] == "llm_error" for e in s.events)


def test_reruns_replace_error_rows_in_load_runs(tmp_path: Path) -> None:
    import pandas as pd

    from opengray.runner.results import dedupe_episodes

    rows = [
        {"agent": "llm:m", "episode_id": "T1-k1-pt_1-s0", "plan_score": float("nan"), "error": "boom"},
        {"agent": "llm:m", "episode_id": "T1-k1-pt_1-s1", "plan_score": 0.5, "error": None},
        {"agent": "llm:m", "episode_id": "T1-k1-pt_1-s0", "plan_score": 0.7, "error": None},  # rerun of s0
        {"agent": "llm:m", "episode_id": "T1-k1-pt_1-s1", "plan_score": float("nan"), "error": "later failure"},
        {"agent": "heuristic", "episode_id": "T1-k1-pt_1-s0", "plan_score": 0.8, "error": None},
    ]
    out = dedupe_episodes(pd.DataFrame(rows))
    assert len(out) == 3
    got = {(r.agent, r.episode_id): r.plan_score for r in out.itertuples()}
    assert got[("llm:m", "T1-k1-pt_1-s0")] == 0.7 and got[("llm:m", "T1-k1-pt_1-s1")] == 0.5 and got[("heuristic", "T1-k1-pt_1-s0")] == 0.8


def test_resume_skips_completed_episodes_and_reruns_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from opengray.env.tracks import track_config
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.results import load_runs
    from opengray.runner.run import RunConfig, run
    from tests.test_solver import make_case

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    state = {"fail_seed": 1}

    class Model:
        name = "flaky"

        def complete(self, messages, tools, *, seed, temperature=0.0):
            if seed == state["fail_seed"]:
                raise LLMError("HTTP 402: no credits")
            return ScriptedModel(planner_policy).complete(messages, tools, seed=seed, temperature=temperature)

    monkeypatch.setattr("opengray.runner.run.make_agent", lambda name, **p: make_llm_agent(p["model"], chat_model=Model()))
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["synthetic"]}))
    cfg = RunConfig(track=track_config("T1"), agent=AgentSpec("llm", {"model": "flaky"}), split="validation", seeds=[0, 1, 2], out_dir=tmp_path / "runs", split_file=split)
    first = run(cfg, Loader(), openkbp_default_goals())
    assert [r["error"] is None for r in first.rows] == [True, False, True]
    # Credits restored: the same batch with resume=True reruns only the failed episode.
    state["fail_seed"] = None
    cfg2 = RunConfig(track=track_config("T1"), agent=AgentSpec("llm", {"model": "flaky"}), split="validation", seeds=[0, 1, 2], out_dir=tmp_path / "runs", split_file=split, resume=True)
    second = run(cfg2, Loader(), openkbp_default_goals())
    assert [r["episode_id"] for r in second.rows] == ["T1-k1-synthetic-s1"] and second.skipped == ["T1-k1-synthetic-s0", "T1-k1-synthetic-s2"]
    df = load_runs(tmp_path / "runs")
    assert len(df) == 3 and df["error"].isna().all()
    # Nothing left: a third resume writes no results and load_runs still works.
    third = run(cfg2, Loader(), openkbp_default_goals())
    assert third.rows == [] and len(third.skipped) == 3 and (third.run_dir / "skipped.json").exists()
    assert len(load_runs(tmp_path / "runs")) == 3
    # Strict resume: rows under another protocol id are superseded, not resumed. Rewrite the
    # stored rows' protocol id to an old one and rerun; every episode runs again and the
    # newest rows win in load_runs.
    for run_json in (tmp_path / "runs").glob("*/run.json"):
        meta = _json.loads(run_json.read_text())
        res = run_json.parent / "results.parquet"
        if res.exists():
            import pandas as pd

            df_old = pd.read_parquet(res)
            df_old["protocol_id"] = "000000000000"
            df_old.to_parquet(res, index=False)
        run_json.write_text(_json.dumps(meta))
    import time as _time

    _time.sleep(1.1)  # run ids carry a one-second timestamp; the newest run must sort last
    cfg3 = RunConfig(track=track_config("T1"), agent=AgentSpec("llm", {"model": "flaky"}), split="validation", seeds=[0, 1, 2], out_dir=tmp_path / "runs", split_file=split, resume=True, strict_resume=True)
    fourth = run(cfg3, Loader(), openkbp_default_goals())
    assert len(fourth.rows) == 3 and fourth.skipped == []
    df = load_runs(tmp_path / "runs")
    assert len(df) == 3 and set(df["protocol_id"]) == {cfg3.protocol["protocol_id"]}
    fifth = run(cfg3, Loader(), openkbp_default_goals())
    assert fifth.rows == [] and len(fifth.skipped) == 3


    for res in (tmp_path / "runs").glob("*/results.parquet"):
        import pandas as pd

        df_old = pd.read_parquet(res)
        df_old["protocol_id"] = None
        df_old.to_parquet(res, index=False)
    _time.sleep(1.1)
    sixth = run(cfg2, Loader(), openkbp_default_goals())
    assert sixth.rows == [] and len(sixth.skipped) == 3
    seventh = run(cfg3, Loader(), openkbp_default_goals())
    assert len(seventh.rows) == 3 and seventh.skipped == []


def test_rescore_rewrites_results_from_saved_w(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    import pandas as pd

    from opengray.env.tracks import track_config
    from opengray.goals import scoring
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.rescore import rescore_all
    from opengray.runner.run import RunConfig, run
    from tests.test_solver import make_case

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["synthetic"]}))
    cfg = RunConfig(track=track_config("T2", 3), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split)
    res = run(cfg, Loader(), openkbp_default_goals())
    meta = _json.loads((res.run_dir / "run.json").read_text())
    assert meta["scoring_version"] == scoring.SCORING_VERSION
    # A run stamped with an older version gets rescored; scores are recomputed from final_w.
    meta["scoring_version"] = 2
    (res.run_dir / "run.json").write_text(_json.dumps(meta))
    before = pd.read_parquet(res.run_dir / "results.parquet")
    rep = rescore_all(tmp_path / "runs", Loader(), openkbp_default_goals())
    assert rep[0]["status"] == "rescored" and rep[0]["from"] == 2 and rep[0]["missing_w"] == 0
    after = pd.read_parquet(res.run_dir / "results.parquet")
    assert (res.run_dir / "results.scoring_v2.parquet").exists()
    assert after["plan_score"].iloc[0] == pytest.approx(before["plan_score"].iloc[0])  # same rules, same score
    assert _json.loads((res.run_dir / "run.json").read_text())["scoring_version"] == scoring.SCORING_VERSION
    # Second pass: nothing to do.
    assert rescore_all(tmp_path / "runs", Loader(), openkbp_default_goals())[0]["status"] == "current"


def test_rescore_is_resumable_per_case(tmp_path: Path) -> None:
    import json as _json

    import pandas as pd

    from opengray.env.tracks import track_config
    from opengray.goals import scoring
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.rescore import rescore_all
    from opengray.runner.run import RunConfig, run
    from tests.test_solver import make_case

    class Loader:
        def list_cases(self):
            return ["a", "b"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11 if cid == "a" else 12)

    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["a", "b"]}))
    cfg = RunConfig(track=track_config("T1"), agent=AgentSpec("heuristic"), split="validation", seeds=[0], out_dir=tmp_path / "runs", split_file=split)
    res = run(cfg, Loader(), openkbp_default_goals())
    meta = _json.loads((res.run_dir / "run.json").read_text())
    meta["scoring_version"] = 2
    (res.run_dir / "run.json").write_text(_json.dumps(meta))
    rep = rescore_all(tmp_path / "runs", Loader(), openkbp_default_goals(), cases=["a"])
    assert rep[0]["status"] == "partial" and rep[0]["remaining_cases"] == ["b"]
    df = pd.read_parquet(res.run_dir / "results.parquet")
    assert df.set_index("case_id")["scoring_version"].to_dict() == {"a": scoring.SCORING_VERSION, "b": 2}
    assert _json.loads((res.run_dir / "run.json").read_text())["scoring_version"] == 2
    rep = rescore_all(tmp_path / "runs", Loader(), openkbp_default_goals(), cases=["b"])
    assert rep[0]["status"] == "rescored" and rep[0]["remaining_cases"] == []
    assert _json.loads((res.run_dir / "run.json").read_text())["scoring_version"] == scoring.SCORING_VERSION
    assert (res.run_dir / "results.scoring_v2.parquet").exists() and not (res.run_dir / "results.parquet.tmp").exists()


def test_mark_provider_failures_turns_old_autosubmits_into_error_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    "Legacy provider-error submissions become error records eligible for resume."
    import json as _json

    import pandas as pd

    from opengray.env.tracks import track_config
    from opengray.goals.defaults import openkbp_default_goals
    from opengray.runner.regrade import mark_provider_failures
    from opengray.runner.results import load_runs
    from opengray.runner.run import RunConfig, run
    from tests.test_solver import make_case

    class Loader:
        def list_cases(self):
            return ["synthetic"]

        def load(self, cid):
            return make_case(n_vox=90, n_beamlets=24, seed=11)

    monkeypatch.setattr("opengray.runner.run.make_agent", lambda name, **p: make_llm_agent(p["model"], chat_model=ScriptedModel(planner_policy)))
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"validation": ["synthetic"]}))
    cfg = RunConfig(track=track_config("T1"), agent=AgentSpec("llm", {"model": "scripted"}), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split)
    res = run(cfg, Loader(), openkbp_default_goals())
    assert all(r["outcome"] == "submitted" for r in res.rows)
    # Forge the old behaviour on seed 1: a terminal note that says the model died mid-episode.
    log = res.run_dir / "episodes.jsonl"
    lines = log.read_text().splitlines()
    out = []
    for line in lines:
        r = _json.loads(line)
        if r.get("episode_id") == "T1-k1-synthetic-s1" and r.get("terminal"):
            r["terminal"]["note"] = "auto-submitted: model error: HTTP 403 from provider: Key limit exceeded"
        out.append(_json.dumps(r))
    log.write_text("\n".join(out) + "\n")
    rep = mark_provider_failures(tmp_path / "runs")
    assert rep and rep[0]["status"] == "marked" and rep[0]["n"] == 1
    df = pd.read_parquet(res.run_dir / "results.parquet").set_index("episode_id")
    assert df.at["T1-k1-synthetic-s1", "error"].startswith("provider error mid-episode") and pd.isna(df.at["T1-k1-synthetic-s1", "plan_score"])
    assert pd.isna(df.at["T1-k1-synthetic-s0", "error"]) and (res.run_dir / "results.before_provider_fix.parquet").exists()
    assert mark_provider_failures(tmp_path / "runs")[0]["status"] == "current"  # idempotent
    lr = load_runs(tmp_path / "runs")
    assert lr[lr["error"].isna()]["episode_id"].tolist() == ["T1-k1-synthetic-s0"]
    cfg2 = RunConfig(track=track_config("T1"), agent=AgentSpec("llm", {"model": "scripted"}), split="validation", seeds=[0, 1], out_dir=tmp_path / "runs", split_file=split, resume=True)
    again = run(cfg2, Loader(), openkbp_default_goals())
    assert [r["episode_id"] for r in again.rows] == ["T1-k1-synthetic-s1"]
