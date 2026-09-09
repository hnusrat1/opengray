import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from opengray.agents.llm import LLMError, OpenAICompatibleModel
from opengray.agents.spending import SpendingLedger, SpendingLimitError, request_cost_bound


def test_concurrent_reservations_and_restart_preserve_ceiling(tmp_path):
    ledger = SpendingLedger(tmp_path / "cost.sqlite", 1.0)
    def reserve(i):
        try:
            return ledger.reserve(str(i), "model", 0.2)
        except SpendingLimitError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(reserve, range(20)))
    assert sum(x is not None for x in ids) == 5
    ledger = SpendingLedger(tmp_path / "cost.sqlite", 1.0)
    assert ledger.snapshot()["committed_usd"] == 1.0
    one = next(x for x in ids if x)
    ledger.finish(one, 0.05, state="response", metadata={"generation_id": "one"})
    ledger.reserve("next", "model", 0.15)
    assert ledger.snapshot()["committed_usd"] == 1.0
    with pytest.raises(ValueError, match="different ceiling"):
        SpendingLedger(tmp_path / "cost.sqlite", 2.0)


def test_uncertain_failure_and_overrun_stop_new_requests(tmp_path):
    ledger = SpendingLedger(tmp_path / "cost.sqlite", 1.0)
    a = ledger.reserve("a", "m", 0.4)
    ledger.finish(a, None, state="transport_error", metadata={})
    b = ledger.reserve("b", "m", 0.1)
    with pytest.raises(SpendingLimitError, match="exceeded"):
        ledger.finish(b, 0.2, state="response", metadata={})
    assert ledger.snapshot()["committed_usd"] == 0.6
    with pytest.raises(SpendingLimitError):
        ledger.reserve("c", "m", 0.1)


def test_budget_checks_precede_network_and_each_retry_is_retained(tmp_path):
    requests = []
    def handler(req):
        requests.append(json.loads(req.content))
        if len(requests) == 1:
            return httpx.Response(503, text="down")
        return httpx.Response(200, json={"id":"gen-1", "provider":"example", "model":"m", "choices":[{"message":{"content":"ok"}}],
            "usage":{"cost":0.001,"prompt_tokens":3,"completion_tokens":1,"completion_tokens_details":{"reasoning_tokens":1}}})
    ledger = SpendingLedger(tmp_path / "cost.sqlite", 1.0)
    m = OpenAICompatibleModel("m", api_key="test", transport=httpx.MockTransport(handler), max_tokens=100,
        spending=ledger, price_bounds=(2e-6,12e-6), supported_parameters=["seed"], sleep=lambda s:None)
    reply = m.complete([{"role":"user","content":"hello"}], [], seed=1)
    assert "temperature" not in requests[0] and requests[0]["seed"] == 1
    assert reply.billing["generation_id"] == "gen-1"
    assert reply.billing["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert ledger.snapshot()["requests"] == 2 and ledger.snapshot()["unreconciled_requests"] == 1
    assert ledger.snapshot()["reported_cost_usd"] == 0.001
    small = SpendingLedger(tmp_path / "small.sqlite", 0.00001)
    m.spending = small
    with pytest.raises(LLMError, match="spending"):
        m.complete([{"role":"user","content":"hello"}], [], seed=2)
    assert len(requests) == 2


def test_unsupported_billing_features_and_missing_output_cap_are_rejected():
    for body in ({"messages":[],"max_tokens":None}, {"messages":[],"max_tokens":10,"plugins":[]},
                 {"messages":[{"role":"user","content":[{"type":"image_url"}]}],"max_tokens":10}):
        with pytest.raises(ValueError):
            request_cost_bound(body, 2e-6, 12e-6)
