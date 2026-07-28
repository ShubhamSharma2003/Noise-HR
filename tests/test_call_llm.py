import json
import os
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError, RateLimitError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hr_system.agents.base as base


def _bad_request(message):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    return BadRequestError(message, response=resp, body=None)


def _rate_limit(message="rate limit exceeded", retry_after=None):
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    resp = httpx.Response(429, request=req, headers=headers)
    return RateLimitError(message, response=resp, body=None)


class FakeCompletions:
    """Records calls; raises scripted errors before finally succeeding."""

    def __init__(self, errors=(), content='{"ok": true}', finish_reason="stop"):
        self.errors = list(errors)
        self.content = content
        self.finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        msg = SimpleNamespace(content=self.content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=self.finish_reason)]
        )


@pytest.fixture
def fake(monkeypatch):
    # The per-model fix memo must not leak between tests
    monkeypatch.setattr(base, "_MODEL_PARAM_FIXES", {})

    def _install(errors=(), content='{"ok": true}', finish_reason="stop"):
        completions = FakeCompletions(errors, content, finish_reason)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(base, "_client", client)
        return completions
    return _install


class TestRequestConstruction:
    def test_strict_schema_temperature_and_cache_key(self, fake):
        completions = fake()
        out = base.call_llm(
            "sys", "user",
            model="gpt-5.4-nano",
            json_schema={"type": "object"},
            prompt_cache_key="triage-job-7",
            max_tokens=500,
        )
        assert out == {"ok": True}
        kw = completions.calls[0]
        assert kw["model"] == "gpt-5.4-nano"
        assert kw["max_tokens"] == 500
        assert kw["temperature"] == 0.0
        assert kw["response_format"]["type"] == "json_schema"
        assert kw["response_format"]["json_schema"]["strict"] is True
        assert kw["extra_body"] == {"prompt_cache_key": "triage-job-7"}

    def test_default_json_mode_unchanged_for_graph_agents(self, fake):
        completions = fake()
        base.call_llm("sys", "user", expect_json=True)
        kw = completions.calls[0]
        assert kw["model"] == base.MODEL
        assert kw["response_format"] == {"type": "json_object"}


class TestParameterFallbacks:
    def test_max_tokens_swapped_for_max_completion_tokens(self, fake):
        completions = fake(errors=[_bad_request(
            "Unsupported parameter: 'max_tokens' is not supported with this model. "
            "Use 'max_completion_tokens' instead."
        )])
        base.call_llm("sys", "user", max_tokens=500)
        assert "max_tokens" not in completions.calls[1]
        assert completions.calls[1]["max_completion_tokens"] == 500

    def test_fixed_temperature_model(self, fake):
        completions = fake(errors=[_bad_request(
            "Unsupported value: 'temperature' does not support 0.0 with this model."
        )])
        base.call_llm("sys", "user")
        assert "temperature" not in completions.calls[1]

    def test_multiple_complaints_fixed_iteratively(self, fake):
        completions = fake(errors=[
            _bad_request("Use 'max_completion_tokens' instead of 'max_tokens'."),
            _bad_request("Unsupported value: 'temperature' must be 1 for this model."),
        ])
        base.call_llm("sys", "user", json_schema={"type": "object"})
        final = completions.calls[-1]
        assert "max_completion_tokens" in final and "temperature" not in final
        assert final["response_format"]["type"] == "json_schema"

    def test_schema_degrades_to_json_object(self, fake):
        completions = fake(errors=[_bad_request(
            "Invalid parameter: response_format json_schema is not supported for this model."
        )])
        base.call_llm("sys", "user", json_schema={"type": "object"})
        assert completions.calls[1]["response_format"] == {"type": "json_object"}

    def test_unrelated_400_reraised(self, fake):
        fake(errors=[_bad_request("This model's maximum context length is exceeded.")])
        with pytest.raises(BadRequestError):
            base.call_llm("sys", "user")

    def test_all_four_rungs_then_clean_attempt_succeeds(self, fake):
        completions = fake(errors=[
            _bad_request("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."),
            _bad_request("Unsupported value: 'temperature' must be 1 for this model."),
            _bad_request("response_format json_schema is not supported for this model."),
            _bad_request("'messages' must contain the word 'json' to use response_format json_object."),
        ])
        out = base.call_llm("sys", "user", json_schema={"type": "object"})
        assert out == {"ok": True}
        assert len(completions.calls) == 5
        final = completions.calls[-1]
        assert "response_format" not in final and "temperature" not in final
        assert final["max_completion_tokens"] == 2048

    def test_fixes_memoized_per_model(self, fake, monkeypatch):
        completions = fake(errors=[
            _bad_request("Use 'max_completion_tokens' instead of 'max_tokens'."),
            _bad_request("Unsupported value: 'temperature' must be 1 for this model."),
        ])
        base.call_llm("sys", "user", model="gpt-5.4-nano")
        assert len(completions.calls) == 3
        # Second call to the same model: no re-learning, single request
        fresh = FakeCompletions()
        monkeypatch.setattr(base, "_client",
                            SimpleNamespace(chat=SimpleNamespace(completions=fresh)))
        base.call_llm("sys", "user", model="gpt-5.4-nano")
        assert len(fresh.calls) == 1
        assert "max_completion_tokens" in fresh.calls[0]
        assert "temperature" not in fresh.calls[0]

    def test_invalid_schema_error_reraised_not_downgraded(self, fake):
        fake(errors=[_bad_request(
            "Invalid schema for response_format 'response': 'required' is required to be supplied."
        )])
        with pytest.raises(BadRequestError):
            base.call_llm("sys", "user", json_schema={"type": "object"})

    def test_truncated_output_raises_clear_error(self, fake):
        fake(content='{"partial": ', finish_reason="length")
        with pytest.raises(ValueError, match="truncated"):
            base.call_llm("sys", "user", expect_json=True)

    def test_invalid_json_raises_value_error(self, fake):
        fake(content="not json at all")
        with pytest.raises(ValueError):
            base.call_llm("sys", "user", expect_json=True)


class TestRateLimitBackoff:
    def test_rate_limit_retried_then_succeeds(self, fake, monkeypatch):
        slept = []
        monkeypatch.setattr(base.time, "sleep", lambda s: slept.append(s))
        completions = fake(errors=[_rate_limit(), _rate_limit()])
        out = base.call_llm("sys", "user")
        assert out == {"ok": True}
        assert len(completions.calls) == 3      # 2 failures + 1 success
        assert len(slept) == 2                   # backed off before each retry

    def test_rate_limit_exhausted_reraises(self, fake, monkeypatch):
        monkeypatch.setattr(base.time, "sleep", lambda s: None)
        monkeypatch.setattr(base, "_MAX_TRANSIENT_RETRIES", 2)
        fake(errors=[_rate_limit(), _rate_limit(), _rate_limit()])
        with pytest.raises(RateLimitError):
            base.call_llm("sys", "user")

    def test_retry_after_header_respected(self, fake, monkeypatch):
        slept = []
        monkeypatch.setattr(base.time, "sleep", lambda s: slept.append(s))
        fake(errors=[_rate_limit(retry_after=17)])
        base.call_llm("sys", "user")
        assert slept and slept[0] >= 17          # honored the server's Retry-After

    def test_rate_limit_then_param_fix_compose(self, fake, monkeypatch):
        monkeypatch.setattr(base.time, "sleep", lambda s: None)
        completions = fake(errors=[
            _rate_limit(),
            _bad_request("Use 'max_completion_tokens' instead of 'max_tokens'."),
        ])
        out = base.call_llm("sys", "user", max_tokens=500)
        assert out == {"ok": True}
        assert completions.calls[-1]["max_completion_tokens"] == 500

    def test_retry_after_capped(self, fake, monkeypatch):
        slept = []
        monkeypatch.setattr(base.time, "sleep", lambda s: slept.append(s))
        monkeypatch.setattr(base, "_MAX_RETRY_SLEEP", 60)
        fake(errors=[_rate_limit(retry_after=9999)])
        base.call_llm("sys", "user")
        assert slept and slept[0] <= 60          # huge Retry-After is capped


class TestTruncationRetry:
    def test_truncated_json_retried_with_bigger_budget(self, fake, monkeypatch):
        # First response truncates; second (after budget bump) succeeds.
        completions = FakeCompletions(finish_reason="length")
        monkeypatch.setattr(base, "_MODEL_PARAM_FIXES", {})
        monkeypatch.setattr(base, "_client",
                            SimpleNamespace(chat=SimpleNamespace(completions=completions)))

        original_create = completions.create
        state = {"n": 0}

        def flaky_create(**kwargs):
            state["n"] += 1
            completions.calls.append(kwargs)
            fr = "length" if state["n"] == 1 else "stop"
            msg = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=fr)])

        completions.create = flaky_create
        out = base.call_llm("sys", "user", max_tokens=1000)
        assert out == {"ok": True}
        assert len(completions.calls) == 2
        assert completions.calls[1]["max_tokens"] == 2000   # budget doubled

    def test_still_truncated_raises(self, fake):
        fake(content='{"partial":', finish_reason="length")
        with pytest.raises(ValueError, match="truncated"):
            base.call_llm("sys", "user", expect_json=True)
