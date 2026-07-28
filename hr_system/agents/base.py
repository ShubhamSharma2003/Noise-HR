from __future__ import annotations

import os
import json
import random
import time
from datetime import datetime, timezone
from openai import (
    OpenAI,
    BadRequestError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
from dotenv import load_dotenv

load_dotenv()

_client = None
# Deep screening (the resume-reading judgment) — the accuracy-critical step, so
# it defaults to a strong model. Override with HR_DEEP_MODEL.
MODEL = os.environ.get("HR_DEEP_MODEL", "gpt-5.6-terra")
# Manager/CoS review the screener's *writeup* only (never the resume). They run
# only in the one-off Single Applicant path (bulk scans skip the manager), on a
# capable, fast, non-reasoning model — strong enough not to spuriously reject
# good analyses, without reasoning-model latency.
MANAGER_MODEL = os.environ.get("HR_MANAGER_MODEL", "gpt-4o")
# Bulk triage pass — cheap, fast, high-volume.
TRIAGE_MODEL = os.environ.get("HR_TRIAGE_MODEL", "gpt-5.4-nano")

# Transient-error retry (rate limits, timeouts, 5xx). Bulk scans fire hundreds
# of concurrent calls, so 429s are expected under load and must be ridden out
# rather than turned into missing scores.
_MAX_TRANSIENT_RETRIES = int(os.environ.get("HR_LLM_MAX_RETRIES", "6"))
# Cap on any single backoff sleep, so a server-supplied Retry-After of "300"
# can't freeze a worker (and the whole scan) for minutes.
_MAX_RETRY_SLEEP = float(os.environ.get("HR_LLM_MAX_RETRY_SLEEP", "60"))


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


# Parameter quirks discovered per model via 400s, remembered for the process
# lifetime so bulk scans don't replay the same failing request per applicant.
_MODEL_PARAM_FIXES = {}

_FIX_NAMES = ("swap_max_tokens", "drop_temperature", "downgrade_schema", "drop_response_format")


def _apply_param_fixes(kwargs: dict, fixes: set) -> None:
    if "swap_max_tokens" in fixes and "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    if "drop_temperature" in fixes:
        kwargs.pop("temperature", None)
    if "downgrade_schema" in fixes \
            and (kwargs.get("response_format") or {}).get("type") == "json_schema":
        kwargs["response_format"] = {"type": "json_object"}
    if "drop_response_format" in fixes:
        kwargs.pop("response_format", None)


def _retry_after_seconds(err) -> float | None:
    """Honor a Retry-After header on a rate-limit response, if present."""
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None) or {}
    val = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    model: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    json_schema: dict = None,
    prompt_cache_key: str = None,
) -> dict | str:
    """
    Call the LLM (default: MODEL) and return parsed JSON or raw text.

    - temperature defaults to 0 so repeated screenings score consistently;
      pass temperature=None to use the model's default.
    - json_schema switches to strict Structured Outputs (guaranteed schema);
      otherwise expect_json=True uses plain JSON mode.
    - prompt_cache_key improves OpenAI prompt-cache hit rates when many calls
      share the same system-prompt + JD prefix (e.g. one key per job posting).
    """
    kwargs = {"max_tokens": max_tokens}
    if json_schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": json_schema},
        }
    elif expect_json:
        kwargs["response_format"] = {"type": "json_object"}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if prompt_cache_key:
        kwargs["extra_body"] = {"prompt_cache_key": prompt_cache_key}

    def _create(kw):
        return _get_client().chat.completions.create(
            model=model or MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            **kw,
        )

    # Not every model accepts every knob: reasoning models want
    # max_completion_tokens instead of max_tokens and reject non-default
    # temperature; some models lack strict Structured Outputs. Each 400 names
    # one offending parameter, so fix complaints one at a time and retry —
    # remembering the fixes per model so later calls start with working params.
    model_name = model or MODEL
    fixes = _MODEL_PARAM_FIXES.setdefault(model_name, set())
    _apply_param_fixes(kwargs, fixes)

    def _create_with_param_fixes():
        for _ in range(len(_FIX_NAMES) + 1):
            try:
                return _create(kwargs)
            except BadRequestError as e:
                msg = str(e).lower()
                if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                    fixes.add("swap_max_tokens")
                elif "temperature" in msg and "temperature" in kwargs:
                    fixes.add("drop_temperature")
                elif "invalid schema" in msg:
                    raise  # a broken schema definition is a bug, not unsupported
                elif ("response_format" in msg or "schema" in msg) \
                        and (kwargs.get("response_format") or {}).get("type") == "json_schema":
                    fixes.add("downgrade_schema")
                elif "response_format" in msg and "response_format" in kwargs:
                    fixes.add("drop_response_format")
                else:
                    raise
                _apply_param_fixes(kwargs, fixes)
        raise RuntimeError("LLM call failed after exhausting parameter fallbacks")

    # Ride out rate limits and transient server errors with exponential backoff
    # + jitter, so high-concurrency bulk scans don't drop resumes to errors.
    def _transient_create():
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            try:
                return _create_with_param_fixes()
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
                if attempt >= _MAX_TRANSIENT_RETRIES:
                    raise
                backoff = min(2 ** attempt, 30) + random.uniform(0, 1)
                wait = min(max(backoff, _retry_after_seconds(e) or 0.0), _MAX_RETRY_SLEEP)
                time.sleep(wait)
        raise RuntimeError("LLM call failed after exhausting rate-limit retries")

    wants_json = expect_json or json_schema is not None

    # A reasoning model can spend its whole token budget on reasoning and
    # truncate the JSON (finish_reason == "length"). Bump the budget once and
    # retry before giving up.
    choice = None
    for _ in range(2):
        response = _transient_create()
        choice = response.choices[0]
        if wants_json and getattr(choice, "finish_reason", None) == "length":
            key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
            current = kwargs.get(key, max_tokens)
            bumped = min(current * 2, 32000)
            if bumped > current:
                kwargs[key] = bumped
                continue
        break

    text = (choice.message.content or "").strip()

    if wants_json:
        if getattr(choice, "finish_reason", None) == "length":
            raise ValueError(
                "LLM output was truncated by the token limit before the JSON "
                "completed even after raising the budget — increase max_tokens."
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output: {text[:500]}") from e

    return text


def append_history(state: dict, node: str, data: dict) -> list:
    """Return a new history list with an event appended (immutable update pattern)."""
    entry = {
        "node": node,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    return [*(state.get("history") or []), entry]
