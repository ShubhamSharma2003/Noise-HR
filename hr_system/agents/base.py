import os
import json
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL = "gpt-4o"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def call_llm(system_prompt: str, user_prompt: str, expect_json: bool = True) -> dict | str:
    """
    Call GPT-4o.
    When expect_json=True, uses JSON mode and parses before returning.
    """
    kwargs = {}
    if expect_json:
        kwargs["response_format"] = {"type": "json_object"}

    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        **kwargs,
    )

    text = response.choices[0].message.content.strip()

    if expect_json:
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
