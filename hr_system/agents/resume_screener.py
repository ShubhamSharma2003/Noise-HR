from hr_system.state import HRState, AgentOutput
from hr_system.agents.base import call_llm, append_history
from hr_system.prompts.resume_screener import (
    SYSTEM_PROMPT,
    build_user_prompt,
    DEEP_JSON_SCHEMA,
    clamp_confidence,
)


def resume_screener_node(state: HRState) -> dict:
    """LangGraph node: A1 — Resume Screener (the resume-reading fit judgment)."""
    feedback = (state.get("manager_decision") or {}).get("feedback", "")

    user_prompt = build_user_prompt(
        resume_text=state["task_input"].get("resume_text", ""),
        job_description=state["task_input"].get("job_description", ""),
        job_title=state["task_input"].get("job_title", ""),
        feedback=feedback,
    )

    # Strict schema guarantees all fields + 6 dimension scores; generous token
    # budget covers the detailed prose plus any reasoning tokens; a per-job
    # cache key discounts the repeated system-prompt + JD prefix.
    raw: dict = call_llm(
        SYSTEM_PROMPT,
        user_prompt,
        json_schema=DEEP_JSON_SCHEMA,
        max_tokens=12000,
        prompt_cache_key=f"deep-job-{state['task_input'].get('job_id')}",
    )

    # .get(default) only fires when the key is ABSENT; an explicit null would
    # slip through and crash float(None), so coalesce None separately.
    conf = raw.get("confidence_score")
    agent_output: AgentOutput = {
        "content": raw.get("content") or "",
        "confidence_score": clamp_confidence(
            raw.get("recommendation", ""),
            float(conf if conf is not None else 0.5),
        ),
        "reasoning": raw.get("reasoning", ""),
        "raw_json": raw,
    }

    return {
        "agent_output": agent_output,
        "active_agent": "A1",
        "history": append_history(state, "A1", {
            "action": "resume_screening",
            "confidence_score": agent_output["confidence_score"],
            "retry_count": state.get("retry_count", 0),
            "recommendation": raw.get("recommendation", ""),
        }),
    }
