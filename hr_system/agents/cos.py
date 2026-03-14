from hr_system.state import HRState
from hr_system.agents.base import call_llm, append_history
from hr_system.prompts.cos import SYSTEM_PROMPT, build_user_prompt

SENSITIVE_KEYWORDS = {"confidential", "executive"}


def _determine_escalation_reason(state: HRState) -> str:
    """Determine why this task was escalated to COS."""
    combined = (
        state.get("task_description", "").lower()
        + " "
        + (state.get("agent_output") or {}).get("content", "").lower()
    )
    if any(kw in combined for kw in SENSITIVE_KEYWORDS):
        return "confidential_keyword"
    score = (state.get("agent_output") or {}).get("confidence_score", 1.0)
    if score < 0.7:
        return "low_confidence"
    return "retry_exhausted"


def cos_node(state: HRState) -> dict:
    """LangGraph node: COS (Chief of Staff) — handles escalated tasks."""
    escalation_reason = _determine_escalation_reason(state)

    user_prompt = build_user_prompt(
        task_description=state.get("task_description", ""),
        agent_output=state.get("agent_output"),
        is_confidential=state.get("is_confidential", False),
        escalation_reason=escalation_reason,
    )

    resolution: str = call_llm(SYSTEM_PROMPT, user_prompt, expect_json=False)

    return {
        "cos_output": resolution,
        "final_result": resolution,
        "escalated_to_cos": True,
        "active_agent": "done",
        "history": append_history(state, "cos", {
            "escalation_reason": escalation_reason,
            "resolution_preview": resolution[:200],
        }),
    }
