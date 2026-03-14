from datetime import datetime, timezone
from hr_system.state import HRState, ManagerDecision
from hr_system.agents.base import call_llm, append_history
from hr_system.prompts.manager import SYSTEM_PROMPT, build_user_prompt


def manager_node(state: HRState) -> dict:
    """LangGraph node: Manager (TA Lead) — reviews sub-agent output."""
    user_prompt = build_user_prompt(
        task_type=state["task_type"],
        task_description=state["task_description"],
        agent_output=state.get("agent_output"),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", 3),
    )

    raw: dict = call_llm(SYSTEM_PROMPT, user_prompt, expect_json=True)

    verdict = raw.get("verdict", "REJECT").upper()
    if verdict not in ("APPROVE", "REJECT"):
        verdict = "REJECT"

    decision: ManagerDecision = {
        "verdict": verdict,
        "feedback": raw.get("feedback", ""),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Increment retry counter on REJECT
    new_retry_count = (
        state.get("retry_count", 0) + 1
        if verdict == "REJECT"
        else state.get("retry_count", 0)
    )

    # On APPROVE, set final_result from agent output
    final_result = None
    if verdict == "APPROVE":
        agent_output = state.get("agent_output") or {}
        final_result = agent_output.get("content", "")

    return {
        "manager_decision": decision,
        "retry_count": new_retry_count,
        "active_agent": "manager",
        "final_result": final_result,
        "history": append_history(state, "manager", {
            "verdict": verdict,
            "feedback": decision["feedback"],
            "retry_count": new_retry_count,
        }),
    }
