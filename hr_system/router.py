from __future__ import annotations
from hr_system.state import HRState

CONFIDENCE_THRESHOLD = 0.7
SENSITIVE_KEYWORDS = {"confidential", "executive"}


def route_after_subagent(state: HRState) -> str:
    """
    Decides the next node after A1 or A2 produces output.

    Priority:
      1. Keyword check (task description + output content) → "cos"
      2. Confidence score < CONFIDENCE_THRESHOLD           → "cos"
      3. Otherwise                                         → "manager"
    """
    task_desc = (state.get("task_description") or "").lower()
    output_content = ((state.get("agent_output") or {}).get("content") or "").lower()
    combined = task_desc + " " + output_content

    # 1. Keyword escalation
    if any(kw in combined for kw in SENSITIVE_KEYWORDS):
        return "cos"

    # 2. Confidence threshold
    score = (state.get("agent_output") or {}).get("confidence_score", 1.0)
    if score < CONFIDENCE_THRESHOLD:
        return "cos"

    # 3. Normal path
    return "manager"


def route_after_manager(state: HRState) -> str:
    """
    Decides the next node after the Manager reviews output.

      - APPROVE                         → "END"
      - REJECT + retry_count < max      → back to sub-agent ("A1" or "A2")
      - REJECT + retry_count >= max     → "cos" (safety valve)
    """
    verdict = ((state.get("manager_decision") or {}).get("verdict") or "REJECT").upper()

    if verdict == "APPROVE":
        return "END"

    # REJECT path
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count >= max_retries:
        return "cos"

    return "A1" if state.get("task_type") == "resume_screening" else "A2"
