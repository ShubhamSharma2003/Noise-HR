from hr_system.state import HRState, AgentOutput
from hr_system.agents.base import call_llm, append_history
from hr_system.prompts.resume_screener import SYSTEM_PROMPT, build_user_prompt


def resume_screener_node(state: HRState) -> dict:
    """LangGraph node: A1 — Resume Screener."""
    feedback = (state.get("manager_decision") or {}).get("feedback", "")

    user_prompt = build_user_prompt(
        resume_text=state["task_input"].get("resume_text", ""),
        job_description=state["task_input"].get("job_description", ""),
        job_title=state["task_input"].get("job_title", ""),
        feedback=feedback,
    )

    raw: dict = call_llm(SYSTEM_PROMPT, user_prompt, expect_json=True)

    agent_output: AgentOutput = {
        "content": raw.get("content", ""),
        "confidence_score": float(raw.get("confidence_score", 0.5)),
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
