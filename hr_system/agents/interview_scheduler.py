from hr_system.state import HRState, AgentOutput
from hr_system.agents.base import call_llm, append_history
from hr_system.prompts.interview_scheduler import SYSTEM_PROMPT, build_user_prompt


def interview_scheduler_node(state: HRState) -> dict:
    """LangGraph node: A2 — Interview Scheduler."""
    feedback = (state.get("manager_decision") or {}).get("feedback", "")

    user_prompt = build_user_prompt(
        candidate_name=state["task_input"].get("candidate_name", "Unknown"),
        job_title=state["task_input"].get("job_title", ""),
        available_slots=state["task_input"].get("available_slots", []),
        interviewers=state["task_input"].get("interviewers", []),
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
        "active_agent": "A2",
        "history": append_history(state, "A2", {
            "action": "interview_scheduling",
            "confidence_score": agent_output["confidence_score"],
            "retry_count": state.get("retry_count", 0),
            "proposed_schedule": raw.get("proposed_schedule", []),
        }),
    }
