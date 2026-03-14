from langgraph.graph import StateGraph, END

from hr_system.state import HRState
from hr_system.agents.resume_screener import resume_screener_node
from hr_system.agents.interview_scheduler import interview_scheduler_node
from hr_system.agents.manager import manager_node
from hr_system.agents.cos import cos_node
from hr_system.router import route_after_subagent, route_after_manager


def dispatch_node(state: HRState) -> dict:
    """
    Entry-point node. No LLM calls — just fills in missing defaults so every
    downstream node can safely read any field without KeyError.
    """
    return {
        "retry_count":       state.get("retry_count", 0),
        "max_retries":       state.get("max_retries", 3),
        "is_confidential":   state.get("is_confidential", False),
        "escalated_to_cos":  state.get("escalated_to_cos", False),
        "history":           state.get("history") or [],
        "agent_output":      state.get("agent_output"),
        "manager_decision":  state.get("manager_decision"),
        "cos_output":        state.get("cos_output"),
        "final_result":      state.get("final_result"),
        "error":             state.get("error"),
    }


def build_graph():
    """Assemble and compile the HR multi-agent LangGraph."""
    graph = StateGraph(HRState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("dispatcher", dispatch_node)
    graph.add_node("A1",         resume_screener_node)
    graph.add_node("A2",         interview_scheduler_node)
    graph.add_node("manager",    manager_node)
    graph.add_node("cos",        cos_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("dispatcher")

    # ── dispatcher → A1 or A2 based on task_type ──────────────────────────────
    graph.add_conditional_edges(
        "dispatcher",
        lambda s: "A1" if s.get("task_type") == "resume_screening" else "A2",
        {"A1": "A1", "A2": "A2"},
    )

    # ── Sub-agent → manager or cos ────────────────────────────────────────────
    graph.add_conditional_edges(
        "A1",
        route_after_subagent,
        {"cos": "cos", "manager": "manager"},
    )
    graph.add_conditional_edges(
        "A2",
        route_after_subagent,
        {"cos": "cos", "manager": "manager"},
    )

    # ── manager → END / A1 / A2 / cos ─────────────────────────────────────────
    graph.add_conditional_edges(
        "manager",
        route_after_manager,
        {"END": END, "A1": "A1", "A2": "A2", "cos": "cos"},
    )

    # ── COS is always terminal ─────────────────────────────────────────────────
    graph.add_edge("cos", END)

    return graph.compile()


# Module-level compiled graph — import and call directly
hr_graph = build_graph()
