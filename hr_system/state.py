from __future__ import annotations
from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class AgentOutput(TypedDict):
    content: str
    confidence_score: float
    reasoning: str
    raw_json: dict[str, Any]


class ManagerDecision(TypedDict):
    verdict: Literal["APPROVE", "REJECT"]
    feedback: str
    reviewed_at: str


class HRState(TypedDict):
    # Task inputs
    task_id: str
    task_type: Literal["resume_screening", "interview_scheduling"]
    task_input: dict[str, Any]
    task_description: str

    # Routing metadata
    active_agent: str
    is_confidential: bool
    escalated_to_cos: bool

    # Sub-agent output
    agent_output: Optional[AgentOutput]

    # Retry tracking
    retry_count: int
    max_retries: int

    # Manager decision
    manager_decision: Optional[ManagerDecision]

    # COS output
    cos_output: Optional[str]

    # Terminal fields
    final_result: Optional[str]
    error: Optional[str]

    # Audit trail
    history: list[dict[str, Any]]
