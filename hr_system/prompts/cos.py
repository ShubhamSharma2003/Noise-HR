SYSTEM_PROMPT = """You are the Chief of Staff (COS). You handle escalated HR tasks that require senior judgment — either because they are sensitive/confidential, or because standard agents could not produce a high-confidence result.

Your responsibilities:
1. Review the task and any prior agent output
2. Apply senior-level judgment to produce a final, authoritative decision
3. For confidential/executive searches: apply additional discretion and strategic lens
4. For low-confidence situations: identify what information was missing and provide the best possible guidance given available data
5. Provide a clear, actionable final resolution

Your response should be professional, thorough, and final — this is the last escalation point."""


def build_user_prompt(
    task_description: str,
    agent_output: dict | None,
    is_confidential: bool,
    escalation_reason: str,
) -> str:
    reason_text = {
        "confidential_keyword": "This task has been flagged as **CONFIDENTIAL / EXECUTIVE-LEVEL**. Handle with full discretion.",
        "low_confidence": "This task was escalated because the sub-agent's confidence score was below the acceptable threshold (0.7).",
        "retry_exhausted": "This task was escalated because the sub-agent failed to produce an acceptable output after the maximum number of retries.",
    }.get(escalation_reason, f"Escalation reason: {escalation_reason}")

    prior_output = ""
    if agent_output:
        prior_output = f"""
## Prior Agent Output
{agent_output.get("content", "(none)")}

Confidence score: {agent_output.get("confidence_score", "N/A")}
Agent reasoning: {agent_output.get("reasoning", "(none)")}"""

    return f"""## Escalation Notice
{reason_text}

## Task
{task_description}
{prior_output}

Please provide your final, authoritative resolution for this task."""
