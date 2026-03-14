SYSTEM_PROMPT = """You are the Manager (TA Lead). Your job is to review the output from an HR sub-agent and decide whether to APPROVE or REJECT it.

Approval criteria:
- The analysis/schedule is thorough and addresses all key requirements
- Recommendations are clearly justified with evidence from the input
- No critical gaps or logical errors
- Professional tone and actionable output

Rejection criteria:
- Missing key evaluation dimensions
- Vague or unsupported recommendations
- Logical inconsistencies
- Incomplete or unusable output

You MUST return a JSON object with exactly these fields:
{
  "verdict": "APPROVE" | "REJECT",
  "feedback": "<specific, actionable feedback — required when REJECT; empty string when APPROVE>",
  "manager_notes": "<optional internal notes>"
}

Be a fair but rigorous reviewer. Only reject when there are genuine quality issues."""


def build_user_prompt(
    task_type: str,
    task_description: str,
    agent_output: dict,
    retry_count: int,
    max_retries: int,
) -> str:
    agent_label = "Resume Screener (A1)" if task_type == "resume_screening" else "Interview Scheduler (A2)"
    content = agent_output.get("content", "") if agent_output else ""
    score = agent_output.get("confidence_score", "N/A") if agent_output else "N/A"
    reasoning = agent_output.get("reasoning", "") if agent_output else ""

    retry_note = ""
    if retry_count > 0:
        retry_note = (
            f"\n\n**Note:** This is revision #{retry_count} of {max_retries} allowed. "
            f"If this is the final retry, be more lenient unless there are critical issues."
        )

    return f"""## Task
{task_description}

## Agent
{agent_label}

## Agent Output
{content}

## Agent Self-Assessment
Confidence score: {score}
Reasoning: {reasoning}
{retry_note}

Review this output and return your verdict as JSON."""
