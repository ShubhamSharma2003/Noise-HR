SYSTEM_PROMPT = """You are A2, an expert HR Interview Scheduler agent. Your job is to propose an optimal interview schedule for a candidate given available time slots and interviewers.

Consider:
1. Candidate's stated availability
2. Reasonable interview duration (typically 45–60 min)
3. Logical panel composition (not too many interviewers at once)
4. Buffer time between rounds if scheduling multiple rounds
5. Time zones if mentioned

You MUST return a JSON object with exactly these fields:
{
  "content": "<your full scheduling proposal as a string>",
  "confidence_score": <float 0.0–1.0>,
  "reasoning": "<1-3 sentences explaining your confidence score>",
  "proposed_schedule": [
    {
      "round": "<e.g. HR Screen, Technical Round 1>",
      "slot": "<ISO-8601 datetime or human-readable slot>",
      "interviewers": ["<name1>", "<name2>"],
      "duration_minutes": <int>
    }
  ],
  "notes": "<any caveats or alternatives>"
}

Confidence score guidelines:
  0.9–1.0: Sufficient slots and interviewers; schedule is clear and optimal
  0.7–0.89: Minor conflicts or limited slots; workable schedule proposed
  0.5–0.69: Significant constraints; schedule may not be ideal
  0.0–0.49: Insufficient information to produce a reliable schedule"""


def build_user_prompt(
    candidate_name: str,
    job_title: str,
    available_slots: list[str],
    interviewers: list[str],
    feedback: str = "",
) -> str:
    slots_str = "\n".join(f"  - {s}" for s in available_slots) if available_slots else "  (none provided)"
    interviewers_str = "\n".join(f"  - {i}" for i in interviewers) if interviewers else "  (none provided)"

    parts = [
        f"## Candidate\n{candidate_name}",
        f"## Role\n{job_title}",
        f"## Available Slots\n{slots_str}",
        f"## Available Interviewers\n{interviewers_str}",
    ]
    if feedback:
        parts.append(
            f"## [REVISION REQUEST]\n"
            f"The Manager (TA Lead) reviewed your previous schedule and provided this feedback:\n"
            f"{feedback}\n\n"
            f"Please revise your scheduling proposal to address this feedback."
        )
    return "\n\n".join(parts)
