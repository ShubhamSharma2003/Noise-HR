"""
Fast-path bulk triage: one cheap LLM call per resume, no manager/CoS loop.

Used by the "Screen & Rank All" scan. Candidates that triage as promising (or
land in the borderline band) are promoted to the full LangGraph deep screening;
everyone else keeps their triage verdict. The manager review only ever sees the
screener's prose — not the resume or JD — so skipping it here trades no
screening accuracy for a large cost/latency win.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from hr_system.agents.base import call_llm, TRIAGE_MODEL
from hr_system.prompts.resume_screener import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_JSON_SCHEMA,
    build_user_prompt,
    # re-exported so callers/tests can keep importing them from hr_system.triage
    CATEGORY_BANDS,
    clamp_confidence,
)

# Promote to deep screening. Deliberately GENEROUS: the only real accuracy risk
# in the funnel is the cheap model under-scoring a genuinely good candidate, so
# anything MODERATE_FIT or above — or above the (low) borderline confidence —
# gets the full deep screen. Set HR_DEEP_THRESHOLD=0.0 to deep-screen everyone,
# or higher to be stricter (and faster).
DEEP_SCREEN_CATEGORIES = {"PERFECT_FIT", "STRONG_FIT", "GOOD_FIT", "MODERATE_FIT"}
BORDERLINE_CONFIDENCE = float(os.environ.get("HR_DEEP_THRESHOLD", "0.30"))


def triage_summary_md(raw: dict) -> str:
    """Render the compact triage JSON as markdown for the result card."""
    lines = []
    reasoning = raw.get("reasoning", "")
    if reasoning:
        lines.append(reasoning)
    strengths = raw.get("key_strengths") or []
    if strengths:
        lines.append("\n**Key strengths**")
        lines.extend(f"- {s}" for s in strengths)
    concerns = raw.get("concerns") or []
    if concerns:
        lines.append("\n**Concerns**")
        lines.extend(f"- {c}" for c in concerns)
    dims = raw.get("dimension_scores") or {}
    if dims:
        lines.append("\n**Dimension scores (1-10)**")
        labels = [
            ("hard_skills", "Hard skills"), ("experience", "Experience"),
            ("education", "Education"), ("career_trajectory", "Career trajectory"),
            ("role_alignment", "Role alignment"), ("red_flags", "Red flags"),
        ]
        lines.append(" · ".join(f"{label}: {dims.get(key, '—')}" for key, label in labels))
    lines.append("\n_Triage screening — expanded analysis is generated for shortlisted candidates._")
    return "\n".join(lines)


def triage_screen(task_input: dict) -> dict:
    """
    Screen one resume with the cheap triage model.
    Returns a dict shaped like the graph's final_state so the UI can render
    triage and deep results with the same code.
    """
    user_prompt = build_user_prompt(
        resume_text=task_input.get("resume_text", ""),
        job_description=task_input.get("job_description", ""),
        job_title=task_input.get("job_title", ""),
    )

    raw: dict = call_llm(
        TRIAGE_SYSTEM_PROMPT,
        user_prompt,
        model=TRIAGE_MODEL,
        # The compact answer needs ~250 tokens, but reasoning models spend
        # part of the limit on reasoning tokens before any visible output.
        max_tokens=2000,
        json_schema=TRIAGE_JSON_SCHEMA,
        prompt_cache_key=f"triage-job-{task_input.get('job_id')}",
    )

    summary = triage_summary_md(raw)
    # .get(default) only fires when the key is ABSENT; guard an explicit null.
    conf = raw.get("confidence_score")
    confidence = clamp_confidence(
        raw.get("recommendation", ""),
        float(conf if conf is not None else 0.0),
    )
    agent_output = {
        "content": summary,
        "confidence_score": confidence,
        "reasoning": raw.get("reasoning", ""),
        "raw_json": raw,
    }
    return {
        "agent_output": agent_output,
        "final_result": summary,
        "active_agent": "A1-triage",
        "history": [{
            "node": "A1-triage",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "triage_screening",
            "model": TRIAGE_MODEL,
            "confidence_score": agent_output["confidence_score"],
            "recommendation": raw.get("recommendation", ""),
        }],
    }


def dedupe_by_applicant_id(cards: list) -> list:
    """Collapse duplicate scan cards for the same applicant (last write wins)."""
    by_id = {}
    for card in cards:
        by_id[card["applicant_id"]] = card
    return list(by_id.values())


def needs_deep_screen(result: dict) -> bool:
    """Decide whether a triage scan result should get the full deep screening."""
    if result.get("error"):
        return False
    final_state = result.get("final_state") or {}
    agent_output = final_state.get("agent_output") or {}
    raw = agent_output.get("raw_json") or {}
    rec = (raw.get("recommendation") or "").upper().replace(" ", "_")
    confidence = agent_output.get("confidence_score") or 0.0
    return rec in DEEP_SCREEN_CATEGORIES or confidence >= BORDERLINE_CONFIDENCE
