"""
HR Multi-Agent System — CLI Entrypoint
Fetches live data from Freshteam, builds initial state, and invokes the LangGraph.

Usage:
    python run.py                        # Interactive mode — lists jobs and lets you pick
    python run.py --screen  <job_id> <applicant_id>
    python run.py --schedule <job_id> <applicant_id> --slots "2026-03-20T10:00" "2026-03-21T14:00"
"""
from __future__ import annotations

import json
import sys
import argparse
from datetime import datetime, timezone

from hr_system.graph import hr_graph
from hr_system.freshteam import FreshteamClient
from hr_system.state import HRState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    """Return a skeleton HRState with safe defaults."""
    defaults = {
        "active_agent":     "dispatcher",
        "retry_count":      0,
        "max_retries":      3,
        "is_confidential":  False,
        "escalated_to_cos": False,
        "history":          [],
        "agent_output":     None,
        "manager_decision": None,
        "cos_output":       None,
        "final_result":     None,
        "error":            None,
    }
    return {**defaults, **overrides}


def _print_result(final_state: dict) -> None:
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(final_state.get("final_result", "(no result)"))

    if final_state.get("escalated_to_cos"):
        print("\n[Escalated to Chief of Staff]")

    print("\n" + "=" * 60)
    print("AUDIT TRAIL")
    print("=" * 60)
    for i, event in enumerate(final_state.get("history", []), 1):
        ts = event.get("timestamp", "")
        node = event.get("node", "?")
        print(f"  {i}. [{ts}] {node.upper()}", end="")
        if "confidence_score" in event:
            print(f"  (confidence={event['confidence_score']:.2f})", end="")
        if "verdict" in event:
            print(f"  -> {event['verdict']}", end="")
        if "escalation_reason" in event:
            print(f"  -> escalated ({event['escalation_reason']})", end="")
        print()


# ── Task runners ──────────────────────────────────────────────────────────────

def run_resume_screening(job_id: int, applicant_id: int) -> None:
    print(f"\nFetching Freshteam data for job={job_id}, applicant={applicant_id}...")
    client = FreshteamClient()
    task_input = client.build_resume_screening_input(job_id, applicant_id)

    task_description = (
        f"Screen resume for {task_input['applicant_name']} "
        f"applying to: {task_input['job_title']}"
    )

    initial_state = _base_state(
        task_id=f"RS-{job_id}-{applicant_id}",
        task_type="resume_screening",
        task_description=task_description,
        task_input=task_input,
    )

    print(f"Task: {task_description}")
    print("Running agent graph...\n")
    final_state = hr_graph.invoke(initial_state)
    _print_result(final_state)


def run_interview_scheduling(
    job_id: int, applicant_id: int, available_slots: list[str]
) -> None:
    print(f"\nFetching Freshteam data for job={job_id}, applicant={applicant_id}...")
    client = FreshteamClient()
    task_input = client.build_interview_scheduling_input(
        job_id, applicant_id, available_slots
    )

    task_description = (
        f"Schedule interview for {task_input['candidate_name']} "
        f"applying to: {task_input['job_title']}"
    )

    initial_state = _base_state(
        task_id=f"IS-{job_id}-{applicant_id}",
        task_type="interview_scheduling",
        task_description=task_description,
        task_input=task_input,
    )

    print(f"Task: {task_description}")
    print("Running agent graph...\n")
    final_state = hr_graph.invoke(initial_state)
    _print_result(final_state)


def interactive_mode() -> None:
    """Interactive mode — ask for job ID directly, then list applicants."""
    print("Connecting to Freshteam...")
    client = FreshteamClient()

    job_id = int(input("Enter Job ID (e.g. 2000073751): ").strip())

    print(f"\nFetching applicants for job {job_id}...")
    try:
        applicants = client.get_applicants(job_id)
    except Exception as e:
        print(f"Error fetching applicants: {e}")
        sys.exit(1)

    if not applicants:
        print(f"No applicants found for job {job_id}.")
        sys.exit(0)

    job = client.get_job_posting(job_id)
    print(f"\nApplicants for '{job.get('title', job_id)}':\n")
    for i, a in enumerate(applicants[:10], 1):
        name = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
        print(f"  {i}. [{a['id']}] {name}")

    print()
    app_idx = int(input("Select applicant number: ")) - 1
    applicant = applicants[app_idx]
    applicant_id = applicant["id"]

    print("\nWhat would you like to do?")
    print("  1. Screen resume (A1)")
    print("  2. Schedule interview (A2)")
    action = input("Select action (1 or 2): ").strip()

    if action == "1":
        run_resume_screening(job_id, applicant_id)
    elif action == "2":
        print("\nEnter available slots (ISO-8601 or human-readable, one per line).")
        print("Leave blank and press Enter twice when done:")
        slots = []
        while True:
            s = input("  Slot: ").strip()
            if not s:
                break
            slots.append(s)
        if not slots:
            slots = ["2026-03-20T10:00+05:30", "2026-03-21T14:00+05:30", "2026-03-22T11:00+05:30"]
            print(f"  (using default slots: {slots})")
        run_interview_scheduling(job_id, applicant_id, slots)
    else:
        print("Invalid selection.")
        sys.exit(1)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HR Multi-Agent System")
    parser.add_argument("--screen", nargs=2, metavar=("JOB_ID", "APPLICANT_ID"),
                        help="Screen a resume: --screen <job_id> <applicant_id>")
    parser.add_argument("--schedule", nargs=2, metavar=("JOB_ID", "APPLICANT_ID"),
                        help="Schedule an interview: --schedule <job_id> <applicant_id>")
    parser.add_argument("--slots", nargs="+", metavar="SLOT",
                        help="Available time slots for interview scheduling")
    args = parser.parse_args()

    if args.screen:
        run_resume_screening(int(args.screen[0]), int(args.screen[1]))
    elif args.schedule:
        slots = args.slots or ["2026-03-20T10:00+05:30", "2026-03-21T14:00+05:30"]
        run_interview_scheduling(int(args.schedule[0]), int(args.schedule[1]), slots)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
