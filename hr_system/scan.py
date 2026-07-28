"""
Background two-stage scan engine.

A ScanJob runs the whole "Screen & Rank All" scan on its own thread, decoupled
from Streamlit's rerun cycle: one ThreadPoolExecutor saturates the triage stage
across ALL applicants, then a second pool deep-screens the promoted shortlist
(one strong-model screener call each — no manager/retry cascade).
The UI polls snapshot()/merged_results() on a timer (st.fragment) instead of
driving the work with chunk-by-chunk full-page reruns — so no worker idles at a
chunk boundary and tab navigation keeps working during a scan.

Thread-safety contract:
  - Worker functions NEVER touch Streamlit (no st.*, no session_state); they
    only read inputs and return plain dicts.
  - All shared result state is guarded by a lock; the UI thread only reads via
    snapshot()/recent_cards()/merged_results().
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from hr_system.freshteam import FreshteamClient
from hr_system.triage import triage_screen, needs_deep_screen, dedupe_by_applicant_id


def applicant_label(a: dict) -> str:
    c = a.get("candidate") or a
    name = f"{c.get('first_name','')} {c.get('last_name','')}".strip()
    return name or f"Applicant #{a.get('id')}"


class ScanJob:
    """A running two-stage scan for one job posting."""

    def __init__(self, job_id, applicants, workers=40, deep_workers=None):
        self.job_id = job_id
        # Dedupe by applicant id up front so the same person isn't screened
        # twice and the progress denominator matches what actually gets done.
        # Records without an id are kept (they degrade to per-applicant error
        # cards rather than being silently dropped).
        seen = set()
        deduped = []
        for a in applicants:
            aid = a.get("id")
            if aid is not None:
                if aid in seen:
                    continue
                seen.add(aid)
            deduped.append(a)
        self.applicants = deduped
        self.workers = max(1, int(workers))
        # Deep screening runs the full graph (several sequential calls per
        # candidate); default its concurrency to the same worker count.
        self.deep_workers = max(1, int(deep_workers or workers))
        self.total = len(self.applicants)

        self._lock = threading.Lock()
        self._triage = {}     # applicant_id -> triage card
        self._deep = {}       # applicant_id -> deep card
        self._order = []      # applicant_ids in triage-completion order
        self.survivors = []   # triage cards promoted to deep screening
        self.stage = "triage"  # triage | deep | done | error
        self.error = None
        self.cancelled = False
        self._thread = None

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def cancel(self):
        self.cancelled = True

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── worker bodies (run in pool threads; must not touch Streamlit) ─────────
    def _triage_one(self, app, client, job_detail):
        if self.cancelled:
            return None
        name = applicant_label(app)
        aid = app.get("id")  # defensive: a malformed record must not abort the batch
        try:
            if aid is None:
                raise ValueError("applicant record missing 'id'")
            ti = client.build_resume_screening_input(self.job_id, aid, job=job_detail)
            fs = triage_screen(ti)
            conf = (fs.get("agent_output") or {}).get("confidence_score", 0.0)
            return {"name": name, "applicant_id": aid, "confidence": conf,
                    "final_state": fs, "task_input": ti, "stage": "triage"}
        except Exception as e:
            return {"name": name, "applicant_id": aid, "confidence": -1,
                    "error": str(e), "final_state": {}, "task_input": {},
                    "stage": "triage"}

    def _deep_one(self, card):
        if self.cancelled:
            return None
        # Bulk deep screening runs ONE strong-model screener call (strict schema,
        # detailed analysis) — NOT the full manager/retry graph. The manager is
        # resume-blind, so it cannot improve the fit judgment; on a weak model it
        # spuriously rejects good analyses and cascades into 5-7 sequential calls
        # (~50s/candidate) with no accuracy gain. The Single Applicant tab still
        # uses the full graph for a considered one-off review.
        from hr_system.agents.resume_screener import resume_screener_node
        aid = card["applicant_id"]
        ti = card.get("task_input") or {}
        try:
            out = resume_screener_node({
                "task_input": ti, "manager_decision": None, "history": [], "retry_count": 0,
            })
            ao = out["agent_output"]
            fs = {
                "agent_output": ao,
                "final_result": ao.get("content", ""),
                "active_agent": "A1-deep",
                "history": out.get("history", []),
                "manager_decision": None,
            }
            conf = (ao or {}).get("confidence_score", 0.0)
            return {"name": card["name"], "applicant_id": aid, "confidence": conf,
                    "final_state": fs, "task_input": ti, "stage": "deep"}
        except Exception as e:
            # Deep screening failed — keep the triage verdict, flag the failure.
            fallback = dict(card)
            fallback["deep_error"] = str(e)
            return fallback

    # ── the background driver ────────────────────────────────────────────────
    def _run(self):
        try:
            client = FreshteamClient()
            job_detail = client.get_job_posting(self.job_id)
            if not (job_detail.get("description") or job_detail.get("job_description")):
                with self._lock:
                    self.error = "no_jd"
                    self.stage = "error"
                return

            # Stage 1 — triage every applicant.
            pool = ThreadPoolExecutor(max_workers=self.workers)
            try:
                futs = [pool.submit(self._triage_one, a, client, job_detail)
                        for a in self.applicants]
                for fut in as_completed(futs):
                    if self.cancelled:
                        break
                    card = fut.result()
                    if card is None:      # worker bailed early on cancel
                        continue
                    with self._lock:
                        if card["applicant_id"] not in self._triage:
                            self._order.append(card["applicant_id"])
                        self._triage[card["applicant_id"]] = card
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            if self.cancelled:
                with self._lock:
                    self.stage = "done"
                return

            # Build the shortlist from the COMPLETE triage set.
            with self._lock:
                triage_cards = dedupe_by_applicant_id(list(self._triage.values()))
                self.survivors = [c for c in triage_cards if needs_deep_screen(c)]
                self.stage = "deep"

            # Stage 2 — deep-screen the shortlist.
            pool = ThreadPoolExecutor(max_workers=self.deep_workers)
            try:
                futs = [pool.submit(self._deep_one, c) for c in self.survivors]
                for fut in as_completed(futs):
                    if self.cancelled:
                        break
                    card = fut.result()
                    if card is None:      # worker bailed early on cancel
                        continue
                    with self._lock:
                        self._deep[card["applicant_id"]] = card
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            with self._lock:
                self.stage = "done"
        except Exception as e:  # pragma: no cover - defensive
            with self._lock:
                self.error = str(e)
                self.stage = "error"

    # ── UI-thread readers ────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stage": self.stage,
                "error": self.error,
                "total": self.total,
                "triage_done": len(self._triage),
                "deep_total": len(self.survivors),
                "deep_done": len(self._deep),
                "deep_failed": sum(1 for c in self._deep.values() if c.get("deep_error")),
            }

    def recent_cards(self, n=8) -> list:
        with self._lock:
            ids = self._order[-n:]
            return [self._triage[i] for i in ids]

    def merged_results(self) -> list:
        """Deep result where available, else the triage card. Unsorted."""
        with self._lock:
            triage_cards = dedupe_by_applicant_id(list(self._triage.values()))
            deep = dict(self._deep)
        return [deep.get(c["applicant_id"], c) for c in triage_cards]
