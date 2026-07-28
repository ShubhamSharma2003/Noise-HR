"""
HR Multi-Agent System — Streamlit UI
Run with: streamlit run app.py
"""
import os
import json
import requests as _requests
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from hr_system.freshteam import FreshteamClient
from hr_system.graph import hr_graph
from hr_system.scan import ScanJob
from hr_system.agents.base import MODEL as DEEP_MODEL, TRIAGE_MODEL

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Agent System", page_icon="🧑‍💼", layout="wide")

st.markdown("""
<style>
.verdict-perfect-fit {
    background:#c3e6cb;color:#155724;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-strong-fit {
    background:#d4edda;color:#155724;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-good-fit {
    background:#d1ecf1;color:#0c5460;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-moderate-fit {
    background:#fff3cd;color:#856404;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-low-fit {
    background:#f8d7da;color:#721c24;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-no-fit {
    background:#d6d8d9;color:#1b1e21;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-pending {
    background:#fff3cd;color:#856404;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.rank-card {
    border:1px solid #dee2e6;border-radius:10px;
    padding:16px 20px;margin-bottom:12px;
    background:#ffffff;
}
.rank-number {
    font-size:2rem;font-weight:900;color:#6c757d;
    line-height:1;
}
.audit-step {
    background:#f8f9fa;border-left:4px solid #6c757d;
    padding:6px 12px;margin:4px 0;border-radius:4px;
    font-family:monospace;font-size:0.8rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧑‍💼 HR Multi-Agent System")
st.caption("Powered by LangGraph + OpenAI · Connected to Freshteam")
st.divider()

# ── Shared helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def format_resume(raw_text: str) -> str:
    if not raw_text or len(raw_text.strip()) < 20:
        return "_No resume content available._"
    try:
        response = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a resume parser. Given raw text extracted from a resume file, "
                        "reformat it into clean, structured markdown with these sections (only include "
                        "sections that have actual data): \n"
                        "## Name\n## Contact\n## Summary\n## Experience\n## Education\n## Skills\n## Certifications\n\n"
                        "Use bullet points for lists. If the input does not look like a resume at all, "
                        "respond with exactly: `[Not a resume — raw content shown below]`"
                    ),
                },
                {"role": "user", "content": raw_text[:6000]},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Never let a formatting failure crash the caller — show the raw text.
        return f"_(AI formatting unavailable: {str(e)[:120]})_\n\n```\n{raw_text[:6000]}\n```"

@st.cache_data(show_spinner="Fetching jobs from Freshteam...", ttl=60)
def load_jobs():
    return FreshteamClient().get_job_postings()

@st.cache_data(show_spinner="Fetching applicants...", ttl=30)
def load_applicants(job_id):
    return FreshteamClient().get_applicants(job_id)

@st.cache_data(show_spinner="Fetching job details...", ttl=60)
def load_job_detail(job_id):
    return FreshteamClient().get_job_posting(job_id)

def applicant_label(a):
    c = a.get("candidate") or a
    name = f"{c.get('first_name','')} {c.get('last_name','')}".strip()
    return name or f"Applicant #{a['id']}"

def run_screening(job_id, applicant_id, job=None):
    client = FreshteamClient()
    task_input = client.build_resume_screening_input(job_id, applicant_id, job=job)
    state = {
        "task_id": f"RS-{job_id}-{applicant_id}",
        "task_type": "resume_screening",
        "task_description": f"Screen resume for {task_input['applicant_name']}",
        "task_input": task_input,
        "active_agent": "dispatcher",
        "retry_count": 0, "max_retries": 3,
        "is_confidential": False, "escalated_to_cos": False,
        "history": [], "agent_output": None, "manager_decision": None,
        "cos_output": None, "final_result": None, "error": None,
    }
    return hr_graph.invoke(state), task_input

def run_scheduling(job_id, applicant_id, slots):
    client = FreshteamClient()
    task_input = client.build_interview_scheduling_input(job_id, applicant_id, slots)
    state = {
        "task_id": f"IS-{job_id}-{applicant_id}",
        "task_type": "interview_scheduling",
        "task_description": f"Schedule interview for {task_input['candidate_name']}",
        "task_input": task_input,
        "active_agent": "dispatcher",
        "retry_count": 0, "max_retries": 3,
        "is_confidential": False, "escalated_to_cos": False,
        "history": [], "agent_output": None, "manager_decision": None,
        "cos_output": None, "final_result": None, "error": None,
    }
    return hr_graph.invoke(state), task_input

_VERDICT_RANK = {
    "PERFECT_FIT": 0, "STRONG_YES": 0,
    "STRONG_FIT":  1, "YES":        1,
    "GOOD_FIT":    2,
    "MODERATE_FIT":3, "MAYBE":      3,
    "LOW_FIT":     4,
    "NO_FIT":      5, "NO":         5,
}

_DIMENSION_KEYS = ("hard_skills", "experience", "education",
                   "career_trajectory", "role_alignment", "red_flags")


def _dimension_total(raw_json):
    """Sum of the 6 sub-scores (each 1-10, higher = better). Used to break ties
    when many candidates share the same coarse confidence % — the model scores
    confidence in blunt steps, but the sub-scores separate similar resumes."""
    dims = raw_json.get("dimension_scores") or {}
    total = 0.0
    for k in _DIMENSION_KEYS:
        try:
            total += float(dims.get(k, 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def _verdict_sort_key(r):
    """Primary: verdict rank (lower = better). Secondary: confidence descending.
    Tertiary: total sub-score descending (separates same-confidence clusters)."""
    if r.get("error"):
        return (99, 0.0, 0.0)
    fs = r.get("final_state", {})
    raw_json = ((fs.get("agent_output") or {}).get("raw_json") or {})
    rec = raw_json.get("recommendation", "").upper().replace(" ", "_")
    rank = _VERDICT_RANK.get(rec, 6)
    return (rank, -r.get("confidence", 0), -_dimension_total(raw_json))

def verdict_badge(final_state):
    # Extract recommendation from agent output
    raw_json = ((final_state.get("agent_output") or {}).get("raw_json") or {})
    rec = raw_json.get("recommendation", "").upper().replace(" ", "_")

    badge_map = {
        "PERFECT_FIT": ("verdict-perfect-fit", "Perfect Fit"),
        "STRONG_FIT":  ("verdict-strong-fit",  "Strong Fit"),
        "GOOD_FIT":    ("verdict-good-fit",    "Good Fit"),
        "MODERATE_FIT":("verdict-moderate-fit", "Moderate Fit"),
        "LOW_FIT":     ("verdict-low-fit",     "Low Fit"),
        "NO_FIT":      ("verdict-no-fit",      "No Fit"),
        # Legacy mappings
        "STRONG_YES":  ("verdict-perfect-fit",  "Perfect Fit"),
        "YES":         ("verdict-strong-fit",   "Strong Fit"),
        "MAYBE":       ("verdict-moderate-fit",  "Moderate Fit"),
        "NO":          ("verdict-no-fit",        "No Fit"),
    }

    if rec in badge_map:
        css_class, label = badge_map[rec]
        return f'<span class="{css_class}">{label}</span>'

    # Fallback: use confidence score
    score = (final_state.get("agent_output") or {}).get("confidence_score", 0)
    if score >= 0.9:
        return '<span class="verdict-perfect-fit">Perfect Fit</span>'
    if score >= 0.75:
        return '<span class="verdict-strong-fit">Strong Fit</span>'
    if score >= 0.6:
        return '<span class="verdict-good-fit">Good Fit</span>'
    if score >= 0.4:
        return '<span class="verdict-moderate-fit">Moderate Fit</span>'
    if score >= 0.2:
        return '<span class="verdict-low-fit">Low Fit</span>'
    if score > 0:
        return '<span class="verdict-no-fit">No Fit</span>'
    return '<span class="verdict-pending">Pending</span>'

def render_audit(history):
    for i, event in enumerate(history, 1):
        node = event.get("node", "?").upper()
        ts = event.get("timestamp", "")[:19].replace("T", " ")
        conf = f"  confidence={event['confidence_score']:.2f}" if "confidence_score" in event else ""
        verdict = f"  → {event['verdict']}" if "verdict" in event else ""
        reason = f"  → escalated ({event['escalation_reason']})" if "escalation_reason" in event else ""
        st.markdown(
            f'<div class="audit-step">{i}. [{ts}] <b>{node}</b>{conf}{verdict}{reason}</div>',
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL TABS
# ════════════════════════════════════════════════════════════════════════════════
main_tab_screening, main_tab_requisition = st.tabs(["📋 Resume Screening", "📝 Requisition Form"])

# ════════════════════════════════════════════════════════════════════════════════
# MAIN TAB 1 — Resume Screening
# ════════════════════════════════════════════════════════════════════════════════
with main_tab_screening:

    # ── Job selector ──────────────────────────────────────────────────────────
    jobs = load_jobs()

    if not jobs:
        st.warning(
            "Could not load jobs from Freshteam (auth/permission/subdomain issue — "
            "see app logs for the HTTP status). Enter a Job ID manually below."
        )
        if "manual_job_ids" not in st.session_state:
            st.session_state.manual_job_ids = [2000073751]
        if "job_titles" not in st.session_state:
            st.session_state.job_titles = {}
        col_sel, col_inp, col_btn = st.columns([3, 2, 1])
        job_id = col_sel.selectbox(
            "Job ID",
            options=st.session_state.manual_job_ids,
            format_func=lambda jid: f"{st.session_state.job_titles[jid]} (#{jid})" if jid in st.session_state.job_titles else str(jid),
        )
        new_id = col_inp.text_input("Add Job ID")
        col_btn.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if col_btn.button("Add", key="add_job_id") and new_id and int(new_id) not in st.session_state.manual_job_ids:
            st.session_state.manual_job_ids.append(int(new_id))
            st.rerun()
    else:
        # ── Job dropdown (searchable) ─────────────────────────────────────────
        job_options = {j["id"]: j.get("title", "Untitled") for j in jobs}
        job_ids = list(job_options.keys())
        selected_job = st.selectbox(
            "Select Job Posting",
            options=job_ids,
            format_func=lambda jid: f"{job_options[jid]}  (#{jid})",
            key="selected_job_id",
        )
        job_id = selected_job

    st.divider()

    applicants = load_applicants(job_id)
    if not applicants:
        st.warning("No applicants found for this job.")
        st.stop()

    st.info(f"**{len(applicants)}** applicant(s) found for this job posting.")

    st.divider()

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    tab_rank, tab_single, tab_linkedin = st.tabs(["📊 Rank All Applicants", "🔍 Single Applicant", "🔗 LinkedIn Sourcing"])

    # ── Sub-tab 1: Rank All ───────────────────────────────────────────────────
    with tab_rank:
        st.subheader("Rank All Applicants by Resume Fit")
        st.caption("Screens every applicant and sorts them best-to-worst by AI confidence score.")

        ranked_key    = f"ranked_results_{job_id}"
        scanning_key  = f"scanning_{job_id}"
        scan_job_key  = f"scan_job_{job_id}"
        scan_error_key = f"scan_error_{job_id}"   # hard failure message (shown as error)
        scan_note_key  = f"scan_note_{job_id}"    # soft warning (e.g. deep stage failed)

        # Clear stale scan state from other jobs; cancel any orphaned scan.
        for k in list(st.session_state.keys()):
            if any(k.startswith(p) for p in (
                "ranked_results_", "scanning_", "scan_job_", "scan_error_", "scan_note_",
                # legacy keys from the old chunk-per-rerun scanner
                "scan_buf_", "scan_queue_", "scan_stage_", "deep_queue_", "deep_buf_",
            )) and not k.endswith(f"_{job_id}"):
                if k.startswith("scan_job_"):
                    old = st.session_state.get(k)
                    if old is not None and hasattr(old, "cancel"):
                        old.cancel()
                del st.session_state[k]

        def _render_rank_card(r, rank, live=False):
            fs   = r["final_state"]
            conf = r["confidence"]
            err  = r.get("error")
            with st.container():
                col_rank, col_info, col_conf, col_badge = st.columns([0.5, 3, 2, 2])
                col_rank.markdown(f'<div class="rank-number">#{rank}</div>', unsafe_allow_html=True)
                ft_url = (
                    f"https://gonoise.freshteam.com/hire/jobs/{job_id}"
                    f"/applicants/listview/{r['applicant_id']}"
                )
                stage_label = {"deep": "🔬 deep screened", "triage": "⚡ triage"}.get(r.get("stage"), "")
                col_info.markdown(
                    f"**{r['name']}**  \nID: `{r['applicant_id']}` &nbsp;"
                    f'<a href="{ft_url}" target="_blank" style="text-decoration:none;">'
                    f'<button style="padding:2px 10px;font-size:12px;border-radius:5px;'
                    f'border:1px solid #ccc;background:#f0f2f6;cursor:pointer;">'
                    f'🔗 Freshteam Profile</button></a>'
                    + (f' &nbsp;<span style="font-size:11px;color:#6c757d;">{stage_label}</span>' if stage_label else ""),
                    unsafe_allow_html=True,
                )
                if err:
                    col_conf.caption("Error")
                    col_badge.markdown('<span class="verdict-low-fit">⚠️ Error</span>', unsafe_allow_html=True)
                else:
                    col_conf.progress(max(conf, 0), text=f"{conf:.0%} fit")
                    col_badge.markdown(verdict_badge(fs), unsafe_allow_html=True)
                if r.get("deep_error"):
                    st.caption(f"⚠️ Deep screening failed — showing triage result. ({str(r['deep_error'])[:100]})")
                if not live:
                    with st.expander("View full result"):
                        if err:
                            st.error(err)
                        else:
                            res_tab, result_tab, audit_tab = st.tabs(["📄 Resume", "🤖 AI Result", "🕵️ Audit"])
                            with res_tab:
                                raw = r["task_input"].get("resume_text", "")
                                if raw:
                                    # Lazy: format with AI only on demand. Calling it
                                    # for every card here would fire one gpt-4o request
                                    # per applicant on render (collapsed expanders still
                                    # execute), stalling and crashing large lists.
                                    fmt_key = f"fmt_{r['applicant_id']}"
                                    if st.session_state.get(fmt_key):
                                        with st.spinner("Formatting resume..."):
                                            st.markdown(format_resume(raw))
                                    elif st.button("✨ Format with AI", key=f"fmtbtn_{r['applicant_id']}"):
                                        st.session_state[fmt_key] = True
                                        st.rerun()
                                    else:
                                        st.text(raw[:8000])
                                else:
                                    st.caption("No resume file attached.")
                            with result_tab:
                                st.markdown(fs.get("final_result", "_(no result)_"))
                            with audit_tab:
                                render_audit(fs.get("history", []))
                st.divider()

        # ── Pre-filter panel ──────────────────────────────────────────────────
        is_scanning = st.session_state.get(scanning_key, False)

        with st.expander("⚙️ Screening filters & settings", expanded=len(applicants) > 50):
            # Stage filter — fetch sub_stages from job posting's interview_process
            job_detail = load_job_detail(job_id)
            interview_process = job_detail.get("interview_process") or {}
            sub_stages_list = interview_process.get("sub_stages") or []

            # Parent stage ordering & labels
            _PARENT_ORDER = {"leads": 0, "candidature": 1, "screening": 2, "on_site": 3, "offer": 4, "hire": 5}
            _PARENT_LABELS = {
                "leads": "Leads", "candidature": "Candidature", "screening": "Screening",
                "on_site": "On-Site / Interviews", "offer": "Offer", "hire": "Hire",
            }

            # Build parent → sub_stage names map (ordered by position)
            parent_to_subs = {}
            for s in sorted(sub_stages_list, key=lambda s: s.get("position", 0)):
                parent_to_subs.setdefault(s.get("stage", ""), []).append(s["name"])

            # Get unique parent stages present in the pipeline, in order
            parent_keys = sorted(parent_to_subs.keys(), key=lambda k: _PARENT_ORDER.get(k, 99))

            # Helper: get applicant's parent stage value
            def _applicant_parent_stage(a):
                sv = a.get("stage")
                if isinstance(sv, dict):
                    return sv.get("name") or "Unknown"
                return sv or "Unknown"

            # Count applicants per parent stage
            parent_counts = {}
            for a in applicants:
                ps = _applicant_parent_stage(a)
                parent_counts[ps] = parent_counts.get(ps, 0) + 1

            # ── Level 1: Parent stage filter ──
            parent_options = [k for k in parent_keys]
            def _format_parent(key):
                label = _PARENT_LABELS.get(key, key.replace("_", " ").title())
                count = parent_counts.get(key, 0)
                return f"{label} ({count})" if count else label

            selected_parents = st.multiselect(
                "Filter by stage",
                options=parent_options,
                default=[],
                format_func=_format_parent,
                key="screen_parent_stages",
                disabled=is_scanning,
                placeholder="Choose stages...",
            )

            # ── Level 2: Sub-stage filter (populated from selected parents) ──
            available_subs = []
            for p in selected_parents:
                available_subs.extend(parent_to_subs.get(p, []))

            selected_subs = st.multiselect(
                "Filter by sub-stage",
                options=available_subs,
                default=available_subs,
                key="screen_sub_stages",
                disabled=is_scanning or not selected_parents,
                placeholder="Select parent stages first..." if not selected_parents else "Choose sub-stages...",
            )

            # Build filtered list: match applicant parent stage AND only if a sub in that parent is selected
            # Since API only gives parent stage, we filter by parent but only if at least one sub of that parent is selected
            active_parents = set()
            for sub_name in selected_subs:
                for p, subs in parent_to_subs.items():
                    if sub_name in subs:
                        active_parents.add(p)

            if selected_subs:
                stage_filtered = [
                    a for a in applicants
                    if _applicant_parent_stage(a) in active_parents
                ]
            else:
                stage_filtered = []

            workers = st.slider(
                "Parallel workers",
                min_value=1, max_value=100, value=40,
                key="screen_workers",
                disabled=is_scanning or not selected_subs,
                help="How many profiles to screen simultaneously. LLM calls are "
                     "network-bound, so higher is faster; rate-limit errors are "
                     "retried automatically with backoff. Lower this only if scans "
                     "keep hitting your OpenAI tier's limits.",
            )

            pre_filtered = stage_filtered
            if selected_subs:
                st.caption(
                    f"**{len(pre_filtered)}** of **{len(applicants)}** applicants will be screened "
                    f"({workers} at a time)."
                )
            else:
                st.caption("Select stages and sub-stages to begin screening.")

        # ── Button row ────────────────────────────────────────────────────────
        btn_col, stop_col = st.columns([2, 1])
        if btn_col.button("Screen & Rank All", type="primary", key="rank_all",
                          disabled=is_scanning or not selected_subs):
            st.session_state[scan_job_key] = ScanJob(job_id, pre_filtered, workers=workers).start()
            st.session_state[scanning_key] = True
            st.session_state.pop(ranked_key, None)
            st.session_state.pop("kw_rank_active", None)
            st.session_state.pop("kw_rank_matched", None)
            st.rerun()

        if stop_col.button("⏹ Stop", key="rank_stop",
                           disabled=not is_scanning):
            job = st.session_state.get(scan_job_key)
            if job is not None:
                job.cancel()
                merged = job.merged_results()   # keep whatever finished so far
                merged.sort(key=_verdict_sort_key)
                st.session_state[ranked_key] = merged
            st.session_state[scanning_key] = False
            st.session_state.pop(scan_job_key, None)
            st.rerun()

        # ── Live scanning: background ScanJob polled by a fragment ────────────
        # The scan runs on its own thread (hr_system/scan.py): stage 1 triages
        # every applicant on TRIAGE_MODEL, then survivors get a single-call
        # deep screen on DEEP_MODEL. This fragment reruns ONLY itself every
        # second to refresh progress, so the rest of the page — including the
        # tab bar — stays interactive during a scan.
        if st.session_state.get(scanning_key, False):

            @st.fragment(run_every="1s")
            def _scan_monitor():
                job = st.session_state.get(scan_job_key)
                if job is None:
                    st.session_state[scanning_key] = False
                    st.rerun()
                    return

                snap = job.snapshot()

                # Error messages are stashed in session_state (not st.error'd
                # here) because st.rerun() discards anything drawn in this run;
                # the main body renders them after the rerun.
                if snap["error"] == "no_jd":
                    st.session_state[scan_error_key] = (
                        "Couldn't fetch the job description from Freshteam — scan "
                        "stopped so candidates aren't ranked against an empty JD. "
                        "Click 'Screen & Rank All' to retry.")
                    st.session_state[scanning_key] = False
                    st.session_state.pop(scan_job_key, None)
                    st.rerun()
                    return
                if snap["error"]:
                    merged = job.merged_results()
                    merged.sort(key=_verdict_sort_key)
                    st.session_state[ranked_key]   = merged
                    st.session_state[scan_error_key] = (
                        f"Scan stopped after an error: {snap['error']}. "
                        f"Showing the {len(merged)} profile(s) screened before it failed.")
                    st.session_state[scanning_key] = False
                    st.session_state.pop(scan_job_key, None)
                    st.rerun()
                    return

                total = snap["total"]
                st.progress(snap["triage_done"] / total if total else 0.0,
                            text=f"Stage 1 · Triage ({TRIAGE_MODEL}): "
                                 f"{snap['triage_done']}/{total}")
                if snap["stage"] in ("deep", "done"):
                    dt = snap["deep_total"]
                    st.progress(snap["deep_done"] / dt if dt else 1.0,
                                text=f"Stage 2 · Deep screening shortlist ({DEEP_MODEL}): "
                                     f"{snap['deep_done']}/{dt}")

                # Live peek at the most-recent completed triage cards.
                for i, r in enumerate(job.recent_cards(6), 1):
                    _render_rank_card(r, i, live=True)

                if snap["stage"] == "done":
                    merged = job.merged_results()
                    merged.sort(key=_verdict_sort_key)
                    # Surface a mass deep-stage failure instead of a silent green
                    # "Done!" — e.g. every shortlisted candidate failed to screen.
                    if snap["deep_total"] and snap["deep_failed"] >= snap["deep_total"]:
                        st.session_state[scan_note_key] = (
                            f"Deep screening failed for all {snap['deep_total']} shortlisted "
                            f"candidate(s) — showing triage results. Check API limits/errors.")
                    elif snap.get("deep_failed"):
                        st.session_state[scan_note_key] = (
                            f"Deep screening failed for {snap['deep_failed']} of "
                            f"{snap['deep_total']} shortlisted candidate(s); those show triage results.")
                    st.session_state[ranked_key]   = merged
                    st.session_state[scanning_key] = False
                    st.session_state.pop(scan_job_key, None)
                    st.rerun()   # app-scope: leave scan mode, show final list

            _scan_monitor()

        # ── Scan error / notes (rendered in the body so they survive rerun) ───
        scan_error = None
        if not st.session_state.get(scanning_key, False):
            scan_error = st.session_state.pop(scan_error_key, None)
            scan_note  = st.session_state.pop(scan_note_key, None)
            if scan_error:
                st.error(scan_error)
            if scan_note:
                st.warning(scan_note)

        # ── Final ranked results (scan complete) ──────────────────────────────
        if ranked_key in st.session_state and not st.session_state.get(scanning_key, False):
            results = st.session_state[ranked_key]
            deep_n = sum(1 for r in results if r.get("stage") == "deep")
            funnel = f" ({deep_n} deep screened, {len(results) - deep_n} triaged)" if deep_n else ""
            if not scan_error:   # don't claim success when the scan errored out
                st.success(f"Done! Ranked {len(results)} applicants{funnel}.")
            st.divider()

            kw_col, btn_col, clear_col = st.columns([4, 1.2, 1])
            kw_input = kw_col.text_input(
                "kw_rank_label",
                placeholder="Smart filter — e.g. tier 1, FAANG experience, entrepreneur, D2C, startup",
                label_visibility="collapsed",
                key="kw_rank_input",
            )
            apply_kw_rank = btn_col.button("🔍 Filter", key="apply_kw_rank")
            clear_kw_rank = clear_col.button("✕ Clear", key="clear_kw_rank")

            if clear_kw_rank:
                st.session_state.pop("kw_rank_active", None)
                st.session_state.pop("kw_rank_matched", None)
                st.rerun()

            if apply_kw_rank and kw_input.strip():
                query = kw_input.strip()
                oai_kw = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                prog_kw = st.progress(0, text="AI filtering...")
                filter_system = (
                    "You are a smart recruiter filter. Given a search query and a resume, "
                    "reply ONLY 'yes' or 'no'.\n\n"
                    "Be semantically intelligent — match INTENT, not just exact words:\n"
                    "- 'tier 1' or 'tier 1 college' → IIT, IIM, IISc, NIT, BITS Pilani, "
                    "Delhi University, NSUT, DTU, ISI, IIIT-H, top-50 NIRF-ranked institutions\n"
                    "- 'tier 2' → decent but non-elite: state universities, Amity, LPU, "
                    "Chandigarh Univ, Manipal, VIT, SRM, etc.\n"
                    "- 'entrepreneur' → co-founded, startup founder, own business, CEO of own venture\n"
                    "- 'fintech' → payments, banking tech, neo-bank, lending platform\n"
                    "- 'FAANG' → Google, Meta, Amazon, Apple, Netflix, Microsoft\n"
                    "- 'remote' → works remotely, distributed team\n"
                    "- 'startup experience' → early-stage company, Series A/B, small team\n"
                    "- 'D2C' → direct to consumer brand, Shopify, e-commerce brand\n"
                    "- 'product management' → PM, product manager, product owner, roadmap\n\n"
                    "For comma-separated queries, ALL criteria must match. "
                    "Think about what a recruiter means, not literal text."
                )

                def _match_one(r):
                    resume_text = r.get("task_input", {}).get("resume_text", "") or r.get("name", "")
                    try:
                        resp = oai_kw.chat.completions.create(
                            model="gpt-4o-mini",
                            max_tokens=5,
                            temperature=0,
                            messages=[
                                {"role": "system", "content": filter_system},
                                {"role": "user", "content": f"Query: {query}\n\nResume:\n{resume_text[:3000]}"},
                            ],
                        )
                        return r["applicant_id"], resp.choices[0].message.content.strip().lower().startswith("yes")
                    except Exception:
                        return r["applicant_id"], True  # fail open: keep the profile

                matched = []
                done_ct = 0
                with ThreadPoolExecutor(max_workers=10) as pool:
                    futures = [pool.submit(_match_one, r) for r in results]
                    for fut in as_completed(futures):
                        aid, ok = fut.result()
                        if ok:
                            matched.append(aid)
                        done_ct += 1
                        prog_kw.progress(done_ct / len(results))
                prog_kw.empty()
                st.session_state["kw_rank_active"] = query
                st.session_state["kw_rank_matched"] = matched
                st.rerun()

            display_results = results
            if st.session_state.get("kw_rank_active"):
                matched_set = set(st.session_state["kw_rank_matched"])
                display_results = [r for r in results if r["applicant_id"] in matched_set]
                active_kws = st.session_state["kw_rank_active"]
                kw_label = " · ".join(f'**"{k.strip()}"**' for k in active_kws.split(",") if k.strip())
                st.info(f"🔍 Keywords: {kw_label} — {len(display_results)} of {len(results)} profile(s) matched.")
                if not display_results:
                    st.warning("No profiles matched. Try different keywords or click **✕ Clear**.")

            # Re-sort at display time so the ranking (incl. the sub-score
            # tiebreaker) applies even to results scanned before this ran.
            display_results = sorted(display_results, key=_verdict_sort_key)

            st.divider()

            # Paginate — rendering hundreds of cards (each an expander with tabs)
            # at once is slow and unwieldy; show a page at a time.
            PER_PAGE = 25
            total_n = len(display_results)
            n_pages = max(1, (total_n + PER_PAGE - 1) // PER_PAGE)
            page_key = f"rank_page_{job_id}"
            cur = min(max(st.session_state.get(page_key, 1), 1), n_pages)

            if n_pages > 1:
                p1, p2, p3 = st.columns([1, 2, 1])
                if p1.button("‹ Prev", disabled=cur <= 1, key=f"pg_prev_{job_id}"):
                    st.session_state[page_key] = cur - 1
                    st.rerun()
                p2.markdown(
                    f"<div style='text-align:center;padding-top:6px;'>Page {cur} of {n_pages} "
                    f"· showing {(cur-1)*PER_PAGE+1}–{min(cur*PER_PAGE, total_n)} of {total_n}</div>",
                    unsafe_allow_html=True,
                )
                if p3.button("Next ›", disabled=cur >= n_pages, key=f"pg_next_{job_id}"):
                    st.session_state[page_key] = cur + 1
                    st.rerun()

            start = (cur - 1) * PER_PAGE
            for rank, r in enumerate(display_results[start:start + PER_PAGE], start + 1):
                _render_rank_card(r, rank, live=False)

    # ── Sub-tab 2: Single Applicant ───────────────────────────────────────────
    with tab_single:
        st.subheader("Single Applicant")

        app_map = {applicant_label(a): a["id"] for a in applicants}
        selected_label = st.selectbox("Applicant", list(app_map.keys()))
        applicant_id = app_map[selected_label]

        action = st.radio("Action", ["Screen Resume", "Schedule Interview"], horizontal=True)

        if action == "Schedule Interview":
            import datetime
            if "slot_count" not in st.session_state:
                st.session_state.slot_count = 1

            slots = []
            for si in range(st.session_state.slot_count):
                c_date, c_time, c_del = st.columns([3, 2, 0.5])
                default_date = datetime.date.today() + datetime.timedelta(days=6 + si)
                d = c_date.date_input(f"Slot {si+1} — Date", value=default_date, key=f"slot_date_{si}")
                t = c_time.time_input(f"Slot {si+1} — Time (IST)", value=datetime.time(10, 0), key=f"slot_time_{si}", step=900)
                slot_iso = f"{d.strftime('%Y-%m-%d')}T{t.strftime('%H:%M')}+05:30"
                slots.append(slot_iso)
                if st.session_state.slot_count > 1:
                    if c_del.button("✕", key=f"del_slot_{si}", help="Remove slot"):
                        st.session_state.slot_count -= 1
                        st.rerun()

            if st.session_state.slot_count < 5:
                if st.button("＋ Add slot", key="add_slot"):
                    st.session_state.slot_count += 1
                    st.rerun()

        if st.button("Run Agent", type="primary", key="single_run"):
            with st.spinner("Running..."):
                if action == "Screen Resume":
                    final_state, task_input = run_screening(job_id, applicant_id)
                else:
                    final_state, task_input = run_scheduling(job_id, applicant_id, slots)

            name = task_input.get("applicant_name") or task_input.get("candidate_name", "—")
            agent_out = final_state.get("agent_output") or {}
            conf = agent_out.get("confidence_score")

            c1, c2, c3 = st.columns(3)
            c1.metric("Candidate", name)
            c2.metric("Job", task_input.get("job_title", "—"))
            if conf is not None:
                c3.metric("AI Confidence", f"{conf:.0%}")

            st.divider()
            st.markdown(verdict_badge(final_state), unsafe_allow_html=True)
            st.divider()
            st.markdown(final_state.get("final_result", "_(no result)_"))

            manager = final_state.get("manager_decision") or {}
            if manager.get("feedback"):
                with st.expander("Manager Feedback"):
                    st.write(manager["feedback"])

            st.divider()
            st.subheader("Audit Trail")
            render_audit(final_state.get("history", []))

            if action == "Screen Resume":
                st.divider()
                st.subheader("Resume")
                raw = task_input.get("resume_text", "")
                if raw:
                    with st.spinner("Formatting resume..."):
                        st.markdown(format_resume(raw))
                else:
                    st.caption("No resume file attached.")

    # ── Sub-tab 3: LinkedIn Sourcing ──────────────────────────────────────────
    with tab_linkedin:
        st.subheader("LinkedIn Profile Sourcing")
        st.caption("Describe the role → AI builds a JD → AI extracts search params → Apify scrapes LinkedIn profiles.")

        APIFY_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
        APIFY_ACTOR = "M2FMdjRVeF1HPGFcc"

        # Read prefilled values from Requisition Form (if submitted)
        _rq_title  = st.session_state.get("rq_title", "")
        _rq_spec   = st.session_state.get("rq_spec", "")
        _rq_exp    = st.session_state.get("rq_exp", "")
        _rq_notice = st.session_state.get("rq_notice", "")
        _rq_max    = int(st.session_state.get("rq_max", 20))

        if _rq_title:
            st.info("Fields pre-filled from the Requisition Form. Edit if needed, then click Source.")

        with st.form("linkedin_sourcing_form"):
            col1, col2 = st.columns(2)
            li_title  = col1.text_input("Job Title", value=_rq_title, placeholder="e.g. Senior iOS Developer")
            li_spec   = col1.text_input("Specialisation", value=_rq_spec, placeholder="e.g. Swift, SwiftUI, UIKit")
            li_exp    = col2.text_input("Work Experience", value=_rq_exp, placeholder="e.g. 3-5 years")
            li_notice = col2.text_input("Notice Period", value=_rq_notice, placeholder="e.g. Immediate / 30 days")
            li_max    = col2.number_input("Max Profiles to Fetch", min_value=5, max_value=100, value=_rq_max, step=5)
            submitted = st.form_submit_button("🚀 Source LinkedIn Profiles", type="primary")

        if submitted and li_title.strip():
            oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            with st.spinner("Step 1/3 — Generating job description..."):
                jd_resp = oai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an HR professional creating a concise and professional job description.\n"
                                "Create a structured job description with: Job Title, Job Summary (2-3 lines), "
                                "Key Responsibilities (5-7 bullets), Required Skills & Qualifications (5-6 bullets), "
                                "Experience Required, Location, Notice Period Preference.\n"
                                "Keep the tone professional. Location is Delhi NCR."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Job Title: {li_title}\nSpecialisation: {li_spec}\n"
                                f"Work Experience Required: {li_exp}\nNotice Period Preference: {li_notice}"
                            ),
                        },
                    ],
                )
                job_description = jd_resp.choices[0].message.content.strip()

            with st.expander("Generated Job Description", expanded=False):
                st.markdown(job_description)

            with st.spinner("Step 2/3 — Extracting LinkedIn search parameters..."):
                params_resp = oai.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Extract LinkedIn search parameters from the job description and return valid JSON with these exact keys:\n"
                                "- currentJobTitles: list of 3-5 relevant job title variants\n"
                                "- locations: [\"Delhi, India\", \"Noida, India\", \"Gurgaon, India\", \"Faridabad, India\", \"Ghaziabad, India\"]\n"
                                "- searchQuery: concise keyword string for LinkedIn search\n"
                                "- functionIds: pick 1-2 most relevant IDs (as strings) from: 1=Accounting, 2=Administrative, 3=Arts and Design, 4=Business Development, 6=Consulting, 7=Education, 8=Engineering, 10=Finance, 12=Human Resources, 13=Information Technology, 15=Marketing, 19=Product Management, 24=Research, 25=Sales\n"
                                "- seniorityLevelIds: pick most relevant IDs (as strings) from: 100=In Training, 110=Entry Level, 120=Senior, 130=Strategic, 200=Entry Level Manager, 210=Experienced Manager, 220=Director, 300=Vice President\n"
                                "- yearsOfExperienceIds: pick most relevant IDs (as strings) from: 1=Less than 1 year, 2=1 to 2 years, 3=3 to 5 years, 4=6 to 10 years, 5=More than 10 years\n"
                                "- autoQuerySegmentation: false\n"
                                "- recentlyChangedJobs: false\n"
                                "- profileScraperMode: \"Full + email search\"\n"
                                "- maxItems: 20"
                            ),
                        },
                        {"role": "user", "content": job_description},
                    ],
                )
                apify_params = json.loads(params_resp.choices[0].message.content)
                apify_params["maxItems"] = int(li_max)

            with st.expander("Apify Search Parameters", expanded=False):
                st.json(apify_params)

            with st.spinner(f"Step 3/3 — Scraping LinkedIn profiles (up to {int(li_max)}, may take ~2 min)..."):
                try:
                    apify_resp = _requests.post(
                        f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/run-sync-get-dataset-items",
                        headers={
                            "Accept": "application/json",
                            "Authorization": f"Bearer {APIFY_TOKEN}",
                        },
                        json=apify_params,
                        timeout=300,
                    )
                    with st.expander("Apify Raw Response (debug)", expanded=False):
                        st.write(f"Status code: {apify_resp.status_code}")
                        try:
                            raw_json = apify_resp.json()
                            st.json(raw_json if isinstance(raw_json, dict) else {"items": raw_json[:2] if raw_json else []})
                        except Exception:
                            st.text(apify_resp.text[:2000])
                    apify_resp.raise_for_status()
                    profiles = apify_resp.json()
                    # Handle case where response is wrapped in a dict
                    if isinstance(profiles, dict):
                        profiles = profiles.get("items", profiles.get("data", []))
                except Exception as e:
                    st.error(f"Apify request failed: {e}")
                    profiles = []

            if not profiles:
                st.warning("No profiles returned. Try adjusting the job title or specialisation.")
            else:
                st.success(f"{len(profiles)} LinkedIn profile(s) found.")
                st.divider()

                for idx, p in enumerate(profiles, 1):
                    name = f"{p.get('firstName', '')} {p.get('lastName', '')}".strip() or "—"
                    headline = p.get("headline", "—")
                    loc_raw = p.get("location")
                    if isinstance(loc_raw, dict):
                        location = loc_raw.get("parsed", {}).get("text", "—")
                    else:
                        location = loc_raw or "—"
                    emails = p.get("emails") or []
                    email = emails[0].get("email", "—") if emails and isinstance(emails[0], dict) else "—"
                    skills_data = p.get("topSkills") or []
                    if isinstance(skills_data, str):
                        skills = skills_data
                    else:
                        skills = ", ".join(s.get("name", s) if isinstance(s, dict) else str(s) for s in skills_data) or "—"
                    open_to_work = "  ✅ Open to Work" if p.get("openToWork") else ""
                    linkedin_url = p.get("linkedinUrl", "")

                    with st.container():
                        c_num, c_info, c_meta = st.columns([0.5, 3, 2.5])
                        c_num.markdown(f'<div class="rank-number">#{idx}</div>', unsafe_allow_html=True)
                        btn_html = ""
                        if linkedin_url:
                            btn_html = (
                                f'  \n<a href="{linkedin_url}" target="_blank" style="text-decoration:none;">'
                                f'<button style="padding:2px 10px;font-size:12px;border-radius:5px;'
                                f'border:1px solid #ccc;background:#f0f2f6;cursor:pointer;">🔗 LinkedIn Profile</button></a>'
                            )
                        c_info.markdown(
                            f"**{name}**{open_to_work}  \n{headline}{btn_html}",
                            unsafe_allow_html=True,
                        )
                        c_meta.caption(f"📍 {location}")
                        c_meta.caption(f"📧 {email}")
                        if skills != "—":
                            c_meta.caption(f"🛠 {skills}")
                        st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# MAIN TAB 2 — Requisition Form
# ════════════════════════════════════════════════════════════════════════════════
with main_tab_requisition:
    st.subheader("Requisition Form")
    st.caption("Describe the role in plain English — AI will extract the details and pre-fill the LinkedIn Sourcing form.")

    with st.form("requisition_form"):
        rq_prompt = st.text_area(
            "Describe the role",
            placeholder=(
                "e.g. I want to hire a Senior iOS Developer with 5+ years of experience in Swift and SwiftUI, "
                "who can join within 30 days. Source around 20 profiles from Delhi NCR."
            ),
            height=120,
        )
        rq_submit = st.form_submit_button("Analyse & Pre-fill LinkedIn Sourcing", type="primary")

    if rq_submit and rq_prompt.strip():
        with st.spinner("Analysing your prompt..."):
            oai_rq = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            rq_resp = oai_rq.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an HR assistant. Extract hiring details from the user's prompt and return JSON with exactly these keys:\n"
                            "- job_title: string (the role being hired for)\n"
                            "- specialisation: string (tech stack, domain, or skills focus)\n"
                            "- work_experience: string (years of experience required, e.g. '5+ years')\n"
                            "- notice_period: string (joining timeline preference, e.g. 'Immediate' or '30 days')\n"
                            "- max_profiles: integer (number of LinkedIn profiles to source, default 20)\n"
                            "If a detail is not mentioned, use a sensible default."
                        ),
                    },
                    {"role": "user", "content": rq_prompt.strip()},
                ],
            )
            extracted = json.loads(rq_resp.choices[0].message.content)

        st.session_state["rq_title"]  = extracted.get("job_title", "")
        st.session_state["rq_spec"]   = extracted.get("specialisation", "")
        st.session_state["rq_exp"]    = extracted.get("work_experience", "")
        st.session_state["rq_notice"] = extracted.get("notice_period", "")
        st.session_state["rq_max"]    = int(extracted.get("max_profiles", 20))

        st.success("Done! Fields extracted and saved.")
        st.markdown("**Extracted details:**")
        col_a, col_b = st.columns(2)
        col_a.markdown(f"- **Job Title:** {st.session_state['rq_title']}")
        col_a.markdown(f"- **Specialisation:** {st.session_state['rq_spec']}")
        col_a.markdown(f"- **Work Experience:** {st.session_state['rq_exp']}")
        col_b.markdown(f"- **Notice Period:** {st.session_state['rq_notice']}")
        col_b.markdown(f"- **Max Profiles:** {st.session_state['rq_max']}")
        st.info("Switch to **Resume Screening → LinkedIn Sourcing** tab — the form is pre-filled and ready to run.")
