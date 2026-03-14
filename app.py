"""
HR Multi-Agent System — Streamlit UI
Run with: streamlit run app.py
"""
import os
import streamlit as st
from openai import OpenAI
from hr_system.freshteam import FreshteamClient
from hr_system.graph import hr_graph

_openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@st.cache_data(show_spinner=False)
def format_resume(raw_text: str) -> str:
    """Use GPT-4o to parse raw resume text into clean structured markdown."""
    if not raw_text or len(raw_text.strip()) < 20:
        return "_No resume content available._"
    response = _openai.chat.completions.create(
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

st.set_page_config(page_title="HR Agent System", page_icon="🧑‍💼", layout="wide")

st.markdown("""
<style>
.verdict-approve {
    background:#d4edda;color:#155724;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-reject {
    background:#f8d7da;color:#721c24;
    padding:6px 14px;border-radius:6px;
    font-weight:bold;display:inline-block;
}
.verdict-escalated {
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

@st.cache_data(show_spinner="Fetching jobs from Freshteam...")
def load_jobs():
    return FreshteamClient().get_job_postings()

@st.cache_data(show_spinner="Fetching applicants...", ttl=30)
def load_applicants(job_id):
    return FreshteamClient().get_applicants(job_id)

def applicant_label(a):
    c = a.get("candidate") or a
    name = f"{c.get('first_name','')} {c.get('last_name','')}".strip()
    return name or f"Applicant #{a['id']}"

def run_screening(job_id, applicant_id):
    client = FreshteamClient()
    task_input = client.build_resume_screening_input(job_id, applicant_id)
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

def verdict_badge(final_state):
    if final_state.get("escalated_to_cos"):
        return '<span class="verdict-escalated">⚠️ Escalated</span>'
    v = (final_state.get("manager_decision") or {}).get("verdict", "")
    if v == "APPROVE":
        return '<span class="verdict-approve">✅ Approved</span>'
    if v == "REJECT":
        return '<span class="verdict-reject">❌ Rejected</span>'
    return '<span class="verdict-escalated">⏳ Pending</span>'

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

# ── Job selector (shared across tabs) ────────────────────────────────────────
jobs = load_jobs()
if jobs:
    job_map = {f"{j.get('title','Untitled')} (#{j['id']})": j["id"] for j in jobs}
    job_label = st.selectbox("Job Posting", list(job_map.keys()))
    job_id = job_map[job_label]
else:
    st.warning("Could not load jobs from Freshteam. Select or add a Job ID below.")
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

applicants = load_applicants(job_id)
# Cache job title from applicant data so it shows in the selector
if applicants and "job_titles" in st.session_state:
    title = next((a.get("job_title") for a in applicants if a.get("job_title")), None)
    if title and st.session_state.job_titles.get(job_id) != title:
        st.session_state.job_titles[job_id] = title
        st.rerun()
if not applicants:
    st.warning("No applicants found for this job.")
    st.stop()

st.caption(f"{len(applicants)} applicant(s) found")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rank, tab_single = st.tabs(["📊 Rank All Applicants", "🔍 Single Applicant"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Rank All
# ════════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.subheader("Rank All Applicants by Resume Fit")
    st.caption("Screens every applicant and sorts them best-to-worst by AI confidence score.")

    if st.button("Screen & Rank All", type="primary", key="rank_all"):
        results = []
        progress = st.progress(0, text="Starting...")
        status_box = st.empty()

        for i, app in enumerate(applicants):
            name = applicant_label(app)
            aid = app["id"]
            status_box.info(f"Screening {i+1}/{len(applicants)}: **{name}**...")
            try:
                final_state, task_input = run_screening(job_id, aid)
                agent_out = final_state.get("agent_output") or {}
                confidence = agent_out.get("confidence_score", 0.0)
                results.append({
                    "name": name,
                    "applicant_id": aid,
                    "confidence": confidence,
                    "final_state": final_state,
                    "task_input": task_input,
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "applicant_id": aid,
                    "confidence": -1,
                    "error": str(e),
                    "final_state": {},
                    "task_input": {},
                })
            progress.progress((i + 1) / len(applicants), text=f"{i+1}/{len(applicants)} screened")

        status_box.empty()
        progress.empty()

        # Sort best → worst
        results.sort(key=lambda r: r["confidence"], reverse=True)

        st.success(f"Done! Ranked {len(results)} applicants.")
        st.divider()

        for rank, r in enumerate(results, 1):
            fs = r["final_state"]
            conf = r["confidence"]
            error = r.get("error")

            with st.container():
                col_rank, col_info, col_conf, col_badge = st.columns([0.5, 3, 2, 2])

                col_rank.markdown(f'<div class="rank-number">#{rank}</div>', unsafe_allow_html=True)
                col_info.markdown(f"**{r['name']}**  \nID: `{r['applicant_id']}`")

                if error:
                    col_conf.caption("Error")
                    col_badge.markdown('<span class="verdict-escalated">⚠️ Error</span>', unsafe_allow_html=True)
                else:
                    col_conf.progress(max(conf, 0), text=f"{conf:.0%} fit")
                    col_badge.markdown(verdict_badge(fs), unsafe_allow_html=True)

                with st.expander("View full result"):
                    if error:
                        st.error(error)
                    else:
                        res_tab, result_tab, audit_tab = st.tabs(["📄 Resume", "🤖 AI Result", "🕵️ Audit"])
                        with res_tab:
                            raw = r["task_input"].get("resume_text", "")
                            if raw:
                                with st.spinner("Formatting resume..."):
                                    st.markdown(format_resume(raw))
                            else:
                                st.caption("No resume file attached.")
                        with result_tab:
                            st.markdown(fs.get("final_result", "_(no result)_"))
                        with audit_tab:
                            render_audit(fs.get("history", []))

                st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Single Applicant
# ════════════════════════════════════════════════════════════════════════════════
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

        # Metrics row
        name = task_input.get("applicant_name") or task_input.get("candidate_name", "—")
        agent_out = final_state.get("agent_output") or {}
        conf = agent_out.get("confidence_score")

        c1, c2, c3 = st.columns(3)
        c1.metric("Candidate", name)
        c2.metric("Job", task_input.get("job_title", "—"))
        if conf is not None:
            c3.metric("AI Confidence", f"{conf:.0%}")

        st.divider()

        # Verdict
        st.markdown(verdict_badge(final_state), unsafe_allow_html=True)
        st.divider()

        # Result
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
