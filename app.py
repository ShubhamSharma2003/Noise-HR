"""
HR Multi-Agent System — Streamlit UI
Run with: streamlit run app.py
"""
import os
import json
import requests as _requests
import streamlit as st
from openai import OpenAI
from hr_system.freshteam import FreshteamClient
from hr_system.graph import hr_graph

# ── Page config ───────────────────────────────────────────────────────────────
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

@st.cache_data(show_spinner=False)
def format_resume(raw_text: str) -> str:
    if not raw_text or len(raw_text.strip()) < 20:
        return "_No resume content available._"
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

@st.cache_data(show_spinner="Fetching jobs from Freshteam...", ttl=60)
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
        st.warning("Could not load jobs from Freshteam. Enter a Job ID manually.")
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
        # ── Search bar ────────────────────────────────────────────────────────
        job_search = st.text_input(
            "job_search_label",
            placeholder="🔍  Search job postings…",
            label_visibility="collapsed",
            key="job_search_query",
        )

        query = job_search.strip().lower()
        filtered_jobs = [
            j for j in jobs
            if query in j.get("title", "").lower() or query in str(j["id"])
        ] if query else jobs

        # ── Pagination state ──────────────────────────────────────────────────
        JOB_CARDS_PER_PAGE = 6
        total_job_pages = max(1, (len(filtered_jobs) + JOB_CARDS_PER_PAGE - 1) // JOB_CARDS_PER_PAGE)

        if "job_page" not in st.session_state or st.session_state.get("_last_job_query") != query:
            st.session_state.job_page = 1
            st.session_state["_last_job_query"] = query

        job_page = st.session_state.job_page
        j_start = (job_page - 1) * JOB_CARDS_PER_PAGE
        j_end = min(j_start + JOB_CARDS_PER_PAGE, len(filtered_jobs))
        page_jobs = filtered_jobs[j_start:j_end]

        if not filtered_jobs:
            st.warning("No job postings match your search.")
            st.stop()

        # ── Initialise selected job ───────────────────────────────────────────
        if "selected_job_id" not in st.session_state:
            st.session_state.selected_job_id = jobs[0]["id"]

        # ── Job cards grid (3 columns) ────────────────────────────────────────
        JOB_COLS = 3
        for row_start in range(0, len(page_jobs), JOB_COLS):
            row_jobs = page_jobs[row_start:row_start + JOB_COLS]
            cols = st.columns(JOB_COLS)
            for col, j in zip(cols, row_jobs):
                jid = j["id"]
                title = j.get("title", "Untitled")
                status = j.get("status", "")
                dept = (j.get("department") or {}).get("name", "") if isinstance(j.get("department"), dict) else ""
                location = ""
                for loc in (j.get("branch_id_list") or j.get("locations") or []):
                    if isinstance(loc, dict):
                        location = loc.get("name", "")
                        break
                is_selected = st.session_state.selected_job_id == jid
                border_color = "#4f8ef7" if is_selected else "#dee2e6"
                bg_color = "#f0f5ff" if is_selected else "#ffffff"
                with col:
                    st.markdown(
                        f"""<div style="border:2px solid {border_color};border-radius:10px;
                        padding:14px 16px;background:{bg_color};min-height:110px;margin-bottom:4px;">
                        <div style="font-weight:700;font-size:0.92rem;margin-bottom:4px;">{title}</div>
                        <div style="font-size:0.75rem;color:#6c757d;">ID: {jid}</div>
                        {"<div style='font-size:0.75rem;color:#6c757d;'>🏢 " + dept + "</div>" if dept else ""}                        
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    btn_label = "✔ Selected" if is_selected else "Select"
                    if col.button(btn_label, key=f"select_job_{jid}", use_container_width=True, disabled=is_selected):
                        st.session_state.selected_job_id = jid
                        st.rerun()

        # ── Pagination controls ───────────────────────────────────────────────
        if total_job_pages > 1:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            pc1, pc2, pc3, pc4, pc5 = st.columns([1, 0.4, 0.5, 0.4, 1])
            if pc2.button("◀", key="job_prev", disabled=job_page == 1):
                st.session_state.job_page = job_page - 1
                st.rerun()
            pc3.markdown(
                f"<div style='text-align:center;padding-top:6px;font-size:0.85rem'>{job_page}/{total_job_pages}</div>",
                unsafe_allow_html=True,
            )
            if pc4.button("▶", key="job_next", disabled=job_page == total_job_pages):
                st.session_state.job_page = job_page + 1
                st.rerun()

        job_id = st.session_state.selected_job_id
        selected_job_title = next((j.get("title", "") for j in jobs if j["id"] == job_id), "")
        st.markdown(
            f"<div style='margin-top:8px;font-size:0.85rem;color:#4f8ef7;font-weight:600;'>"
            f"Selected: {selected_job_title} (#{job_id})</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    applicants = load_applicants(job_id)
    if not applicants:
        st.warning("No applicants found for this job.")
        st.stop()

    st.caption(f"{len(applicants)} applicant(s) found")

    # ── Applicant cards with search + pagination ──────────────────────────────
    app_search = st.text_input(
        "app_search_label",
        placeholder="🔍  Search applicants by name, email, or stage…",
        label_visibility="collapsed",
        key=f"app_search_{job_id}",
    )
    app_query = app_search.strip().lower()

    def _app_matches(app, q):
        c = app.get("candidate") or app
        name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip().lower()
        email = (c.get("email") or "").lower()
        stage = app.get("stage", {})
        stage_name = (stage.get("name") if isinstance(stage, dict) else stage or "").lower()
        return q in name or q in email or q in stage_name or q in str(app["id"])

    filtered_applicants = [a for a in applicants if _app_matches(a, app_query)] if app_query else applicants

    if app_query:
        st.caption(f"{len(filtered_applicants)} of {len(applicants)} applicant(s) matched")

    CARDS_PER_PAGE = 8
    total_pages = max(1, (len(filtered_applicants) + CARDS_PER_PAGE - 1) // CARDS_PER_PAGE)
    page_key = f"applicant_page_{job_id}"
    # Reset to page 1 if search query changed
    if st.session_state.get(f"_last_app_query_{job_id}") != app_query:
        st.session_state[page_key] = 1
        st.session_state[f"_last_app_query_{job_id}"] = app_query
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    current_page = st.session_state[page_key]
    page_start = (current_page - 1) * CARDS_PER_PAGE
    page_end = min(page_start + CARDS_PER_PAGE, len(filtered_applicants))
    page_applicants = filtered_applicants[page_start:page_end]

    cols_per_row = 4
    for row_start in range(0, len(page_applicants), cols_per_row):
        row_apps = page_applicants[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, app in zip(cols, row_apps):
            c = app.get("candidate") or app
            name = f"{c.get('first_name', '')} {c.get('last_name', '')}".strip() or f"Applicant #{app['id']}"
            email = c.get("email", "")
            stage = app.get("stage", {})
            stage_name = (stage.get("name") if isinstance(stage, dict) else stage) or "—"
            source = app.get("source") or "—"
            ft_url = f"https://gonoise.freshteam.com/hire/jobs/{job_id}/applicants/listview/{app['id']}"
            with col:
                st.markdown(
                    f"""<div style="border:1px solid #dee2e6;border-radius:10px;padding:14px 16px;
                    background:#fff;min-height:130px;">
                    <div style="font-weight:700;font-size:0.95rem;margin-bottom:4px;">{name}</div>
                    <div style="font-size:0.78rem;color:#6c757d;">ID: {app['id']}</div>
                    {"<div style='font-size:0.78rem;color:#6c757d;'>📧 " + email + "</div>" if email else ""}
                    <div style="font-size:0.78rem;color:#6c757d;">📌 {stage_name}</div>
                    <div style="font-size:0.78rem;color:#6c757d;">🔎 {source}</div>
                    <a href="{ft_url}" target="_blank" style="text-decoration:none;">
                    <button style="margin-top:8px;padding:2px 10px;font-size:11px;border-radius:5px;
                    border:1px solid #ccc;background:#f0f2f6;cursor:pointer;">🔗 Profile</button></a>
                    </div>""",
                    unsafe_allow_html=True,
                )

    if total_pages > 1:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        p_cols = st.columns([1, 0.4, 0.4, 0.4, 1])
        if p_cols[1].button("◀", key=f"prev_page_{job_id}", disabled=current_page == 1):
            st.session_state[page_key] = current_page - 1
            st.rerun()
        p_cols[2].markdown(
            f"<div style='text-align:center;padding-top:6px;font-size:0.85rem'>{current_page}/{total_pages}</div>",
            unsafe_allow_html=True,
        )
        if p_cols[3].button("▶", key=f"next_page_{job_id}", disabled=current_page == total_pages):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    st.divider()

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    tab_rank, tab_single, tab_linkedin = st.tabs(["📊 Rank All Applicants", "🔍 Single Applicant", "🔗 LinkedIn Sourcing"])

    # ── Sub-tab 1: Rank All ───────────────────────────────────────────────────
    with tab_rank:
        st.subheader("Rank All Applicants by Resume Fit")
        st.caption("Screens every applicant and sorts them best-to-worst by AI confidence score.")

        ranked_key    = f"ranked_results_{job_id}"
        scanning_key  = f"scanning_{job_id}"
        scan_buf_key  = f"scan_buf_{job_id}"

        # Clear stale keys from other jobs
        for k in list(st.session_state.keys()):
            if any(k.startswith(p) for p in ("ranked_results_", "scanning_", "scan_buf_")) \
                    and not k.endswith(f"_{job_id}"):
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
                col_info.markdown(
                    f"**{r['name']}**  \nID: `{r['applicant_id']}` &nbsp;"
                    f'<a href="{ft_url}" target="_blank" style="text-decoration:none;">'
                    f'<button style="padding:2px 10px;font-size:12px;border-radius:5px;'
                    f'border:1px solid #ccc;background:#f0f2f6;cursor:pointer;">'
                    f'🔗 Freshteam Profile</button></a>',
                    unsafe_allow_html=True,
                )
                if err:
                    col_conf.caption("Error")
                    col_badge.markdown('<span class="verdict-escalated">⚠️ Error</span>', unsafe_allow_html=True)
                else:
                    col_conf.progress(max(conf, 0), text=f"{conf:.0%} fit")
                    col_badge.markdown(verdict_badge(fs), unsafe_allow_html=True)
                if not live:
                    with st.expander("View full result"):
                        if err:
                            st.error(err)
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

        # ── Button row ────────────────────────────────────────────────────────
        btn_col, stop_col = st.columns([2, 1])
        if btn_col.button("Screen & Rank All", type="primary", key="rank_all",
                          disabled=st.session_state.get(scanning_key, False)):
            st.session_state[scanning_key] = True
            st.session_state[scan_buf_key] = []
            st.session_state.pop(ranked_key, None)
            st.session_state.pop("kw_rank_active", None)
            st.session_state.pop("kw_rank_matched", None)
            st.rerun()

        if stop_col.button("⏹ Stop", key="rank_stop",
                           disabled=not st.session_state.get(scanning_key, False)):
            st.session_state[scanning_key] = False
            scanned = st.session_state.get(scan_buf_key, [])
            if scanned:
                scanned.sort(key=lambda r: r["confidence"], reverse=True)
                st.session_state[ranked_key] = scanned
            st.rerun()

        # ── Live scanning: one profile per rerun ──────────────────────────────
        if st.session_state.get(scanning_key, False):
            scanned = st.session_state[scan_buf_key]
            idx = len(scanned)
            total = len(applicants)

            st.progress(idx / total, text=f"Scanning {idx}/{total}…")

            # Render already-scanned cards (live mode — no expander yet)
            for i, r in enumerate(scanned, 1):
                _render_rank_card(r, i, live=True)

            if idx < total:
                app  = applicants[idx]
                name = applicant_label(app)
                aid  = app["id"]
                with st.spinner(f"Scanning **{name}** ({idx+1}/{total})…"):
                    try:
                        final_state, task_input = run_screening(job_id, aid)
                        agent_out  = final_state.get("agent_output") or {}
                        confidence = agent_out.get("confidence_score", 0.0)
                        scanned.append({
                            "name": name, "applicant_id": aid,
                            "confidence": confidence,
                            "final_state": final_state, "task_input": task_input,
                        })
                    except Exception as e:
                        scanned.append({
                            "name": name, "applicant_id": aid,
                            "confidence": -1, "error": str(e),
                            "final_state": {}, "task_input": {},
                        })
                st.session_state[scan_buf_key] = scanned
                st.rerun()
            else:
                # All done — sort and persist
                scanned.sort(key=lambda r: r["confidence"], reverse=True)
                st.session_state[ranked_key] = scanned
                st.session_state[scanning_key] = False
                st.rerun()

        # ── Final ranked results (scan complete) ──────────────────────────────
        if ranked_key in st.session_state and not st.session_state.get(scanning_key, False):
            results = st.session_state[ranked_key]
            st.success(f"Done! Ranked {len(results)} applicants.")
            st.divider()

            kw_col, btn_col, clear_col = st.columns([4, 1.2, 1])
            kw_input = kw_col.text_input(
                "kw_rank_label",
                placeholder="Filter by keywords (comma-separated) — e.g. Chandigarh University, entrepreneur, AWS",
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
                keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
                matched = []
                oai_kw = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                prog_kw = st.progress(0, text="Filtering...")
                for i, r in enumerate(results):
                    resume_text = r.get("task_input", {}).get("resume_text", "") or r.get("name", "")
                    resume_normalized = " ".join(resume_text.split()).lower()
                    all_matched = True
                    for kw in keywords:
                        kw_lower = kw.lower()
                        kw_found = False
                        if kw_lower in resume_normalized:
                            kw_found = True
                        else:
                            try:
                                resp = oai_kw.chat.completions.create(
                                    model="gpt-4o-mini",
                                    max_tokens=10,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": (
                                                "You are a smart candidate filter. Given a keyword/phrase and a resume, "
                                                "reply ONLY 'yes' or 'no': does this resume match the keyword concept?\n"
                                                "Be semantically intelligent:\n"
                                                "- 'entrepreneur' → co-founded, startup founder, own business\n"
                                                "- 'tier 1 college' → IIT, IIM, IISc, NIT, top-ranked university\n"
                                                "- 'fintech' → payments, banking tech, neo-bank, lending\n"
                                                "Match intent, not just exact words."
                                            ),
                                        },
                                        {"role": "user", "content": f"Keyword: {kw}\n\nResume:\n{resume_text[:3000]}"},
                                    ],
                                )
                                if resp.choices[0].message.content.strip().lower().startswith("yes"):
                                    kw_found = True
                            except Exception:
                                kw_found = True
                        if not kw_found:
                            all_matched = False
                            break
                    if all_matched:
                        matched.append(r["applicant_id"])
                    prog_kw.progress((i + 1) / len(results))
                prog_kw.empty()
                st.session_state["kw_rank_active"] = kw_input.strip()
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

            st.divider()
            for rank, r in enumerate(display_results, 1):
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
                    skills = ", ".join(p.get("topSkills") or []) or "—"
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
