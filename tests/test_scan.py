import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_system.scan import ScanJob, applicant_label


def _triage_card(aid, name="X", stage="triage"):
    return {"name": name, "applicant_id": aid, "confidence": 0.8,
            "final_state": {"agent_output": {"confidence_score": 0.8,
                                             "raw_json": {"recommendation": "STRONG_FIT"}}},
            "task_input": {"applicant_name": name}, "stage": stage}


class TestApplicantLabel:
    def test_prefers_candidate_name(self):
        assert applicant_label({"candidate": {"first_name": "Ada", "last_name": "L"}}) == "Ada L"

    def test_falls_back_to_id(self):
        assert applicant_label({"id": 7}) == "Applicant #7"


class TestScanJobMerge:
    def _job(self):
        job = ScanJob(job_id=1, applicants=[{"id": 1}, {"id": 2}, {"id": 3}], workers=5)
        # Simulate a completed triage stage + partial deep stage without threads.
        job._triage = {1: _triage_card(1, "A"), 2: _triage_card(2, "B"), 3: _triage_card(3, "C")}
        job._order = [1, 2, 3]
        job._deep = {1: _triage_card(1, "A", stage="deep")}
        job.survivors = [job._triage[1], job._triage[2]]
        job.stage = "done"
        return job

    def test_snapshot_counts(self):
        snap = self._job().snapshot()
        assert snap["total"] == 3
        assert snap["triage_done"] == 3
        assert snap["deep_total"] == 2
        assert snap["deep_done"] == 1
        assert snap["stage"] == "done"

    def test_merged_prefers_deep_then_triage(self):
        merged = {c["applicant_id"]: c for c in self._job().merged_results()}
        assert merged[1]["stage"] == "deep"    # deep result wins
        assert merged[2]["stage"] == "triage"  # survivor not yet deep-screened
        assert merged[3]["stage"] == "triage"  # never promoted
        assert len(merged) == 3                 # no dropped or duplicated candidate

    def test_recent_cards_returns_latest(self):
        cards = self._job().recent_cards(2)
        assert [c["applicant_id"] for c in cards] == [2, 3]

    def test_empty_job_snapshot(self):
        job = ScanJob(job_id=1, applicants=[], workers=5)
        snap = job.snapshot()
        assert snap["total"] == 0 and snap["triage_done"] == 0
        assert job.merged_results() == []


class TestScanJobInit:
    def test_dedupes_applicants_by_id(self):
        job = ScanJob(job_id=1, applicants=[{"id": 1}, {"id": 2}, {"id": 1}], workers=5)
        assert job.total == 2
        assert [a["id"] for a in job.applicants] == [1, 2]

    def test_keeps_records_without_id(self):
        job = ScanJob(job_id=1, applicants=[{"id": 1}, {"name": "no-id"}], workers=5)
        assert job.total == 2

    def test_workers_floored_at_one(self):
        job = ScanJob(job_id=1, applicants=[{"id": 1}], workers=0)
        assert job.workers == 1 and job.deep_workers == 1


class TestTriageOneDefensive:
    def test_missing_id_becomes_error_card_not_crash(self):
        job = ScanJob(job_id=1, applicants=[{"name": "x"}], workers=1)
        card = job._triage_one({"name": "x"}, client=None, job_detail={})
        assert card["applicant_id"] is None
        assert card["error"] and card["confidence"] == -1

    def test_cancelled_worker_returns_none(self):
        job = ScanJob(job_id=1, applicants=[{"id": 1}], workers=1)
        job.cancel()
        assert job._triage_one({"id": 1}, client=None, job_detail={}) is None
        assert job._deep_one({"applicant_id": 1, "name": "x", "task_input": {}}) is None

    def test_deep_failed_counter_in_snapshot(self):
        job = ScanJob(job_id=1, applicants=[{"id": 1}], workers=1)
        job._deep = {1: {"applicant_id": 1, "deep_error": "boom", "stage": "triage"}}
        assert job.snapshot()["deep_failed"] == 1
