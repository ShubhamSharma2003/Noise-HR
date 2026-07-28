import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hr_system.triage import (
    clamp_confidence,
    dedupe_by_applicant_id,
    needs_deep_screen,
    triage_summary_md,
)
from hr_system.freshteam import _resume_cache_path


def _result(recommendation="NO_FIT", confidence=0.1, error=None):
    r = {
        "final_state": {
            "agent_output": {
                "confidence_score": confidence,
                "raw_json": {"recommendation": recommendation},
            }
        }
    }
    if error:
        r["error"] = error
    return r


class TestNeedsDeepScreen:
    def test_top_categories_promoted(self):
        for rec in ("PERFECT_FIT", "STRONG_FIT", "GOOD_FIT", "MODERATE_FIT"):
            assert needs_deep_screen(_result(rec, 0.9))

    def test_generous_low_fit_above_threshold_promoted(self):
        # Deliberately generous: a LOW_FIT with non-trivial confidence still
        # gets the deep look (cost-no-object insurance against false negatives).
        assert needs_deep_screen(_result("LOW_FIT", 0.35))

    def test_clear_rejects_not_promoted(self):
        assert not needs_deep_screen(_result("NO_FIT", 0.05))
        assert not needs_deep_screen(_result("LOW_FIT", 0.15))

    def test_errors_never_promoted(self):
        assert not needs_deep_screen(_result("PERFECT_FIT", 0.95, error="boom"))

    def test_missing_fields_safe(self):
        assert not needs_deep_screen({"final_state": {}})
        assert not needs_deep_screen({})


class TestTriageSummary:
    def test_contains_all_sections(self):
        md = triage_summary_md({
            "reasoning": "Solid backend match.",
            "key_strengths": ["Go", "Kubernetes"],
            "concerns": ["No fintech background"],
            "dimension_scores": {"hard_skills": 8, "experience": 7},
        })
        assert "Solid backend match." in md
        assert "- Go" in md
        assert "- No fintech background" in md
        assert "Hard skills: 8" in md

    def test_empty_payload_still_renders(self):
        md = triage_summary_md({})
        assert "Triage screening" in md


class TestClampConfidence:
    def test_confidence_in_verdict_style_score_snapped_into_band(self):
        # NO_FIT with "98% sure it's a no" must not rank above real fits
        assert clamp_confidence("NO_FIT", 0.98) == 0.19

    def test_in_band_score_untouched(self):
        assert clamp_confidence("STRONG_FIT", 0.8) == 0.8

    def test_below_band_raised_to_floor(self):
        assert clamp_confidence("PERFECT_FIT", 0.2) == 0.9

    def test_unknown_category_passthrough(self):
        assert clamp_confidence("WHATEVER", 0.5) == 0.5

    def test_clamped_no_fit_never_promoted(self):
        clamped = clamp_confidence("NO_FIT", 0.98)
        result = {"final_state": {"agent_output": {
            "confidence_score": clamped, "raw_json": {"recommendation": "NO_FIT"}}}}
        assert not needs_deep_screen(result)


class TestDedupe:
    def test_last_write_wins_preserving_first_position(self):
        cards = [
            {"applicant_id": 1, "v": "old"},
            {"applicant_id": 2, "v": "b"},
            {"applicant_id": 1, "v": "new"},
        ]
        out = dedupe_by_applicant_id(cards)
        assert [c["applicant_id"] for c in out] == [1, 2]
        assert out[0]["v"] == "new"

    def test_empty(self):
        assert dedupe_by_applicant_id([]) == []


class TestResumeCachePath:
    def test_presigned_query_string_does_not_change_key(self):
        a = _resume_cache_path("42", "https://x.s3.amazonaws.com/r/1.pdf?X-Amz-Signature=aaa&X-Amz-Expires=300")
        b = _resume_cache_path("42", "https://x.s3.amazonaws.com/r/1.pdf?X-Amz-Signature=bbb&X-Amz-Expires=600")
        assert a == b

    def test_different_files_get_different_keys(self):
        a = _resume_cache_path("42", "https://x.s3.amazonaws.com/r/1.pdf")
        b = _resume_cache_path("42", "https://x.s3.amazonaws.com/r/2.pdf")
        assert a != b

    def test_key_includes_applicant_id(self):
        p = _resume_cache_path("42", "https://x.s3.amazonaws.com/r/1.pdf")
        assert "42-" in os.path.basename(p)
