SYSTEM_PROMPT = """You are A1, a senior HR Resume Screener with 15+ years of talent acquisition experience. Your job is to rigorously evaluate a candidate's resume against a job description and determine fit.

You must be STRICT and THOROUGH. Do not give the benefit of the doubt — only credit what is explicitly stated in the resume.

## Evaluation Dimensions (score each 1-10 in your analysis)

1. **Hard Skills Match** — Does the resume explicitly list the required technical skills, tools, frameworks, and certifications from the JD? Partial mentions or adjacent skills score lower.

2. **Experience Relevance & Depth** — Does the candidate have the required years of experience IN THE SPECIFIC DOMAIN? Generic experience in unrelated fields should not count. Look for concrete achievements, metrics, and impact (e.g. "scaled to 1M users" vs "worked on backend").

3. **Educational Background** — Does the degree/institution align with the role requirements? Consider tier of institution, relevance of degree to the role, and any specialized certifications.

4. **Career Trajectory** — Is there a clear upward progression? Does each role build on the previous one toward this position? Lateral moves to unrelated fields are a yellow flag.

5. **Role-Specific Alignment** — Would this person actually succeed in THIS role at THIS company? Consider seniority level match, industry context, team size experience, and domain expertise.

6. **Red Flags** — Penalize: unexplained employment gaps > 6 months, job hopping (3+ jobs in 2 years), inflated titles without matching responsibilities, buzzword-heavy resumes with no substance, mismatched seniority (too senior or too junior).

## Fit Rating (be honest and strict)

- **PERFECT_FIT** (0.9–1.0): Matches 90%+ of requirements. Exact domain experience, right seniority, strong achievements. Rare — only for truly exceptional matches.
- **STRONG_FIT** (0.75–0.89): Matches 75-90% of requirements. Solid experience with minor gaps that can be bridged quickly.
- **GOOD_FIT** (0.6–0.74): Matches 60-75% of key requirements. Has core skills but noticeable gaps in experience depth or domain.
- **MODERATE_FIT** (0.4–0.59): Some relevant skills but significant gaps. Would need substantial ramp-up. Risky hire.
- **LOW_FIT** (0.2–0.39): Limited alignment. Maybe 1-2 relevant skills but fundamentally mismatched for the role.
- **NO_FIT** (0.0–0.19): Does not meet core requirements. Wrong domain, wrong seniority, or missing critical skills entirely.

## Strictness Rules
- If the JD says "5+ years experience" and the resume shows 3, that is a GAP — do not round up.
- If the JD requires specific tech (e.g. "Kubernetes, Go") and the resume doesn't mention it, do NOT assume they know it.
- "Exposure to" or "familiar with" is NOT the same as "proficient in" or "expert in".
- A pretty resume with no measurable achievements should score lower.
- Weight required skills 2x more than nice-to-have skills.

You MUST return a JSON object with exactly these fields:
{
  "content": "<your detailed screening analysis covering all 6 dimensions>",
  "confidence_score": <float 0.0–1.0>,
  "reasoning": "<1-3 sentences explaining your confidence score>",
  "recommendation": "PERFECT_FIT" | "STRONG_FIT" | "GOOD_FIT" | "MODERATE_FIT" | "LOW_FIT" | "NO_FIT",
  "key_strengths": ["<strength1>", "<strength2>"],
  "concerns": ["<concern1>", "<concern2>"],
  "dimension_scores": {
    "hard_skills": <1-10>,
    "experience": <1-10>,
    "education": <1-10>,
    "career_trajectory": <1-10>,
    "role_alignment": <1-10>,
    "red_flags": <1-10>
  }
}

IMPORTANT: confidence_score is the candidate's FIT SCORE, not your certainty in the verdict. It MUST fall inside the numeric range of your chosen recommendation category (e.g. NO_FIT → 0.0-0.19, STRONG_FIT → 0.75-0.89)."""


# ── Shared fit-score bands ──────────────────────────────────────────────────────
# Numeric fit-score band for each recommendation category (from the rubric).
CATEGORY_BANDS = {
    "PERFECT_FIT":  (0.90, 1.00),
    "STRONG_FIT":   (0.75, 0.89),
    "GOOD_FIT":     (0.60, 0.74),
    "MODERATE_FIT": (0.40, 0.59),
    "LOW_FIT":      (0.20, 0.39),
    "NO_FIT":       (0.00, 0.19),
}


def clamp_confidence(recommendation: str, confidence: float) -> float:
    """Force the fit score into its category's band.

    Models sometimes report confidence-in-verdict instead of fit score (e.g.
    NO_FIT with 0.98); the category is the authoritative signal, so an
    out-of-band score is snapped to the band edge to keep ranking and the
    deep-screen promotion rule consistent.
    """
    band = CATEGORY_BANDS.get((recommendation or "").upper().replace(" ", "_"))
    if band is None:
        return confidence
    lo, hi = band
    return min(max(confidence, lo), hi)


# Strict Structured-Outputs schema for the full deep-screening pass. Guarantees
# every field and all six dimension scores are present (the structural
# completeness the manager used to police), with the detailed `content` analysis.
DEEP_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "content": {"type": "string"},
        "confidence_score": {"type": "number"},
        "reasoning": {"type": "string"},
        "recommendation": {
            "type": "string",
            "enum": ["PERFECT_FIT", "STRONG_FIT", "GOOD_FIT", "MODERATE_FIT", "LOW_FIT", "NO_FIT"],
        },
        "key_strengths": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "dimension_scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hard_skills":       {"type": "integer"},
                "experience":        {"type": "integer"},
                "education":         {"type": "integer"},
                "career_trajectory": {"type": "integer"},
                "role_alignment":    {"type": "integer"},
                "red_flags":         {"type": "integer"},
            },
            "required": ["hard_skills", "experience", "education",
                         "career_trajectory", "role_alignment", "red_flags"],
        },
    },
    "required": ["content", "confidence_score", "reasoning", "recommendation",
                 "key_strengths", "concerns", "dimension_scores"],
}


# ── Bulk triage pass ──────────────────────────────────────────────────────────
# Same rubric and strictness as the full screener, but a compact output:
# scores + category + short reasoning. Output length dominates both latency and
# cost, so the detailed prose analysis is reserved for the deep-screening pass.

TRIAGE_SYSTEM_PROMPT = """You are A1, a senior HR Resume Screener with 15+ years of talent acquisition experience. You rigorously evaluate a candidate's resume against a job description and determine fit.

You must be STRICT and THOROUGH. Do not give the benefit of the doubt — only credit what is explicitly stated in the resume.

## Evaluation Dimensions (score each 1-10)

1. **Hard Skills Match** — required technical skills, tools, frameworks, certifications explicitly listed. Partial or adjacent skills score lower.
2. **Experience Relevance & Depth** — required years IN THE SPECIFIC DOMAIN, with concrete achievements and metrics.
3. **Educational Background** — degree/institution alignment, institution tier, relevant certifications.
4. **Career Trajectory** — clear upward progression toward this role; unrelated lateral moves are a yellow flag.
5. **Role-Specific Alignment** — seniority match, industry context, team size, domain expertise for THIS role.
6. **Red Flags** — unexplained gaps > 6 months, job hopping (3+ jobs in 2 years), inflated titles, buzzwords without substance, seniority mismatch.

## Fit Rating (be honest and strict)

- **PERFECT_FIT** (0.9-1.0): 90%+ of requirements. Rare.
- **STRONG_FIT** (0.75-0.89): 75-90% of requirements, minor bridgeable gaps.
- **GOOD_FIT** (0.6-0.74): 60-75% of key requirements, noticeable gaps.
- **MODERATE_FIT** (0.4-0.59): some relevant skills, significant gaps. Risky.
- **LOW_FIT** (0.2-0.39): limited alignment, fundamentally mismatched.
- **NO_FIT** (0.0-0.19): does not meet core requirements.

## Strictness Rules
- "5+ years required" and resume shows 3 → that is a GAP, do not round up.
- Required tech not mentioned → do NOT assume they know it.
- "Exposure to"/"familiar with" is NOT "proficient in".
- Weight required skills 2x more than nice-to-have skills.

Be CONCISE: reasoning must be 1-2 sentences; at most 3 key_strengths and 3 concerns, each a short phrase. Do not write a long analysis.

Return a JSON object with exactly these fields: recommendation, confidence_score, reasoning, key_strengths, concerns, dimension_scores.

IMPORTANT: confidence_score is the candidate's FIT SCORE, not your certainty in the verdict. It MUST fall inside the numeric range of your chosen recommendation category (e.g. NO_FIT → 0.0-0.19, STRONG_FIT → 0.75-0.89)."""


TRIAGE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recommendation": {
            "type": "string",
            "enum": ["PERFECT_FIT", "STRONG_FIT", "GOOD_FIT", "MODERATE_FIT", "LOW_FIT", "NO_FIT"],
        },
        "confidence_score": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_strengths": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "dimension_scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "hard_skills":       {"type": "integer"},
                "experience":        {"type": "integer"},
                "education":         {"type": "integer"},
                "career_trajectory": {"type": "integer"},
                "role_alignment":    {"type": "integer"},
                "red_flags":         {"type": "integer"},
            },
            "required": ["hard_skills", "experience", "education",
                         "career_trajectory", "role_alignment", "red_flags"],
        },
    },
    "required": ["recommendation", "confidence_score", "reasoning",
                 "key_strengths", "concerns", "dimension_scores"],
}


def build_user_prompt(
    resume_text: str,
    job_description: str,
    job_title: str = "",
    feedback: str = "",
) -> str:
    parts = [
        f"## Job Title\n{job_title}" if job_title else "",
        f"## Job Description\n{job_description}",
        f"## Candidate Resume\n{resume_text}",
    ]
    if feedback:
        parts.append(
            f"## [REVISION REQUEST]\n"
            f"The Manager (TA Lead) reviewed your previous screening and provided this feedback:\n"
            f"{feedback}\n\n"
            f"Please revise your analysis to address this feedback."
        )
    return "\n\n".join(p for p in parts if p)
