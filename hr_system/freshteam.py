import io
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()


class FreshteamClient:
    """Thin wrapper around the Freshteam REST API."""

    def __init__(self):
        api_key = os.environ["FRESHTEAM_API_KEY"]
        subdomain = os.environ["FRESHTEAM_SUBDOMAIN"]
        self.base_url = f"https://{subdomain}.freshteam.com/api"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        response = requests.get(url, headers=self.headers, params=params or {})
        response.raise_for_status()
        return response

    def _paginate(self, path: str, params: dict = None) -> list[dict]:
        """Fetch all pages for a paginated endpoint and return combined results."""
        results = []
        page = 1
        while True:
            p = {**(params or {}), "page": page, "per_page": 100}
            response = self._get(path, p)
            data = response.json()
            if isinstance(data, list):
                results.extend(data)
                total_pages = int(response.headers.get("total-pages", 1))
            else:
                # Some endpoints wrap in a key — try common keys
                for key in ("job_postings", "applicants", "employees", "data"):
                    if key in data:
                        results.extend(data[key])
                        break
                total_pages = int(response.headers.get("total-pages", 1))

            if page >= total_pages:
                break
            page += 1
            time.sleep(0.1)  # stay under rate limit
        return results

    # ── Public methods ────────────────────────────────────────────────────────

    def get_job_postings(self, status: str = "published") -> list[dict]:
        """Return all job postings. Falls back to empty list if role lacks permission."""
        params = {"status": status} if status else {}
        try:
            return self._paginate("/job_postings", params)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                return []  # role can't list jobs — caller should use get_job_posting() instead
            raise

    def get_job_posting(self, job_id: int) -> dict:
        """Return a single job posting by ID."""
        try:
            response = self._get(f"/job_postings/{job_id}")
            return response.json()
        except requests.exceptions.HTTPError:
            return {"id": job_id}  # return minimal stub so callers don't crash

    def get_applicants(self, job_id: int) -> list[dict]:
        """Return all applicants for a specific job posting."""
        return self._paginate(f"/job_postings/{job_id}/applicants")

    def get_applicant(self, job_id: int, applicant_id: int) -> dict:
        """Return details for a single applicant, including resume URLs."""
        try:
            # /applicants/{id} returns full candidate data including resumes
            response = self._get(f"/applicants/{applicant_id}")
            return response.json()
        except requests.exceptions.HTTPError:
            try:
                response = self._get(f"/job_postings/{job_id}/applicants/{applicant_id}")
                return response.json()
            except requests.exceptions.HTTPError:
                for applicant in self._paginate(f"/job_postings/{job_id}/applicants"):
                    if applicant.get("id") == applicant_id:
                        return applicant
                return {"id": applicant_id}

    def get_employees(self, status: str = "active") -> list[dict]:
        """Return employees. status: 'active' | 'inactive' | None for all."""
        params = {"status": status} if status else {}
        try:
            return self._paginate("/employees", params)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                return []
            raise

    def _fetch_resume_text(self, resume_url: str) -> str:
        """Download a resume from Freshteam and return its text content."""
        try:
            # S3 pre-signed URLs are self-authenticating — don't send auth headers
            headers = {} if "s3.amazonaws.com" in resume_url else self.headers
            response = requests.get(resume_url, headers=headers, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            raw = response.content
            if "pdf" in content_type or resume_url.lower().endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                except Exception:
                    return ""
            # Plain text or other text-based formats
            return raw.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    # ── Convenience methods for task_input building ───────────────────────────

    def build_resume_screening_input(
        self, job_id: int, applicant_id: int
    ) -> dict:
        """
        Build the task_input dict for A1 (Resume Screener).
        Returns:
            {
                "job_id": int,
                "applicant_id": int,
                "job_title": str,
                "job_description": str,
                "resume_text": str,
                "applicant_name": str,
            }
        """
        job = self.get_job_posting(job_id)
        applicant = self.get_applicant(job_id, applicant_id)

        # Candidate info may be nested under "candidate" key (list endpoint)
        candidate = applicant.get("candidate") or applicant
        first = candidate.get("first_name", applicant.get("first_name", ""))
        last = candidate.get("last_name", applicant.get("last_name", ""))
        middle = candidate.get("middle_name", "")
        email = candidate.get("email", applicant.get("email", ""))
        mobile = candidate.get("mobile", "") or candidate.get("phone", "")

        # Try to get resume text from the uploaded resume file
        resumes = candidate.get("resumes") or []
        resume_url = resumes[0].get("url") if resumes else None
        resume_text = self._fetch_resume_text(resume_url) if resume_url else ""

        # Fall back to structured profile if no resume file is available
        if not resume_text:
            resume_text = (
                applicant.get("resume", "")
                or applicant.get("resume_text", "")
                or applicant.get("cover_letter", "")
                or "\n".join(filter(None, [
                    f"Name: {' '.join(filter(None, [first, middle, last]))}",
                    f"Email: {email}" if email else "",
                    f"Mobile: {mobile}" if mobile else "",
                    f"Current Stage: {applicant.get('stage', '')}",
                    f"Application Status: {applicant.get('status', '')}",
                ]))
            )

        return {
            "job_id": job_id,
            "applicant_id": applicant_id,
            "job_title": job.get("title", "Unknown Role"),
            "job_description": job.get("description", job.get("job_description", "")),
            "resume_text": resume_text,
            "applicant_name": " ".join(filter(None, [first, middle, last])).strip() or f"Applicant {applicant_id}",
        }

    def build_interview_scheduling_input(
        self, job_id: int, applicant_id: int, available_slots: list[str]
    ) -> dict:
        """
        Build the task_input dict for A2 (Interview Scheduler).
        Returns:
            {
                "job_id": int,
                "applicant_id": int,
                "candidate_name": str,
                "job_title": str,
                "available_slots": list[str],
                "interviewers": list[str],
            }
        """
        applicant = self.get_applicant(job_id, applicant_id)
        employees = self.get_employees(status="active")
        job = self.get_job_posting(job_id)

        candidate = applicant.get("candidate") or applicant
        first = candidate.get("first_name", applicant.get("first_name", ""))
        last = candidate.get("last_name", applicant.get("last_name", ""))

        # Use employees as potential interviewers
        interviewers = [
            f"{e.get('first_name', '')} {e.get('last_name', '')}".strip()
            for e in employees
            if e.get("first_name")
        ]

        return {
            "job_id": job_id,
            "applicant_id": applicant_id,
            "candidate_name": f"{first} {last}".strip() or f"Applicant {applicant_id}",
            "job_title": job.get("title", "Unknown Role"),
            "available_slots": available_slots,
            "interviewers": interviewers[:10],  # cap at 10 for prompt clarity
        }
