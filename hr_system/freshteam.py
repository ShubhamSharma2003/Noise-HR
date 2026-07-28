import io
import hashlib
import os
import time
import urllib.parse
import requests
from dotenv import load_dotenv

load_dotenv()

# Extracted resume text is cached on disk so re-scans, filters, and the deep
# screening pass don't re-download and re-parse PDFs from Freshteam.
RESUME_CACHE_DIR = os.environ.get("HR_RESUME_CACHE_DIR", ".cache/resume_texts")


def _resume_cache_path(cache_key: str, resume_url: str) -> str:
    # Hash only scheme+host+path: S3 pre-signed URLs get a fresh signature in
    # the query string on every listing, but the path identifies the file.
    parts = urllib.parse.urlsplit(resume_url)
    stable_url = f"{parts.scheme}://{parts.netloc}{parts.path}"
    digest = hashlib.sha256(stable_url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(RESUME_CACHE_DIR, f"{cache_key}-{digest}.txt")


def _resume_cache_get(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _resume_cache_put(path: str, text: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError:
        pass  # cache is best-effort


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

    def _get(self, path: str, params: dict = None, retries: int = 5) -> requests.Response:
        url = f"{self.base_url}{path}"
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=self.headers, params=params or {}, timeout=30)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Transient network hiccup (connection reset / timeout) — retry
                # with backoff instead of crashing the whole app.
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 30))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                wait = min(2 ** attempt, 30)  # 1, 2, 4, 8, 16 … capped at 30s
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait = max(wait, int(retry_after))
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        # Final attempt — let it raise
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
            time.sleep(0.5)  # stay under rate limit
        return results

    # ── Public methods ────────────────────────────────────────────────────────

    def get_job_postings(self) -> list[dict]:
        """Return all job postings. Falls back to empty list if role lacks permission."""
        try:
            return self._paginate("/job_postings")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (401, 403, 404):
                # 401 bad/expired key, 403 role lacks permission, 404 wrong subdomain —
                # degrade to empty list so the caller falls back to manual Job ID entry.
                print(f"[Freshteam] get_job_postings failed with HTTP {status} — check API key/subdomain")
                return []
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # Network is down/slow — degrade to manual Job ID entry instead of
            # crashing the app at startup.
            print(f"[Freshteam] get_job_postings network error: {e}")
            return []

    def get_job_posting(self, job_id: int) -> dict:
        """Return a single job posting by ID."""
        try:
            response = self._get(f"/job_postings/{job_id}")
            return response.json()
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            return {"id": job_id}  # return minimal stub so callers don't crash

    def get_applicants(self, job_id: int) -> list[dict]:
        """Return all applicants for a specific job posting."""
        try:
            return self._paginate(f"/job_postings/{job_id}/applicants")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (401, 403, 404):
                print(f"[Freshteam] get_applicants({job_id}) failed with HTTP {status} — check API key/subdomain")
                return []
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"[Freshteam] get_applicants({job_id}) network error: {e}")
            return []

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
            status = e.response.status_code if e.response is not None else None
            if status in (401, 403, 404):
                print(f"[Freshteam] get_employees failed with HTTP {status} — check API key/subdomain")
                return []
            raise

    def _fetch_resume_text(self, resume_url: str, cache_key: str = None) -> str:
        """Download a resume from Freshteam and return its text content.

        When cache_key is given (e.g. the applicant id), the extracted text is
        cached on disk and reused on later scans.
        """
        cache_path = _resume_cache_path(str(cache_key), resume_url) if cache_key else None
        if cache_path:
            cached = _resume_cache_get(cache_path)
            if cached:
                return cached
        text = self._download_resume_text(resume_url)
        if cache_path and text:
            _resume_cache_put(cache_path, text)
        return text

    def _download_resume_text(self, resume_url: str) -> str:
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
        self, job_id: int, applicant_id: int, job: dict = None
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
        # Bulk scans pass the job posting in once instead of re-fetching it
        # for every applicant.
        job = job or self.get_job_posting(job_id)
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
        resume_text = self._fetch_resume_text(resume_url, cache_key=applicant_id) if resume_url else ""

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
