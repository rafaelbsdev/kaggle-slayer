"""Thin wrapper over the official Kaggle CLI. All network access to Kaggle goes through here."""
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kagglesdk.kaggle_client import KaggleClient as SdkClient
from kagglesdk.kaggle_creds import KaggleCredentials

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{1,100}$")
TRANSIENT = re.compile(r"\b(429|500|502|503|504)\b|timed out|temporarily|connection", re.I)
UNAUTHENTICATED = re.compile(r"\b401\b|unauthorized|credentials|authentication required", re.I)
RULES = re.compile(r"\b403\b|forbidden|accept.*rules", re.I)
SUBMISSION_REF = re.compile(r"Submission ref:\s*(\d+)")

PAGE_ORDER = ("abstract", "description", "evaluation", "data-description")

READ_TIMEOUT = 120
DOWNLOAD_TIMEOUT = 7200
SCORING_WAIT = 900

CREDENTIALS_FILE = Path(KaggleCredentials.DEFAULT_CREDENTIALS_FILE).expanduser()
TOKEN_MARGIN = timedelta(minutes=30)


class KaggleError(RuntimeError):
    def __init__(self, message: str, output: str = ""):
        super().__init__(message)
        self.output = output  # what the CLI printed before failing


class NotAuthenticated(KaggleError):
    pass


class RulesNotAccepted(KaggleError):
    pass


def _default_exe() -> str:
    return shutil.which("kaggle", path=str(Path(sys.executable).parent)) or "kaggle"


def _token_is_expiring() -> bool:
    try:
        expiry = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))["access_token_expiration"]
        return datetime.fromisoformat(expiry) - datetime.now(timezone.utc) < TOKEN_MARGIN
    except (OSError, ValueError, KeyError, TypeError):
        return False  # no cached login (API key, token in the environment), or one the CLI will complain about


def renew_expiring_token() -> None:
    """Renews the cached login before it runs out, so a long run survives it.

    The SDK renews only 30 minutes after expiry and sends the dead token until then, which stopped a
    real run. A failed renewal is left to the CLI: its error is the one we know how to classify.
    """
    if not _token_is_expiring():
        return
    try:
        with SdkClient() as sdk:
            credentials = KaggleCredentials.load(client=sdk, file_path=str(CREDENTIALS_FILE))
            if credentials:
                credentials.refresh_access_token()
    except OSError:  # requests' errors are OSErrors too
        pass


def _csv_rows(text: str, first_column: str) -> list[dict]:
    """The CLI may print warnings before the CSV, so the table starts at its header."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{first_column},")), None)
    if start is None:
        return []
    return list(csv.DictReader(lines[start:]))


def _score(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class KaggleClient:
    def __init__(self, runner=subprocess.run, exe: str | None = None, retries: int = 4, sleep=time.sleep):
        self.runner, self.retries, self.sleep = runner, retries, sleep
        self.exe = exe or _default_exe()

    def _run(self, slug: str, args: list[str], timeout: int = READ_TIMEOUT, retries: int | None = None) -> str:
        if not SLUG.match(slug):
            raise ValueError(f"invalid competition slug: {slug!r}")
        renew_expiring_token()
        attempts = self.retries if retries is None else retries
        env = {**os.environ, "PYTHONUTF8": "1"}
        last_error, last_output = "no attempts made", ""
        for attempt in range(attempts):
            try:
                process = self.runner([self.exe, *args], capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                last_error = f"kaggle {' '.join(args[:2])} timed out after {timeout}s"
            else:
                if process.returncode == 0:
                    return process.stdout or ""
                last_output = process.stdout or ""
                last_error = f"{process.stderr or ''}\n{last_output}".strip()
                if UNAUTHENTICATED.search(last_error):
                    raise NotAuthenticated("Kaggle rejected the credentials. Run `kaggle auth login`.")
                if RULES.search(last_error):
                    raise RulesNotAccepted(
                        f"Kaggle refused access to {slug}. Accept its rules on the competition page: "
                        f"https://www.kaggle.com/competitions/{slug}/rules")
                if not TRANSIENT.search(last_error):
                    raise KaggleError(last_error[:500], output=last_output)
            if attempt + 1 < attempts:
                self.sleep(5 * 2 ** attempt)
        raise KaggleError(last_error[:500], output=last_output)

    def info(self, slug: str) -> dict:
        output = self._run(slug, ["competitions", "list", "-s", slug, "--page-size", "50", "-v"])
        for row in _csv_rows(output, "ref"):
            if row["ref"].rstrip("/").split("/")[-1] == slug:
                rank = int(row["userRank"]) if row.get("userRank", "").strip().isdigit() else 0
                return {
                    "entered": row.get("userHasEntered", "").strip().lower() == "true",
                    "team_count": int(row["teamCount"]),
                    "user_rank": rank or None,  # Kaggle reports rank 0 before the first submission
                    "category": row.get("category"),
                    "deadline": row.get("deadline"),
                }
        raise KaggleError(f"competition {slug} not found")

    def overview(self, slug: str) -> str:
        """The competition's text pages, useful ones first and the legal pages left out."""
        output = self._run(slug, ["competitions", "pages", "list", "-c", slug, "--content", "--format", "json"])
        start = output.find("[")  # the CLI may print warnings before the JSON
        try:
            pages = json.loads(output[start:]) if start >= 0 else None
        except json.JSONDecodeError:
            pages = None
        if not isinstance(pages, list):
            raise KaggleError(f"could not read the competition pages: {output[:200]}")

        def priority(page):
            name = str(page.get("name", "")).lower()
            return PAGE_ORDER.index(name) if name in PAGE_ORDER else len(PAGE_ORDER)

        useful = sorted((p for p in pages if "rules" not in str(p.get("name", "")).lower()), key=priority)
        return "\n\n".join(f"## {p.get('name', '')}\n{p.get('content', '')}" for p in useful)

    def download(self, slug: str, dest) -> Path:
        """Downloads beside the destination and moves in at the end, so a cut download never looks finished."""
        dest = Path(dest)
        partial = dest.with_name(dest.name + ".partial")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        self._run(slug, ["competitions", "download", slug, "-p", str(partial), "--unzip", "-q"],
                  timeout=DOWNLOAD_TIMEOUT)
        if dest.exists():
            dest.rmdir()  # only an empty folder can be there: rmdir refuses anything else
        os.replace(partial, dest)
        return dest

    def quota_left(self, slug: str) -> int:
        output = self._run(slug, ["competitions", "submission-limits", slug])
        match = re.search(r"Remaining today:\s*(\d+)", output)
        if not match:
            raise KaggleError(f"could not read the submission quota: {output[:200]}")
        return int(match.group(1))

    def submit(self, slug: str, path, message: str) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        # Never retried: a timeout may hide a submission that did count against the quota.
        error = None
        try:
            output = self._run(slug, ["competitions", "submit", slug, "-f", str(path), "-m", message,
                                      "--wait", str(SCORING_WAIT)], timeout=SCORING_WAIT + 120, retries=1)
        except KaggleError as exc:
            if not SUBMISSION_REF.search(exc.output):
                raise
            output, error = exc.output, str(exc)  # scoring failed, but the submission exists and counted
        match = SUBMISSION_REF.search(output)
        if not match:
            raise KaggleError(f"submission ref not found in: {output[:200]}")
        ref = match.group(1)
        result = {"ref": ref, "public_score": self._score_of(slug, ref)}
        return result if error is None else {**result, "error": error}

    def _score_of(self, slug: str, ref: str) -> float | None:
        try:
            return next((s["public_score"] for s in self.submissions(slug) if s["ref"] == ref), None)
        except KaggleError:  # the submission already counted: a failed read must not lose its ref
            return None

    def submissions(self, slug: str) -> list[dict]:
        output = self._run(slug, ["competitions", "submissions", slug, "-v", "--page-size", "200"])
        return [{
            "ref": row["ref"],
            "date": row.get("date"),
            "description": row.get("description"),
            "status": row.get("status"),
            "public_score": _score(row.get("publicScore")),
        } for row in _csv_rows(output, "ref")]

    def leaderboard(self, slug: str, dest) -> list[dict]:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._run(slug, ["competitions", "leaderboard", slug, "-d", "-p", str(dest), "-q"])
        archives = sorted(dest.glob(f"{slug}*.zip"), key=lambda p: p.stat().st_mtime)
        if not archives:
            raise KaggleError(f"no leaderboard archive found in {dest}")
        with zipfile.ZipFile(archives[-1]) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".csv"))
            rows = list(csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8")))
        return [{"rank": int(row.get("Rank") or i + 1), "team": row["TeamName"], "score": float(row["Score"])}
                for i, row in enumerate(rows) if _score(row.get("Score")) is not None]
