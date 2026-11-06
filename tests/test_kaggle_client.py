import json
import subprocess
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from slayer.kaggle_client import DOWNLOAD_TIMEOUT, KaggleClient, KaggleError, NotAuthenticated, RulesNotAccepted

SLUG = "playground-series-s6e11"


class Script:
    """Replaces subprocess.run. Records each command and answers from a list."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        out, err, code = self.answers.pop(0)
        return subprocess.CompletedProcess(args, code, out, err)


def client(*answers):
    script = Script(*answers)
    return KaggleClient(runner=script, retries=1, sleep=lambda s: None), script


def test_invalid_slug_never_reaches_the_shell():
    kaggle, script = client()
    with pytest.raises(ValueError, match="slug"):
        kaggle.quota_left("foo; rm -rf /")
    assert script.calls == []


def test_quota_is_parsed_from_the_cli_output():
    kaggle, script = client(("Submissions today: 2\nLifetime submissions: 9\nRemaining today: 3\n", "", 0))
    assert kaggle.quota_left(SLUG) == 3
    assert script.calls[0][-3:] == ["competitions", "submission-limits", SLUG]


def test_unreadable_quota_is_an_error_not_a_guess():
    kaggle, _ = client(("something else", "", 0))
    with pytest.raises(KaggleError, match="quota"):
        kaggle.quota_left(SLUG)


def test_info_finds_the_exact_competition():
    csv = ("ref,deadline,category,reward,teamCount,userHasEntered,userRank\n"
           f"https://www.kaggle.com/competitions/{SLUG}-other,2026-12-01,Playground,Swag,5,False,\n"
           f"https://www.kaggle.com/competitions/{SLUG},2026-11-30,Playground,Swag,590,True,42\n")
    kaggle, _ = client((csv, "", 0))
    info = kaggle.info(SLUG)
    assert info["entered"] is True
    assert info["team_count"] == 590
    assert info["user_rank"] == 42


def test_info_treats_rank_zero_as_no_rank():
    # Observed on 2026-11-03: Kaggle reports userRank 0 before the first submission.
    csv = ("ref,deadline,category,reward,teamCount,userHasEntered,userRank\n"
           f"https://www.kaggle.com/competitions/{SLUG},2026-11-30,Playground,Swag,11,True,0\n")
    kaggle, _ = client((csv, "", 0))
    assert kaggle.info(SLUG)["user_rank"] is None


def test_info_on_unknown_competition_is_an_error():
    header = "ref,deadline,category,reward,teamCount,userHasEntered,userRank\n"
    kaggle, _ = client((header, "", 0))
    with pytest.raises(KaggleError, match="not found"):
        kaggle.info(SLUG)


def test_submit_uses_wait_and_reads_score_back_by_ref(tmp_path):
    file = tmp_path / "s.csv"
    file.write_text("id,prediction\n1,0.5\n")
    listing = ("ref,fileName,date,description,status,publicScore,privateScore\n"
               "111,x.csv,2026-11-03,old,complete,0.90,\n"
               "222,s.csv,2026-11-03,msg,complete,0.95123,\n")
    kaggle, script = client(("Submission ref: 222\n", "", 0), (listing, "", 0))
    result = kaggle.submit(SLUG, file, "msg")
    assert result == {"ref": "222", "public_score": 0.95123}
    command = script.calls[0]
    assert "--wait" in command and str(file) in command and "msg" in command


def test_submit_never_retries_on_failure(tmp_path):
    file = tmp_path / "s.csv"
    file.write_text("id,prediction\n1,0.5\n")
    script = Script(("", "503 Service Unavailable", 1), ("Submission ref: 2\n", "", 0))
    kaggle = KaggleClient(runner=script, retries=3, sleep=lambda s: None)
    with pytest.raises(KaggleError):
        kaggle.submit(SLUG, file, "msg")
    assert len(script.calls) == 1


def test_submit_without_a_score_returns_none(tmp_path):
    file = tmp_path / "s.csv"
    file.write_text("id,prediction\n1,0.5\n")
    listing = ("ref,fileName,date,description,status,publicScore,privateScore\n"
               "222,s.csv,2026-11-03,msg,pending,,\n")
    kaggle, _ = client(("Submission ref: 222\n", "", 0), (listing, "", 0))
    result = kaggle.submit(SLUG, file, "msg")
    assert result["public_score"] is None
    assert "error" not in result


def test_submit_refuses_a_missing_file(tmp_path):
    kaggle, script = client()
    with pytest.raises(FileNotFoundError):
        kaggle.submit(SLUG, tmp_path / "nothing.csv", "msg")
    assert script.calls == []


def test_read_commands_retry_on_transient_errors():
    script = Script(("", "503 Service Unavailable", 1), ("Remaining today: 4\n", "", 0))
    kaggle = KaggleClient(runner=script, retries=3, sleep=lambda s: None)
    assert kaggle.quota_left(SLUG) == 4
    assert len(script.calls) == 2


@pytest.mark.parametrize("stderr,error", [
    ("403 Client Error: Forbidden", RulesNotAccepted),
    ("You must accept the competition rules", RulesNotAccepted),
    ("401 Client Error: Unauthorized", NotAuthenticated),
    ("Could not find kaggle credentials", NotAuthenticated),
    ("Authentication required to call the Kaggle API.", NotAuthenticated),
    ("some other failure", KaggleError),
])
def test_errors_are_classified_with_no_retry(stderr, error):
    script = Script((stderr and "", stderr, 1), ("", stderr, 1), ("", stderr, 1))
    kaggle = KaggleClient(runner=script, retries=3, sleep=lambda s: None)
    with pytest.raises(error):
        kaggle.quota_left(SLUG)
    assert len(script.calls) == 1


def test_leaderboard_reads_the_downloaded_zip(tmp_path):
    def fake_download(args, **kwargs):
        with zipfile.ZipFile(tmp_path / f"{SLUG}-publicleaderboard.zip", "w") as archive:
            archive.writestr("board.csv", "Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount\n"
                                          "1,10,Alpha,2026-11-03,0.9524,5\n2,11,Beta,2026-11-03,0.9520,3\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    kaggle = KaggleClient(runner=fake_download, retries=1, sleep=lambda s: None)
    board = kaggle.leaderboard(SLUG, tmp_path)
    assert board == [{"rank": 1, "team": "Alpha", "score": 0.9524}, {"rank": 2, "team": "Beta", "score": 0.9520}]


PAGES = json.dumps([
    {"name": "data-description", "content": "## Data\nColumns and files."},
    {"name": "rules", "content": "ENTRY CONSTITUTES ACCEPTANCE of a very long legal text."},
    {"name": "Evaluation", "content": "Submissions are evaluated using AUC."},
    {"name": "abstract", "content": "### Overview\nPredict repayment."},
    {"name": "foundational-rules", "content": "More legal text."},
])


def test_overview_lists_pages_as_json_and_orders_the_useful_ones_first():
    kaggle, script = client((PAGES, "", 0))
    text = kaggle.overview(SLUG)
    assert script.calls[0][-7:] == ["competitions", "pages", "list", "-c", SLUG, "--content", "--format", "json"][1:]
    assert text.index("Predict repayment.") < text.index("evaluated using AUC.") < text.index("Columns and files.")
    assert "legal text" not in text


def test_overview_with_unreadable_output_is_a_kaggle_error():
    kaggle, _ = client(("name  content\n----  -------\nb'data-description ...", "", 0))
    with pytest.raises(KaggleError, match="pages"):
        kaggle.overview(SLUG)


def test_download_appears_at_the_destination_only_when_complete(tmp_path):
    dest = tmp_path / "data" / "raw"
    seen = {}

    def fake_download(args, **kwargs):
        seen["args"], seen["timeout"] = args, kwargs["timeout"]
        partial = Path(args[args.index("-p") + 1])
        assert partial != dest and not dest.exists()
        (partial / "train.csv").write_text("id\n1\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    kaggle = KaggleClient(runner=fake_download, retries=1, sleep=lambda s: None)
    assert kaggle.download(SLUG, dest) == dest
    assert (dest / "train.csv").exists()
    assert seen["args"][1:4] == ["competitions", "download", SLUG]
    assert "--unzip" in seen["args"]
    assert seen["timeout"] == DOWNLOAD_TIMEOUT


def test_failed_download_leaves_nothing_at_the_destination(tmp_path):
    dest = tmp_path / "raw"

    def half_then_fail(args, **kwargs):
        partial = Path(args[args.index("-p") + 1])
        (partial / "train.csv").write_text("id\n1\n")
        return subprocess.CompletedProcess(args, 1, "", "disk full")

    kaggle = KaggleClient(runner=half_then_fail, retries=1, sleep=lambda s: None)
    with pytest.raises(KaggleError):
        kaggle.download(SLUG, dest)
    assert not dest.exists()


def test_download_starts_clean_after_an_earlier_interruption(tmp_path):
    dest = tmp_path / "raw"
    leftover = tmp_path / "raw.partial"
    leftover.mkdir()
    (leftover / "train.csv").write_text("cut sho")

    def fake_download(args, **kwargs):
        partial = Path(args[args.index("-p") + 1])
        assert list(partial.iterdir()) == []
        (partial / "train.csv").write_text("id\n1\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    kaggle = KaggleClient(runner=fake_download, retries=1, sleep=lambda s: None)
    kaggle.download(SLUG, dest)
    assert (dest / "train.csv").read_text() == "id\n1\n"


def test_submission_counted_before_a_scoring_failure_keeps_its_ref(tmp_path):
    file = tmp_path / "s.csv"
    file.write_text("id,prediction\n1,0.5\n")
    listing = ("ref,fileName,date,description,status,publicScore,privateScore\n"
               "333,s.csv,2026-11-03,msg,error,,\n")
    kaggle, script = client(("Submission ref: 333\nScoring failed", "error while scoring", 1), (listing, "", 0))
    result = kaggle.submit(SLUG, file, "msg")
    assert result["ref"] == "333"
    assert result["public_score"] is None
    assert "scoring" in result["error"]
    assert len(script.calls) == 2


def test_submission_failure_with_no_ref_is_still_an_error(tmp_path):
    file = tmp_path / "s.csv"
    file.write_text("id,prediction\n1,0.5\n")
    kaggle, _ = client(("", "400 Bad Request: wrong columns", 1))
    with pytest.raises(KaggleError, match="wrong columns"):
        kaggle.submit(SLUG, file, "msg")


def test_cli_timeout_becomes_a_kaggle_error():
    def too_slow(args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    kaggle = KaggleClient(runner=too_slow, retries=1, sleep=lambda s: None)
    with pytest.raises(KaggleError, match="timed out"):
        kaggle.quota_left(SLUG)


class FakeLogin:
    """Replaces the SDK's credentials: counts renewals and can fail like the network does."""

    def __init__(self, events, error=None):
        self.events, self.error = events, error

    def load(self, client, file_path):
        return self

    def refresh_access_token(self):
        self.events.append("renew")
        if self.error:
            raise self.error


def login_expiring_in(path, minutes):
    expiry = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    path.write_text(json.dumps({"access_token_expiration": expiry.isoformat()}), encoding="utf-8")


def recording_cli(events, answer=("Remaining today: 3\n", "", 0)):
    def run(args, **kwargs):
        events.append("cli")
        return subprocess.CompletedProcess(args, answer[2], answer[0], answer[1])
    return KaggleClient(runner=run, retries=1, sleep=lambda s: None)


@pytest.mark.parametrize("minutes", [10, -10, -600])
def test_expiring_or_expired_login_is_renewed_before_the_cli_runs(no_real_kaggle_login, monkeypatch, minutes):
    login_expiring_in(no_real_kaggle_login, minutes)
    events = []
    monkeypatch.setattr("slayer.kaggle_client.KaggleCredentials", FakeLogin(events))
    recording_cli(events).quota_left(SLUG)
    assert events == ["renew", "cli"]


def test_login_with_time_left_is_not_renewed(no_real_kaggle_login, monkeypatch):
    login_expiring_in(no_real_kaggle_login, 60)
    events = []
    monkeypatch.setattr("slayer.kaggle_client.KaggleCredentials", FakeLogin(events))
    recording_cli(events).quota_left(SLUG)
    assert events == ["cli"]


@pytest.mark.parametrize("content", [None, "not json", "{}", '{"access_token_expiration": ""}', "[]"])
def test_absent_or_unreadable_login_is_left_to_the_cli(no_real_kaggle_login, monkeypatch, content):
    if content is not None:
        no_real_kaggle_login.write_text(content, encoding="utf-8")
    events = []
    monkeypatch.setattr("slayer.kaggle_client.KaggleCredentials", FakeLogin(events))
    recording_cli(events).quota_left(SLUG)
    assert events == ["cli"]


def test_failed_renewal_still_reaches_the_cli_and_its_error_is_classified(no_real_kaggle_login, monkeypatch):
    login_expiring_in(no_real_kaggle_login, -10)
    events = []
    monkeypatch.setattr("slayer.kaggle_client.KaggleCredentials", FakeLogin(events, error=ConnectionError("down")))
    with pytest.raises(NotAuthenticated):
        recording_cli(events, answer=("", "401 Client Error: Unauthorized", 1)).quota_left(SLUG)
    assert events == ["renew", "cli"]


def test_login_with_no_refresh_token_is_left_to_the_cli(no_real_kaggle_login, monkeypatch):
    login_expiring_in(no_real_kaggle_login, -10)
    events = []

    class NoRefreshToken:
        @staticmethod
        def load(client, file_path):
            return None

    monkeypatch.setattr("slayer.kaggle_client.KaggleCredentials", NoRefreshToken)
    recording_cli(events).quota_left(SLUG)
    assert events == ["cli"]
