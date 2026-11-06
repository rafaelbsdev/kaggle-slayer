import os

import pytest

from fakes import FakeKaggle
from slayer import __main__ as cli
from slayer.brains import HeuristicBrain
from slayer.ledger import Ledger
from slayer.orchestrator import Orchestrator, RunConfig
from slayer.report import write_report
from slayer.verifier import Verifier

SLUG = "fake-competition"


# --- report ---------------------------------------------------------------------------------
@pytest.fixture
def finished_run(tmp_path, iid_raw):
    raw, hidden = iid_raw
    config = RunConfig(slug=SLUG, root=tmp_path / "project", rounds=2, allow_submit=True, max_trees=25, patience=10,
                       device="cpu", min_gain=0.0)
    Orchestrator(config, FakeKaggle(raw, hidden), HeuristicBrain(), Verifier()).run()
    return config.root / "runs" / SLUG


def test_report_shows_the_observed_results(finished_run):
    path = write_report(finished_run / "ledger.jsonl")
    text = path.read_text(encoding="utf-8")
    ledger = Ledger(finished_run / "ledger.jsonl")
    submission = ledger.read("submission")[0]
    board = ledger.read("leaderboard")[-1]
    assert f"{submission['public_score']:.5f}" in text
    assert f"{submission['cv']:.5f}" in text
    assert f"{board['rank']} of {board['team_count']}" in text
    assert "| critic |" in text


def test_report_draws_cv_against_public_score(finished_run):
    write_report(finished_run / "ledger.jsonl")
    assert (finished_run / "cv_vs_public.png").stat().st_size > 0


def test_report_lists_what_the_verifier_rejected(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("agent_decision", role="validation", attempt=1, passed=False, layer="deterministic",
                  reason="random_folds_on_drift: temporal_hint=True", notes=[], output={})
    ledger.append("fallback", role="validation", passed=True, reason="", output={})
    text = write_report(ledger.path).read_text(encoding="utf-8")
    assert "random_folds_on_drift" in text
    assert "| validation | 1 | 1 | 1 | 0 |" in text


def test_report_marks_an_operator_note(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("lesson", round=3, diagnosis="Note from the operator.", lessons=["Ignore old scores."], source="operator")
    ledger.append("lesson", round=4, diagnosis="Scores rose.", lessons=["Keep the plan."])
    text = write_report(ledger.path).read_text(encoding="utf-8")
    assert "- Round 3 (operator): Ignore old scores." in text
    assert "- Round 4: Keep the plan." in text


def test_report_ranks_the_teams_own_score_and_not_the_boards_entry(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("submission", candidate="own", cv=0.9, public_score=0.89, gap=0.01)
    ledger.append("phase1_snapshot", rank=150, teams=744, top_percent=20.2, taken_at="T", top_20_needs=0.9, top_10_needs=0.91, top_5_needs=0.92)
    ledger.append("leaderboard", rank=107, team_count=744, percentile=14.4, cutoffs={"top_20": 0.9})
    text = write_report(ledger.path).read_text(encoding="utf-8")
    assert "150 of 744" in text and "top 20.2%" in text
    assert "107 of 744" not in text
    assert "left out here" in text


def test_report_on_an_empty_ledger_says_so(tmp_path):
    text = write_report(tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "No submissions yet" in text


# --- .env -----------------------------------------------------------------------------------
def test_env_file_fills_missing_values_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ALREADY_SET", "keep")
    monkeypatch.delenv("FROM_FILE", raising=False)
    monkeypatch.delenv("EMPTY_ONE", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nFROM_FILE="value"\nALREADY_SET=other\nEMPTY_ONE=\nnot a line\n', encoding="utf-8")
    cli.load_env_file(env)
    assert os.environ["FROM_FILE"] == "value"
    assert os.environ["ALREADY_SET"] == "keep"
    assert "EMPTY_ONE" not in os.environ


def test_missing_env_file_is_fine(tmp_path):
    cli.load_env_file(tmp_path / "none.env")


# --- commands -------------------------------------------------------------------------------
def test_claude_brain_requires_a_budget(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["run", "--competition", SLUG, "--brain", "claude"])
    assert stop.value.code != 0
    assert "--budget-usd" in capsys.readouterr().err


def test_check_reports_access_and_quota(monkeypatch, capsys, iid_raw):
    raw, hidden = iid_raw
    monkeypatch.setattr(cli, "KaggleClient", lambda: FakeKaggle(raw, hidden))
    assert cli.main(["check", "--competition", SLUG]) == 0
    output = capsys.readouterr().out
    assert "joined: True" in output and "quota left today: 5" in output


def test_check_fails_when_the_rules_are_not_accepted(monkeypatch, capsys, iid_raw):
    raw, hidden = iid_raw
    monkeypatch.setattr(cli, "KaggleClient", lambda: FakeKaggle(raw, hidden, entered=False))
    assert cli.main(["check", "--competition", SLUG]) == 1
    assert "rules" in capsys.readouterr().out


def test_run_with_heuristic_brain_end_to_end(monkeypatch, capsys, tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    monkeypatch.setattr(cli, "KaggleClient", lambda: kaggle)
    code = cli.main(["run", "--competition", SLUG, "--root", str(tmp_path), "--rounds", "1", "--max-trees", "20",
                     "--patience", "10", "--device", "cpu", "--allow-submit"])
    assert code == 0
    assert len(kaggle.submitted) == 1
    assert (tmp_path / "runs" / SLUG / "report.md").exists()
    assert "rounds_done" in capsys.readouterr().out


def test_run_is_a_dry_run_unless_told_otherwise(monkeypatch, tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    monkeypatch.setattr(cli, "KaggleClient", lambda: kaggle)
    cli.main(["run", "--competition", SLUG, "--root", str(tmp_path), "--rounds", "1", "--max-trees", "20",
              "--patience", "10", "--device", "cpu"])
    assert kaggle.submitted == []


def test_no_gain_rounds_reaches_the_run(monkeypatch, tmp_path, iid_raw):
    raw, hidden = iid_raw
    configs = []

    class Recording(Orchestrator):
        def __init__(self, config, *rest):
            configs.append(config)
            super().__init__(config, *rest)

    monkeypatch.setattr(cli, "KaggleClient", lambda: FakeKaggle(raw, hidden))
    monkeypatch.setattr(cli, "Orchestrator", Recording)
    cli.main(["run", "--competition", SLUG, "--root", str(tmp_path), "--rounds", "1", "--max-trees", "20",
              "--patience", "10", "--device", "cpu", "--no-gain-rounds", "5"])
    assert configs[0].no_gain_rounds == 5


def test_notes_reach_the_run(monkeypatch, tmp_path, iid_raw):
    raw, hidden = iid_raw
    monkeypatch.setattr(cli, "KaggleClient", lambda: FakeKaggle(raw, hidden))
    cli.main(["run", "--competition", SLUG, "--root", str(tmp_path), "--rounds", "1", "--max-trees", "20",
              "--patience", "10", "--device", "cpu", "--note", "First.", "--note", "Second."])
    ledger = Ledger(tmp_path / "runs" / SLUG / "ledger.jsonl")
    assert ledger.read("lesson")[0]["lessons"] == ["First.", "Second."]


def test_budget_spent_survives_a_restart(tmp_path):
    ledger = Ledger(tmp_path / "runs" / SLUG / "ledger.jsonl")
    ledger.append("llm_spend", spent_usd=1.25, limit_usd=10)
    ledger.append("llm_spend", spent_usd=2.5, limit_usd=10)
    assert cli.spent_so_far(tmp_path, SLUG) == 2.5
    assert cli.spent_so_far(tmp_path, "other") == 0.0
