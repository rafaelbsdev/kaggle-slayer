"""End to end, offline: real training and real scoring on hidden labels, with a fake Kaggle."""
import json
from types import SimpleNamespace

import polars as pl
import pytest

from fakes import SENTINEL_CATEGORY, SENTINEL_NUMBER, FakeKaggle
from slayer.brains import BrainRejected, BrainUnavailable, Budget, BudgetExceeded, HeuristicBrain
from slayer.contracts import CompetitionBrief, ValidationPlan
from slayer.kaggle_client import KaggleError
from slayer.ledger import Ledger
from slayer import orchestrator as orchestrator_module
from slayer.orchestrator import Orchestrator, RunConfig, plan_key
from slayer.privacy import PayloadRejected, assert_no_rows
from slayer.verifier import Verifier

SLUG = "fake-competition"


def config(tmp_path, **overrides):
    settings = {"slug": SLUG, "root": tmp_path / "project", "rounds": 2, "allow_submit": True, "max_trees": 25,
                "patience": 10, "device": "cpu", "min_gain": 0.0001}
    settings.update(overrides)
    return RunConfig(**settings)


def orchestrator(tmp_path, kaggle, brain=None, **overrides):
    return Orchestrator(config(tmp_path, **overrides), kaggle, brain or HeuristicBrain(), Verifier())


def ledger_of(run):
    return Ledger(run.config.root / "runs" / SLUG / "ledger.jsonl")


class Recording(HeuristicBrain):
    """Follows the heuristic and records everything an LLM provider would have received."""

    def __init__(self):
        self.payloads = []

    def decide(self, role, payload, schema):
        assert_no_rows(payload)
        self.payloads.append((role, payload))
        return super().decide(role, payload, schema)


class Overriding(HeuristicBrain):
    """Answers one role with a fixed output, the way a misbehaving model would."""

    def __init__(self, role, answer):
        self.role, self.answer, self.calls = role, answer, 0

    def decide(self, role, payload, schema):
        if role == self.role:
            self.calls += 1
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer
        return super().decide(role, payload, schema)


# --- the whole loop -------------------------------------------------------------------------
def test_full_run_collects_trains_submits_and_monitors(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle)
    assert run.run() == "rounds_done"

    entries = ledger_of(run).read()
    kinds = {e["kind"] for e in entries}
    assert {"data_ready", "profile", "agent_decision", "leak_canary", "experiment", "blend", "submission",
            "leaderboard", "stop"} <= kinds

    assert len(kaggle.submitted) >= 1
    submission = ledger_of(run).read("submission")[0]
    assert submission["public_score"] > 0.7
    assert submission["gap"] == pytest.approx(submission["cv"] - submission["public_score"])

    board = ledger_of(run).read("leaderboard")[-1]
    assert board["percentile"] == pytest.approx(100 * board["rank"] / board["team_count"])
    assert set(board["cutoffs"]) == {"top_1", "top_5", "top_20"}
    assert board["cutoffs"]["top_1"] > board["cutoffs"]["top_20"]


def test_submission_files_are_valid_and_scored_ids_match(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle)
    run.run()
    files = list((run.config.root / "submissions" / SLUG).glob("*.csv"))
    assert files
    assert sorted(pl.read_csv(files[0])["id"].to_list()) == sorted(pl.read_csv(raw / "test.csv")["id"].to_list())


def test_dry_run_never_submits(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle, allow_submit=False)
    run.run()
    assert kaggle.submitted == []
    assert ledger_of(run).read("dry_run")


def test_rules_not_accepted_stops_before_download(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden, entered=False)
    run = orchestrator(tmp_path, kaggle)
    assert run.run() == "kaggle_access"
    assert not (run.config.root / "data" / SLUG / "raw" / "train.csv").exists()
    assert "rules" in ledger_of(run).read("stop")[0]["detail"]


def test_zero_quota_means_no_submission(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden, daily_limit=0)
    run = orchestrator(tmp_path, kaggle)
    run.run()
    assert kaggle.submitted == []


def test_never_exceeds_the_daily_quota(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden, daily_limit=1)
    run = orchestrator(tmp_path, kaggle, rounds=3, min_gain=0.0)
    run.run()
    assert len(kaggle.submitted) <= 1


# --- agents that misbehave ------------------------------------------------------------------
def test_unsupported_competition_stops_after_scout(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    declined = CompetitionBrief(supported=False, summary="This is not a tabular competition.")
    run = orchestrator(tmp_path, kaggle, brain=Overriding("scout", declined))
    assert run.run() == "unsupported"
    assert not ledger_of(run).read("experiment")


def test_random_folds_on_temporal_data_are_rejected_then_replaced(tmp_path, temporal_raw):
    raw, hidden = temporal_raw
    kaggle = FakeKaggle(raw, hidden, overview="All loans in test occurred strictly after train. AUC metric.")
    bad = ValidationPlan(scheme="stratified_kfold", n_splits=3, rationale="Random folds are simple and fast.")
    brain = Overriding("validation", bad)
    run = orchestrator(tmp_path, kaggle, brain=brain, rounds=1)
    run.run()

    decisions = [e for e in ledger_of(run).read("agent_decision") if e["role"] == "validation"]
    assert brain.calls == 2
    assert all(not d["passed"] for d in decisions)
    assert "random_folds_on_drift" in decisions[0]["reason"]
    fallback = [e for e in ledger_of(run).read("fallback") if e["role"] == "validation"]
    assert fallback[0]["output"]["scheme"] == "time_kfold"
    assert ledger_of(run).read("experiment")[0]["validation_scheme"] == "time_kfold"


def test_shuffled_time_is_validated_by_the_drift_clock_not_by_row_order(tmp_path, shuffled_temporal_raw):
    raw, hidden = shuffled_temporal_raw
    kaggle = FakeKaggle(raw, hidden, overview="All loans in test occurred strictly after train. AUC metric.")
    bad = ValidationPlan(scheme="time_kfold", n_splits=3, rationale="Rows look ordered, so folds follow them.")
    run = orchestrator(tmp_path, kaggle, brain=Overriding("validation", bad), rounds=1)
    run.run()
    rejected = [e for e in ledger_of(run).read("agent_decision") if e["role"] == "validation"]
    assert "order_is_time" in rejected[0]["reason"]
    plan = [e for e in ledger_of(run).read("plan") if e["role"] == "validation"][0]["output"]
    assert plan["order_by"] == "drift_clock"
    assert ledger_of(run).read("experiment")


def test_contract_rejection_goes_back_to_the_agent_with_its_reason(tmp_path, iid_raw):
    raw, hidden = iid_raw
    seen = []

    class OutOfBounds(HeuristicBrain):
        def decide(self, role, payload, schema):
            if role == "tuner":
                seen.append(payload.get("rejected_because"))
                if len(seen) == 1:
                    raise BrainRejected("models.0.params.learning_rate: must be at least 0.003")
            return super().decide(role, payload, schema)

    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), brain=OutOfBounds(), rounds=1)
    assert run.run() == "rounds_done"
    assert seen[0] is None
    assert "learning_rate" in seen[1]
    assert not ledger_of(run).read("fallback")


def test_interruption_keeps_the_spend_and_records_the_stop(tmp_path, iid_raw):
    raw, hidden = iid_raw

    class ChargesThenStops(HeuristicBrain):
        def __init__(self):
            self.budget = Budget(10.0)

        def decide(self, role, payload, schema):
            self.budget.spent += 0.5
            if role == "tuner":
                raise KeyboardInterrupt
            return super().decide(role, payload, schema)

    brain = ChargesThenStops()
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), brain=brain)
    with pytest.raises(KeyboardInterrupt):
        run.run()
    assert ledger_of(run).read("llm_spend")[-1]["spent_usd"] == brain.budget.spent
    assert ledger_of(run).read("stop")[-1]["reason"] == "crashed"


def test_each_charge_reaches_the_ledger_as_it_happens(tmp_path, iid_raw):
    raw, hidden = iid_raw

    class Charging(HeuristicBrain):
        budget = Budget(10.0)

    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), brain=Charging())
    Charging.budget.charge("claude-opus-5-5", SimpleNamespace(input_tokens=1000, output_tokens=100,
                                                              cache_read_input_tokens=0, cache_creation_input_tokens=0))
    assert ledger_of(run).read("llm_spend")[-1]["spent_usd"] == pytest.approx(Charging.budget.spent)


def test_unexpected_error_is_recorded_before_it_propagates(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw

    def broken(*args, **kwargs):
        raise MemoryError("blend ran out of memory")

    monkeypatch.setattr("slayer.orchestrator.blend", broken)
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), rounds=1)
    with pytest.raises(MemoryError):
        run.run()
    stop = ledger_of(run).read("stop")[-1]
    assert stop["reason"] == "crashed" and "blend" in stop["detail"]


def test_scoring_failure_after_a_counted_submission_is_not_resubmitted(tmp_path, iid_raw):
    raw, hidden = iid_raw

    class ScoringFails(FakeKaggle):
        def submit(self, slug, path, message):
            result = super().submit(slug, path, message)
            result.update(public_score=None, error="scoring failed")
            self.submitted[-1]["public_score"] = 0.0
            return result

    kaggle = ScoringFails(raw, hidden)
    run = orchestrator(tmp_path, kaggle, rounds=2, min_gain=0.5, no_gain_rounds=9)
    run.run()
    assert len(kaggle.submitted) == 1
    entry = ledger_of(run).read("submission")[0]
    assert entry["public_score"] is None and entry["error"] == "scoring failed"


def test_rejection_reason_goes_back_to_the_agent(tmp_path, temporal_raw):
    raw, hidden = temporal_raw
    kaggle = FakeKaggle(raw, hidden, overview="Temporal split. AUC metric.")
    seen = []

    class Learner(Overriding):
        def decide(self, role, payload, schema):
            if role == "validation":
                seen.append(payload.get("rejected_because"))
            return super().decide(role, payload, schema)

    bad = ValidationPlan(scheme="stratified_kfold", n_splits=3, rationale="Random folds are simple and fast.")
    orchestrator(tmp_path, kaggle, brain=Learner("validation", bad), rounds=1).run()
    assert seen[0] is None
    assert "random_folds_on_drift" in seen[1]


def test_brain_exception_falls_back_and_the_run_continues(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle, brain=Overriding("features", RuntimeError("provider down")), rounds=1)
    assert run.run() == "rounds_done"
    assert ledger_of(run).read("agent_error")
    assert ledger_of(run).read("experiment")


def test_budget_stops_the_run_cleanly(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle, brain=Overriding("tuner", BudgetExceeded("limit reached")))
    assert run.run() == "budget"
    assert ledger_of(run).read("stop")[0]["reason"] == "budget"


def test_privacy_guard_stops_the_run_instead_of_falling_back(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle, brain=Overriding("scout", PayloadRejected("row values")))
    assert run.run() == "privacy_guard"


def test_failed_experiment_does_not_end_the_run(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    real = __import__("slayer.training", fromlist=["run_experiment"]).run_experiment

    def explode_on_catboost(validation, features, model, data, **kwargs):
        if model.family == "catboost":
            raise MemoryError("GPU out of memory")
        return real(validation, features, model, data, **kwargs)

    monkeypatch.setattr("slayer.orchestrator.run_experiment", explode_on_catboost)
    run = orchestrator(tmp_path, kaggle, rounds=1)
    assert run.run() == "rounds_done"
    assert ledger_of(run).read("experiment_failed")
    assert ledger_of(run).read("experiment")


def test_kaggle_outage_stops_with_a_recorded_reason(tmp_path, iid_raw):
    raw, hidden = iid_raw

    class Down(FakeKaggle):
        def quota_left(self, slug):
            raise KaggleError("503 Service Unavailable")

    run = orchestrator(tmp_path, Down(raw, hidden), rounds=1)
    assert run.run() == "kaggle_error"
    assert "503" in ledger_of(run).read("stop")[0]["detail"]


def test_missing_overview_is_recorded_and_the_run_goes_on(tmp_path, iid_raw):
    raw, hidden = iid_raw

    class NoPages(FakeKaggle):
        def overview(self, slug):
            raise KaggleError("pages are not available for this competition")

    run = orchestrator(tmp_path, NoPages(raw, hidden), rounds=1)
    assert run.run() == "rounds_done"
    assert ledger_of(run).read("overview_unavailable")
    assert ledger_of(run).read("experiment")


def test_step_with_no_valid_answer_stops_with_a_recorded_reason(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    real_check = Verifier.check

    def reject_features(self, role, raw_output, ctx, evidence=None):
        verdict = real_check(self, role, raw_output, ctx, evidence)
        if role == "features":
            verdict.passed, verdict.reason = False, "always rejected"
        return verdict

    monkeypatch.setattr(Verifier, "check", reject_features)
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), rounds=1)
    assert run.run() == "step_failed"
    assert "features" in ledger_of(run).read("stop")[0]["detail"]


def test_time_limit_is_checked_between_rounds(tmp_path, iid_raw):
    raw, hidden = iid_raw
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), rounds=3, max_minutes=0)
    assert run.run() == "time_limit"
    assert not ledger_of(run).read("experiment")


def test_unusable_brain_stops_the_run_instead_of_silently_using_heuristics(tmp_path, iid_raw):
    raw, hidden = iid_raw
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden), brain=Overriding("scout", BrainUnavailable("bad API key")))
    assert run.run() == "brain_unavailable"
    assert not ledger_of(run).read("fallback")


# --- gates ----------------------------------------------------------------------------------
def test_failed_canary_blocks_every_submission(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    monkeypatch.setattr("slayer.orchestrator.leak_canary", lambda *a, **k: {"auc": 0.97, "passed": False})

    class Reckless(HeuristicBrain):
        def decide(self, role, payload, schema):
            answer = super().decide(role, payload, schema)
            if role == "critic":
                best = max(payload["candidates"], key=lambda c: payload["candidates"][c]["cv"])
                return schema(action="submit", candidate_id=best, reason="Submit it anyway, the score looks great.")
            return answer

    run = orchestrator(tmp_path, kaggle, brain=Reckless(), rounds=1)
    run.run()
    assert kaggle.submitted == []
    rejected = [e for e in ledger_of(run).read("agent_decision") if e["role"] == "critic" and not e["passed"]]
    assert "leak_canary_passed" in rejected[0]["reason"]


def test_invalid_submission_file_blocks_submission(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    monkeypatch.setattr("slayer.orchestrator.validate_submission", lambda path, sample: ["constant predictions"])
    run = orchestrator(tmp_path, kaggle, rounds=1)
    run.run()
    assert kaggle.submitted == []


# --- privacy --------------------------------------------------------------------------------
def test_no_row_value_reaches_any_agent_during_a_whole_run(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    brain = Recording()
    orchestrator(tmp_path, kaggle, brain=brain).run()
    assert {role for role, _ in brain.payloads} >= {"scout", "validation", "features", "tuner", "critic", "analyst"}
    text = json.dumps([payload for _, payload in brain.payloads], default=str)
    assert SENTINEL_CATEGORY not in text
    assert str(SENTINEL_NUMBER) not in text


def test_the_semantic_check_receives_each_agents_input_and_no_row_values(tmp_path, iid_raw):
    raw, hidden = iid_raw
    checker = Recording()
    Orchestrator(config(tmp_path, rounds=1), FakeKaggle(raw, hidden), HeuristicBrain(), Verifier(checker)).run()
    checks = {payload["role"]: payload for role, payload in checker.payloads if role == "semantic_check"}
    assert checks["features"]["agent_input"]["profile"]["n_features"] == 9
    text = json.dumps(list(checks.values()), default=str)
    assert SENTINEL_CATEGORY not in text
    assert str(SENTINEL_NUMBER) not in text


def test_ledger_holds_no_row_values(tmp_path, iid_raw):
    raw, hidden = iid_raw
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden))
    run.run()
    text = ledger_of(run).path.read_text(encoding="utf-8")
    assert SENTINEL_CATEGORY not in text
    assert str(SENTINEL_NUMBER) not in text


# --- learning and resuming ------------------------------------------------------------------
def test_lessons_from_the_analyst_reach_the_next_round(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)

    class Teaching(Recording):
        def decide(self, role, payload, schema):
            answer = super().decide(role, payload, schema)
            if role == "analyst":
                return answer.model_copy(update={"lessons": ["Prefer stronger regularisation."]})
            return answer

    brain = Teaching()
    orchestrator(tmp_path, kaggle, brain=brain, min_gain=0.0).run()
    tuner_payloads = [p for role, p in brain.payloads if role == "tuner"]
    assert tuner_payloads[0]["lessons"] == []
    assert "Prefer stronger regularisation." in tuner_payloads[1]["lessons"]


def test_an_operator_note_replaces_earlier_lessons_and_stays_in_view(tmp_path, iid_raw):
    raw, hidden = iid_raw
    note = "This option was built wrong, so ignore comparisons against it."

    class Teaching(Recording):
        round = 0

        def decide(self, role, payload, schema):
            answer = super().decide(role, payload, schema)
            if role == "analyst":
                Teaching.round += 1
                return answer.model_copy(update={"lessons": [f"Lesson {Teaching.round}.{i}." for i in range(5)]})
            return answer

    kaggle = FakeKaggle(raw, hidden, daily_limit=50)
    orchestrator(tmp_path, kaggle, brain=Teaching(), rounds=1, min_gain=0.0).run()

    brain = Teaching()
    run = orchestrator(tmp_path, kaggle, brain=brain, rounds=3, min_gain=0.0, no_gain_rounds=5, notes=[note])
    run.run()
    tuner_payloads = [p for role, p in brain.payloads if role == "tuner"]
    assert tuner_payloads[0]["lessons"] == [note]  # the first run's lessons are gone
    assert all(p["lessons"][0] == note for p in tuner_payloads)
    assert len(tuner_payloads[2]["lessons"]) == 11  # still there once the window has filled
    assert ledger_of(run).read("lesson")[1]["source"] == "operator"


def test_runs_of_a_retired_option_stay_out_of_what_agents_see(tmp_path, iid_raw):
    raw, hidden = iid_raw
    brain = Recording()
    kaggle = FakeKaggle(raw, hidden, daily_limit=50)
    run = orchestrator(tmp_path, kaggle, brain=brain, rounds=1)
    run.run()
    ledger = ledger_of(run)
    kept = ledger.read("experiment")[0]
    ledger.append("experiment", **{**{k: v for k, v in kept.items() if k not in ("ts", "kind")}, "id": "retired",
                                   "features": {**kept["features"], "digit_keys": True}, "cv": 0.999})
    (run.artifacts / "retired.npz").write_bytes((run.artifacts / f"{kept['id']}.npz").read_bytes())
    assert "retired" not in {e["id"] for e in run._experiment_summaries()}
    assert kept["id"] in {e["id"] for e in run._experiment_summaries()}

    ledger.append("experiment", **{**{k: v for k, v in kept.items() if k not in ("ts", "kind")}, "id": "old_family", "family": "gone"})
    (run.artifacts / "old_family.npz").write_bytes((run.artifacts / f"{kept['id']}.npz").read_bytes())
    assert "old_family" not in {e["id"] for e in run._experiment_summaries()}


def test_a_submission_from_an_older_blend_does_not_set_the_bar(tmp_path, iid_raw):
    raw, hidden = iid_raw
    brain = Recording()
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden, daily_limit=50), brain=brain, rounds=2, min_gain=0.0, no_gain_rounds=5)
    ledger_of(run).append("submission", ref="old", candidate="blend-r0", cv=0.9999, public_score=0.9, gap=0.0999, reason="An old blend.", error=None)
    run.run()
    bars = [p["best_submitted_cv"] for role, p in brain.payloads if role == "critic"]
    assert bars[0] is None
    new = [s for s in ledger_of(run).read("submission") if s["ref"] != "old"]
    assert new and bars[1] == new[0]["cv"]
    assert all(s["blend_version"] == orchestrator_module.BLEND_VERSION for s in new)


def test_the_feature_engineer_is_told_whether_tokens_can_run(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    monkeypatch.setattr(orchestrator_module, "tokenizer_available", lambda: False)
    brain = Recording()
    orchestrator(tmp_path, FakeKaggle(raw, hidden, daily_limit=50), brain=brain, rounds=1).run()
    assert [p["tokenizer_available"] for role, p in brain.payloads if role == "features"] == [False]


def test_a_plan_is_what_runs_apart_from_the_seed_and_later_options():
    row = {"family": "lightgbm", "params": {"max_depth": 2}, "features": {"number_keys": True},
           "recency_half_life": None, "seed": 1}
    assert plan_key(row) == plan_key({**row, "seed": 2})
    assert plan_key(row) == plan_key({**row, "features": {"number_keys": True, "row_stats": False}})
    assert plan_key(row) != plan_key({**row, "features": {"number_keys": True, "row_stats": True}})
    assert plan_key(row) != plan_key({**row, "family": "xgboost"})
    assert plan_key(row) != plan_key({**row, "params": {"max_depth": 3}})
    assert plan_key(row) != plan_key({**row, "recency_half_life": 1.0})


def test_the_blend_is_told_which_models_share_a_plan(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    seen = {}
    real = orchestrator_module.blend

    def spying(*args, **kwargs):
        seen.update(kwargs["plans"])
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "blend", spying)
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden, daily_limit=50), rounds=1)
    run.run()
    experiments = ledger_of(run).read("experiment")
    assert seen == {e["id"]: plan_key(e) for e in experiments}


def test_the_blend_is_told_each_models_family(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    seen = {}
    real = orchestrator_module.blend

    def spying(*args, **kwargs):
        seen.update(kwargs["families"])
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "blend", spying)
    run = orchestrator(tmp_path, FakeKaggle(raw, hidden, daily_limit=50), rounds=1)
    run.run()
    assert seen == {e["id"]: e["family"] for e in ledger_of(run).read("experiment")}


def test_resume_reuses_experiments_and_the_validation_plan(tmp_path, iid_raw, monkeypatch):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden, daily_limit=50)
    orchestrator(tmp_path, kaggle, rounds=1).run()

    trained = []
    real = __import__("slayer.training", fromlist=["run_experiment"]).run_experiment

    def counting(validation, features, model, data, **kwargs):
        trained.append(model.seed)
        return real(validation, features, model, data, **kwargs)

    monkeypatch.setattr("slayer.orchestrator.run_experiment", counting)
    brain = Recording()
    run = orchestrator(tmp_path, kaggle, brain=brain, rounds=1)
    run.run()
    assert "validation" not in {role for role, _ in brain.payloads}
    assert "scout" not in {role for role, _ in brain.payloads}
    tuner_payload = next(p for role, p in brain.payloads if role == "tuner")
    assert len(tuner_payload["experiments"]) >= 1
    assert trained


def test_stops_when_rounds_bring_no_gain(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle, rounds=10, min_gain=0.5, no_gain_rounds=2)
    assert run.run() == "no_gain"
    assert len(ledger_of(run).read("blend")) < 10


def test_stops_when_target_percentile_is_reached(tmp_path, iid_raw):
    raw, hidden = iid_raw
    kaggle = FakeKaggle(raw, hidden)
    run = orchestrator(tmp_path, kaggle, rounds=5, target_percentile=100.0)
    assert run.run() == "target_reached"
