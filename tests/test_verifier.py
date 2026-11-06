import pytest

from slayer.contracts import (
    FeaturePlan,
    ModelBatch,
    ModelParams,
    ModelPlan,
    SemanticCheck,
    ValidationPlan,
    experiment_id,
)
from slayer.verifier import Verifier

STRATIFIED = {"scheme": "stratified_kfold", "n_splits": 5, "order_by": "row_order",
              "rationale": "No drift between train and test, so random folds are fine."}
FEATURES = {"drop_drifting_top_k": 0, "row_stats": True, "frequency_encode_max_cardinality": 0,
            "target_encode_max_cardinality": 0, "rationale": "Start from the plain feature set."}
PARAMS = {"learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "subsample": 0.8,
          "colsample": 0.5, "reg_lambda": 1.0, "min_child": 20}


def batch(**overrides):
    model = {"family": "lightgbm", "params": dict(PARAMS), "recency_half_life": None, "seed": 1,
             "hypothesis": "A regularised baseline gives a stable reference."}
    model.update(overrides)
    return {"models": [model]}


def check(role, raw, ctx):
    return Verifier().check(role, raw, ctx)


# --- schema layer ---------------------------------------------------------------------------
def test_schema_rejects_out_of_range_param_before_any_other_layer():
    bad = batch(params={**PARAMS, "learning_rate": 5.0})
    verdict = check("tuner", bad, {"known_experiments": set(), "feature_plan": FeaturePlan(**FEATURES),
                                   "validation": ValidationPlan(**STRATIFIED)})
    assert not verdict.passed
    assert verdict.layer == "schema"
    assert "learning_rate" in verdict.reason


def test_schema_rejects_unknown_role():
    with pytest.raises(KeyError):
        check("trader", {}, {})


# --- scout ----------------------------------------------------------------------------------
BRIEF = {"supported": True, "summary": "Tabular binary classification scored by AUC.", "risks": []}


def test_scout_cannot_claim_support_when_target_is_not_binary():
    verdict = check("scout", BRIEF, {"target_is_binary": False, "metric": "auc"})
    assert not verdict.passed and "target_is_binary" in verdict.reason


def test_scout_cannot_claim_support_for_other_metric():
    verdict = check("scout", BRIEF, {"target_is_binary": True, "metric": "rmse"})
    assert not verdict.passed and "metric_is_auc" in verdict.reason


def test_scout_accepts_area_under_curve_spelling_and_unknown_metric():
    assert check("scout", BRIEF, {"target_is_binary": True, "metric": "Area Under ROC Curve"}).passed
    assert check("scout", BRIEF, {"target_is_binary": True, "metric": None}).passed


def test_scout_may_decline_anything():
    declined = {**BRIEF, "supported": False}
    assert check("scout", declined, {"target_is_binary": False, "metric": "rmse"}).passed


# --- validation -----------------------------------------------------------------------------
@pytest.mark.parametrize("ctx", [
    {"adversarial_auc": 0.9, "temporal_hint": False},
    {"adversarial_auc": 0.5, "temporal_hint": True, "row_order_drift_auc": 0.9, "id_order_drift_auc": 0.5,
     "drift_clock_available": False},
])
def test_random_folds_rejected_when_data_is_temporal_or_drifting(ctx):
    verdict = check("validation", STRATIFIED, ctx)
    assert not verdict.passed and "random_folds_on_drift" in verdict.reason


def test_random_folds_allowed_when_time_cannot_be_recovered_and_nothing_drifts():
    ctx = {"adversarial_auc": 0.5, "temporal_hint": True, "row_order_drift_auc": 0.5, "id_order_drift_auc": 0.5,
           "drift_clock_available": False}
    assert check("validation", STRATIFIED, ctx).passed


def test_time_folds_rejected_without_temporal_evidence():
    plan = {**STRATIFIED, "scheme": "time_kfold"}
    verdict = check("validation", plan, {"adversarial_auc": 0.5, "temporal_hint": False, "row_order_drift_auc": 0.9,
                                         "id_order_drift_auc": 0.9, "drift_clock_available": False})
    assert not verdict.passed and "time_folds_need_evidence" in verdict.reason


def order_ctx(**overrides):
    ctx = {"adversarial_auc": 0.9, "temporal_hint": True, "row_order_drift_auc": 0.5, "id_order_drift_auc": 0.5,
           "drift_clock_available": True}
    ctx.update(overrides)
    return ctx


def test_time_folds_accepted_when_row_order_is_shown_to_follow_time():
    plan = {**STRATIFIED, "scheme": "time_kfold"}
    assert check("validation", plan, order_ctx(row_order_drift_auc=0.9)).passed


def test_time_folds_rejected_when_row_order_is_not_time():
    # Observed on CloudWalk's data: rows are shuffled, so "time" folds were random and CV was inflated.
    plan = {**STRATIFIED, "scheme": "time_kfold"}
    verdict = check("validation", plan, order_ctx())
    assert not verdict.passed and "order_is_time" in verdict.reason


def test_time_folds_rejected_when_id_order_is_not_time():
    plan = {**STRATIFIED, "scheme": "time_kfold", "order_by": "id"}
    verdict = check("validation", plan, order_ctx(row_order_drift_auc=0.9))
    assert not verdict.passed and "order_is_time" in verdict.reason


def test_time_folds_by_id_accepted_when_id_follows_time():
    plan = {**STRATIFIED, "scheme": "time_kfold", "order_by": "id"}
    assert check("validation", plan, order_ctx(id_order_drift_auc=0.9)).passed


def test_drift_clock_order_needs_a_clock_to_exist():
    plan = {**STRATIFIED, "scheme": "time_kfold", "order_by": "drift_clock"}
    assert check("validation", plan, order_ctx()).passed
    verdict = check("validation", plan, order_ctx(drift_clock_available=False))
    assert not verdict.passed and "order_is_time" in verdict.reason


# --- features -------------------------------------------------------------------------------
def test_feature_plan_cannot_drop_almost_everything():
    plan = {**FEATURES, "drop_drifting_top_k": 90}
    verdict = check("features", plan, {"n_features": 100})
    assert not verdict.passed and "keeps_enough_features" in verdict.reason


def test_feature_plan_within_limits_passes():
    assert check("features", {**FEATURES, "drop_drifting_top_k": 80}, {"n_features": 100}).passed


def test_a_plan_with_number_tokens_needs_the_tokenizer_file():
    plan = {**FEATURES, "number_tokens": True}
    refused = check("features", plan, {"n_features": 100, "tokenizer_available": False})
    assert not refused.passed and "tokenizer_present" in refused.reason
    assert check("features", plan, {"n_features": 100, "tokenizer_available": True}).passed


def test_a_plan_with_shared_token_rates_needs_the_tokenizer_file():
    plan = {**FEATURES, "shared_token_rates": True}
    refused = check("features", plan, {"n_features": 100, "tokenizer_available": False})
    assert not refused.passed and "tokenizer_present" in refused.reason
    assert check("features", plan, {"n_features": 100, "tokenizer_available": True}).passed


def test_joint_keys_are_refused_past_the_columns_a_scan_can_take():
    plan = {**FEATURES, "joint_keys": 8}
    refused = check("features", plan, {"n_features": 41})
    assert not refused.passed and "joint_scan_fits" in refused.reason
    assert check("features", plan, {"n_features": 40}).passed
    assert check("features", FEATURES, {"n_features": 41}).passed


# --- tuner ----------------------------------------------------------------------------------
def tuner_ctx(known=()):
    return {"known_experiments": set(known), "feature_plan": FeaturePlan(**FEATURES),
            "validation": ValidationPlan(**STRATIFIED)}


def test_tuner_cannot_repeat_an_experiment():
    model = ModelPlan(**batch()["models"][0])
    known = experiment_id(ValidationPlan(**STRATIFIED), FeaturePlan(**FEATURES), model)
    verdict = check("tuner", batch(), tuner_ctx([known]))
    assert not verdict.passed and "no_repeated_experiment" in verdict.reason


def test_experiment_id_ignores_free_text():
    plan = ModelPlan(**batch()["models"][0])
    reworded = plan.model_copy(update={"hypothesis": "Same model, explained differently."})
    assert experiment_id(ValidationPlan(**STRATIFIED), FeaturePlan(**FEATURES), plan) == experiment_id(
        ValidationPlan(**STRATIFIED), FeaturePlan(**FEATURES), reworded)


def test_tuner_cannot_repeat_within_the_same_batch():
    raw = batch()
    raw["models"].append(dict(raw["models"][0]))
    verdict = check("tuner", raw, tuner_ctx())
    assert not verdict.passed and "no_repeated_experiment" in verdict.reason


def test_lightgbm_leaves_must_fit_depth():
    raw = batch(params={**PARAMS, "max_depth": 3, "num_leaves": 64})
    verdict = check("tuner", raw, tuner_ctx())
    assert not verdict.passed and "leaves_fit_depth" in verdict.reason


def test_catboost_depth_limit():
    raw = batch(family="catboost", params={**PARAMS, "max_depth": 12})
    verdict = check("tuner", raw, tuner_ctx())
    assert not verdict.passed and "catboost_depth" in verdict.reason


def test_recency_weights_need_time_folds():
    verdict = check("tuner", batch(recency_half_life=1.0), tuner_ctx())
    assert not verdict.passed and "recency_needs_time_order" in verdict.reason


def test_valid_batch_passes_and_returns_parsed_output():
    verdict = check("tuner", batch(), tuner_ctx())
    assert verdict.passed
    assert isinstance(verdict.output, ModelBatch)
    assert isinstance(verdict.output.models[0].params, ModelParams)


# --- critic ---------------------------------------------------------------------------------
def critic_ctx(**overrides):
    ctx = {"candidates": {"blend-1": {"cv": 0.80, "std": 0.001}}, "best_submitted_cv": 0.79,
           "quota_left": 3, "min_gain": 0.0002, "canary_passed": True, "file_ok": True}
    ctx.update(overrides)
    return ctx


SUBMIT = {"action": "submit", "candidate_id": "blend-1", "reason": "CV improved past the noise level."}


def test_critic_submit_passes_when_all_gates_hold():
    assert check("critic", SUBMIT, critic_ctx()).passed


@pytest.mark.parametrize("override,name", [
    ({"candidates": {}}, "candidate_exists"),
    ({"quota_left": 0}, "quota_left"),
    ({"canary_passed": False}, "leak_canary_passed"),
    ({"file_ok": False}, "submission_file_valid"),
    ({"best_submitted_cv": 0.7999}, "cv_gain_above_noise"),
])
def test_critic_submit_blocked_by_each_gate(override, name):
    verdict = check("critic", SUBMIT, critic_ctx(**override))
    assert not verdict.passed and name in verdict.reason


def test_critic_first_submission_needs_no_gain():
    assert check("critic", SUBMIT, critic_ctx(best_submitted_cv=None)).passed


def test_critic_can_always_hold():
    hold = {**SUBMIT, "action": "hold", "reason": "Nothing beats the last submission."}
    assert check("critic", hold, critic_ctx(quota_left=0, candidates={}, canary_passed=False)).passed


# --- analyst --------------------------------------------------------------------------------
REPORT = {"diagnosis": "CV tracks the public score closely so far.", "cv_lb_gap_cited": 0.0020,
          "lessons": ["Keep the time folds."]}


def test_analyst_must_quote_the_observed_gap():
    verdict = check("analyst", REPORT, {"observed_gap": 0.0100})
    assert not verdict.passed and "gap_matches_ledger" in verdict.reason


def test_analyst_quoting_the_gap_passes():
    assert check("analyst", REPORT, {"observed_gap": 0.0021}).passed


def test_analyst_without_any_score_yet_may_omit_gap():
    assert check("analyst", {**REPORT, "cv_lb_gap_cited": None}, {"observed_gap": None}).passed


def test_analyst_cannot_omit_gap_when_observed():
    verdict = check("analyst", {**REPORT, "cv_lb_gap_cited": None}, {"observed_gap": 0.001})
    assert not verdict.passed and "gap_matches_ledger" in verdict.reason


# --- semantic layer -------------------------------------------------------------------------
class FakeCheckBrain:
    def __init__(self, ok):
        self.ok, self.calls = ok, []

    def decide(self, role, payload, schema):
        self.calls.append((role, payload))
        return SemanticCheck(ok=self.ok, notes=["Rationale contradicts the drift score."])


def test_semantic_layer_runs_only_after_deterministic_passes():
    brain = FakeCheckBrain(ok=True)
    verdict = Verifier(brain).check("validation", STRATIFIED, {"adversarial_auc": 0.9, "temporal_hint": False})
    assert not verdict.passed and verdict.layer == "deterministic"
    assert brain.calls == []


def test_semantic_layer_can_reject():
    brain = FakeCheckBrain(ok=False)
    verdict = Verifier(brain).check("validation", STRATIFIED, {"adversarial_auc": 0.5, "temporal_hint": False})
    assert not verdict.passed and verdict.layer == "semantic"
    assert "contradicts" in verdict.reason
    assert brain.calls[0][1]["role"] == "validation"


def test_semantic_layer_sees_what_the_agent_saw():
    # Observed on the first live run: the checker saw only n_features and called real facts invented.
    brain = FakeCheckBrain(ok=True)
    evidence = {"profile": {"drift_clock": {"name": "feature_492"}}}
    Verifier(brain).check("features", FEATURES, {"n_features": 100}, evidence=evidence)
    assert brain.calls[0][1]["agent_input"] == evidence


def test_semantic_layer_failure_does_not_block_the_run():
    class Broken:
        def decide(self, role, payload, schema):
            raise RuntimeError("provider down")

    verdict = Verifier(Broken()).check("validation", STRATIFIED, {"adversarial_auc": 0.5, "temporal_hint": False})
    assert verdict.passed
    assert "semantic check skipped" in verdict.notes
