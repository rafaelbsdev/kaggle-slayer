from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from slayer.brains import (
    MODEL_PRICES,
    PROMPTS,
    BrainRefused,
    BrainRejected,
    BrainUnavailable,
    Budget,
    BudgetExceeded,
    ClaudeBrain,
    HeuristicBrain,
    _answer_schema,
)
from slayer.contracts import CONTRACTS, CompetitionBrief, FeaturePlan, ModelBatch, SemanticCheck, ValidationPlan
from slayer.privacy import PayloadRejected

BRIEF = CompetitionBrief(supported=True, summary="A tabular binary competition.")


def usage(input_tokens=1000, output_tokens=100, cache_read=0, cache_write=0):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)


class FakeMessages:
    def __init__(self, response):
        self.response, self.calls = response, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(stop_reason="end_turn", text=BRIEF.model_dump_json(), tokens=None):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    response = SimpleNamespace(stop_reason=stop_reason, content=content, usage=tokens or usage(), stop_details=None)
    return SimpleNamespace(messages=FakeMessages(response), beta=SimpleNamespace(messages=FakeMessages(response)))


def test_every_role_has_a_prompt():
    assert set(CONTRACTS) | {"semantic_check"} == set(PROMPTS)


def test_returns_parsed_output_and_charges_the_budget():
    budget = Budget(1.0)
    brain = ClaudeBrain(fake_client(), "claude-opus-5-5", budget)
    answer = brain.decide("scout", {"n_train": 10}, CompetitionBrief)
    assert answer == BRIEF
    assert budget.spent == pytest.approx((1000 * 4 + 100 * 20) / 1_000_000)


def test_cache_tokens_are_priced_differently():
    budget = Budget(1.0)
    client = fake_client(tokens=usage(input_tokens=0, output_tokens=0, cache_read=1000, cache_write=1000))
    ClaudeBrain(client, "claude-opus-5-5", budget).decide("scout", {}, CompetitionBrief)
    assert budget.spent == pytest.approx((1000 * 4 * 0.1 + 1000 * 4 * 1.25) / 1_000_000)


def test_opus_requests_server_side_fallback_and_explicit_effort():
    client = fake_client()
    ClaudeBrain(client, "claude-opus-5-5", Budget(1.0), effort="high").decide("scout", {}, CompetitionBrief)
    call = client.beta.messages.calls[0]
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"]["effort"] == "high"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": _answer_schema(CompetitionBrief)}
    assert client.messages.calls == []


def objects_in(node):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield node
        for value in node.values():
            yield from objects_in(value)
    elif isinstance(node, list):
        for value in node:
            yield from objects_in(value)


@pytest.mark.parametrize("schema", [*CONTRACTS.values(), SemanticCheck, ModelBatch], ids=lambda schema: schema.__name__)
def test_no_field_is_optional_in_what_the_api_compiles(schema):
    objects = list(objects_in(_answer_schema(schema)))
    assert objects
    assert all(sorted(node["required"]) == sorted(node["properties"]) for node in objects)


def test_nested_objects_are_required_too_and_defaults_stay_visible():
    assert len(list(objects_in(_answer_schema(ModelBatch)))) == 3
    fields = _answer_schema(FeaturePlan)["properties"]
    assert "default: 0.0001" in fields["logistic_tolerance"]["description"]
    assert "default: 1.0" in fields["logistic_start_penalty"]["description"]


def test_haiku_has_no_server_side_fallback():
    client = fake_client(text=SemanticCheck(ok=True).model_dump_json())
    ClaudeBrain(client, "claude-haiku-5-5", Budget(1.0), effort="low").decide("semantic_check", {}, SemanticCheck)
    assert client.beta.messages.calls == []
    assert "fallbacks" not in client.messages.calls[0]


def test_payload_with_row_values_never_reaches_the_provider():
    client = fake_client()
    brain = ClaudeBrain(client, "claude-opus-5-5", Budget(1.0))
    with pytest.raises(PayloadRejected):
        brain.decide("scout", {"columns": [{"name": "f1", "min": 1.0}]}, CompetitionBrief)
    assert client.beta.messages.calls == []


def test_budget_stops_the_call_before_it_is_made():
    client = fake_client()
    brain = ClaudeBrain(client, "claude-opus-5-5", Budget(1.0, spent=1.0))
    with pytest.raises(BudgetExceeded):
        brain.decide("scout", {}, CompetitionBrief)
    assert client.beta.messages.calls == []


def test_refusal_is_raised_not_parsed():
    brain = ClaudeBrain(fake_client(stop_reason="refusal"), "claude-opus-5-5", Budget(1.0))
    with pytest.raises(BrainRefused):
        brain.decide("scout", {}, CompetitionBrief)


def test_answer_outside_the_contract_is_charged_and_returned_with_the_reason():
    # The API only enforces the JSON shape, so bounds such as ge/le are ours to check.
    budget = Budget(1.0)
    answer = '{"scheme": "stratified_kfold", "n_splits": 500, "order_by": "row_order", "rationale": "Many folds."}'
    brain = ClaudeBrain(fake_client(text=answer), "claude-opus-5-5", budget)
    with pytest.raises(BrainRejected, match="n_splits"):
        brain.decide("validation", {}, ValidationPlan)
    assert budget.spent > 0


def test_answer_cut_off_by_the_token_limit_is_charged_and_explained():
    budget = Budget(1.0)
    brain = ClaudeBrain(fake_client(stop_reason="max_tokens", text='{"supported": tr'), "claude-opus-5-5", budget)
    with pytest.raises(BrainRejected, match="max_tokens"):
        brain.decide("scout", {}, CompetitionBrief)
    assert budget.spent > 0


def test_answer_with_no_text_block_is_rejected():
    client = fake_client()
    client.beta.messages.response.content = []
    with pytest.raises(BrainRejected):
        ClaudeBrain(client, "claude-opus-5-5", Budget(1.0)).decide("scout", {}, CompetitionBrief)


def test_each_charge_is_reported_as_it_happens():
    reported = []
    budget = Budget(1.0, on_charge=reported.append)
    brain = ClaudeBrain(fake_client(stop_reason="refusal"), "claude-opus-5-5", budget)
    with pytest.raises(BrainRefused):
        brain.decide("scout", {}, CompetitionBrief)
    assert reported == [budget.spent]


def test_refused_call_is_still_charged():
    budget = Budget(1.0)
    with pytest.raises(BrainRefused):
        ClaudeBrain(fake_client(stop_reason="refusal"), "claude-opus-5-5", budget).decide("scout", {}, CompetitionBrief)
    assert budget.spent > 0


def test_unknown_model_cannot_silently_cost_nothing():
    with pytest.raises(ValueError, match="price"):
        ClaudeBrain(fake_client(), "claude-unknown-1", Budget(1.0))


def test_every_priced_model_has_input_and_output():
    assert all(len(prices) == 2 for prices in MODEL_PRICES.values())


def test_bad_credentials_surface_as_brain_unavailable():
    response = httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)

    class Failing:
        def create(self, **kwargs):
            raise error

    client = SimpleNamespace(messages=Failing(), beta=SimpleNamespace(messages=Failing()))
    with pytest.raises(BrainUnavailable):
        ClaudeBrain(client, "claude-opus-5-5", Budget(1.0)).decide("scout", {}, CompetitionBrief)


def test_bad_request_surfaces_as_brain_unavailable_instead_of_silent_heuristics():
    # A malformed request or an empty credit balance fails the same way on every retry and every agent.
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    error = anthropic.BadRequestError("credit balance is too low", response=response, body=None)

    class Failing:
        def create(self, **kwargs):
            raise error

    client = SimpleNamespace(messages=Failing(), beta=SimpleNamespace(messages=Failing()))
    with pytest.raises(BrainUnavailable, match="credit"):
        ClaudeBrain(client, "claude-opus-5-5", Budget(1.0)).decide("scout", {}, CompetitionBrief)


@pytest.mark.parametrize("schema", list(CONTRACTS.values()) + [SemanticCheck])
def test_every_contract_converts_to_a_structured_output_schema(schema):
    assert _answer_schema(schema)["additionalProperties"] is False


# --- heuristic brain ------------------------------------------------------------------------
def test_heuristic_answers_every_role_with_a_valid_contract():
    brain = HeuristicBrain()
    payloads = {
        "scout": {"target_is_binary": True, "metric": "auc"},
        "validation": {"temporal_hint": False, "profile": {"adversarial_auc": 0.5}},
        "features": {"round": 0, "profile": {"n_features": 10, "features_with_nulls": 0}, "temporal": False},
        "tuner": {"round": 0},
        "critic": {"candidates": {"a": {"cv": 0.8, "std": 0.0}}, "best_submitted_cv": None, "quota_left": 1,
                   "min_gain": 0.0001, "canary_passed": True, "file_ok": True},
        "analyst": {"observed_gap": 0.01},
    }
    for role, payload in payloads.items():
        assert isinstance(brain.decide(role, payload, CONTRACTS[role]), CONTRACTS[role])


@pytest.mark.parametrize("profile,order_by", [
    ({"adversarial_auc": 0.9, "row_order_drift_auc": 0.9, "id_order_drift_auc": 0.9, "drift_clock": {"name": "a"}}, "row_order"),
    ({"adversarial_auc": 0.9, "row_order_drift_auc": 0.5, "id_order_drift_auc": 0.9, "drift_clock": {"name": "a"}}, "id"),
    ({"adversarial_auc": 1.0, "row_order_drift_auc": 0.5, "id_order_drift_auc": 0.5, "drift_clock": {"name": "a"}}, "drift_clock"),
])
def test_heuristic_validation_uses_an_order_shown_to_follow_time(profile, order_by):
    plan = HeuristicBrain().decide("validation", {"temporal_hint": True, "profile": profile}, CONTRACTS["validation"])
    assert (plan.scheme, plan.order_by) == ("time_kfold", order_by)


def test_heuristic_features_drop_just_enough_to_remove_the_drift():
    brain = HeuristicBrain()
    profile = {"n_features": 1000, "features_with_nulls": 10, "adversarial_auc": 1.0,
               "drift_after_drop": [{"k": 1, "adversarial_auc": 0.99}, {"k": 2, "adversarial_auc": 0.7},
                                    {"k": 5, "adversarial_auc": 0.55}]}
    plan = brain.decide("features", {"round": 0, "profile": profile, "temporal": True}, CONTRACTS["features"])
    assert plan.drop_drifting_top_k == 2


def test_heuristic_features_keep_everything_without_drift():
    brain = HeuristicBrain()
    profile = {"n_features": 24, "features_with_nulls": 0, "adversarial_auc": 0.5,
               "drift_after_drop": [{"k": 1, "adversarial_auc": 0.5}]}
    plan = brain.decide("features", {"round": 1, "profile": profile, "temporal": False}, CONTRACTS["features"])
    assert plan.drop_drifting_top_k == 0


def test_heuristic_taking_over_late_keeps_the_best_measured_feature_plan():
    profile = {"n_features": 24, "features_with_nulls": 0, "adversarial_auc": 0.5, "drift_after_drop": []}
    best = {"drop_drifting_top_k": 0, "row_stats": False, "frequency_encode_max_cardinality": 1000,
            "target_encode_max_cardinality": 700000, "target_encode_bags": 1, "number_keys": True}
    experiments = [{"cv": 0.95, "features": best}, {"cv": 0.94, "features": {"target_encode_max_cardinality": 50}}]
    payload = {"round": 12, "profile": profile, "temporal": False, "experiments": experiments}
    plan = HeuristicBrain().decide("features", payload, CONTRACTS["features"])
    assert plan.number_keys is True and plan.target_encode_max_cardinality == 700000


def test_heuristic_keeps_its_schedule_early_and_with_nothing_measured():
    profile = {"n_features": 24, "features_with_nulls": 0, "adversarial_auc": 0.5, "drift_after_drop": []}
    experiments = [{"cv": 0.95, "features": {"number_keys": True}}]
    early = {"round": 1, "profile": profile, "temporal": False, "experiments": experiments}
    assert HeuristicBrain().decide("features", early, CONTRACTS["features"]).number_keys is False
    empty = {"round": 12, "profile": profile, "temporal": False, "experiments": []}
    assert HeuristicBrain().decide("features", empty, CONTRACTS["features"]).target_encode_max_cardinality == 50


def test_heuristic_critic_holds_when_a_gate_is_closed():
    brain = HeuristicBrain()
    base = {"candidates": {"a": {"cv": 0.8, "std": 0.0}}, "best_submitted_cv": 0.79, "quota_left": 1,
            "min_gain": 0.0001, "canary_passed": True, "file_ok": True}
    assert brain.decide("critic", base, CONTRACTS["critic"]).action == "submit"
    for override in ({"quota_left": 0}, {"canary_passed": False}, {"file_ok": False}, {"best_submitted_cv": 0.80}):
        assert brain.decide("critic", {**base, **override}, CONTRACTS["critic"]).action == "hold"


def test_heuristic_uses_new_seeds_each_round():
    brain = HeuristicBrain()
    first = brain.decide("tuner", {"round": 0}, CONTRACTS["tuner"])
    second = brain.decide("tuner", {"round": 1}, CONTRACTS["tuner"])
    assert {m.seed for m in first.models}.isdisjoint({m.seed for m in second.models})
    assert {m.family for m in first.models} == {"lightgbm", "xgboost", "catboost"}
