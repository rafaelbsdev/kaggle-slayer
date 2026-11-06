from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from slayer.contracts import FeaturePlan, ModelParams, ModelPlan, ValidationPlan
from slayer import training
from slayer.data import CompetitionData, Schema, build_profile, load_competition
from slayer.training import (
    _BoostedLogistic,
    _encode_fold,
    _logistic_view,
    _target_encode_positions,
    apply_feature_plan,
    leak_canary,
    load_predictions,
    make_folds,
    NUMBER_KEY_KINDS,
    number_keys,
    recency_weights,
    run_experiment,
    save_predictions,
    target_encode,
    token_keys,
)

PARAMS = ModelParams(learning_rate=0.1, max_depth=4, num_leaves=15, subsample=0.9, colsample=0.8,
                     reg_lambda=1.0, min_child=5)
STRATIFIED = ValidationPlan(scheme="stratified_kfold", n_splits=3, rationale="Random folds, no drift here.")
TIME = ValidationPlan(scheme="time_kfold", n_splits=3, rationale="Time folds, the data drifts.")
PLAIN = FeaturePlan(rationale="Plain features as a baseline.")


def model(family="lightgbm", **overrides):
    return ModelPlan(**{"family": family, "params": PARAMS, "seed": 1, "hypothesis": "A small baseline model.",
                        **overrides})


@pytest.fixture
def iid(iid_raw, tmp_path):
    raw, hidden = iid_raw
    return load_competition(raw, tmp_path / "work"), hidden


# --- folds ----------------------------------------------------------------------------------
def test_stratified_folds_cover_every_row_once():
    y = np.array([0, 1] * 50)
    folds = make_folds(STRATIFIED, y, np.arange(100))
    assert len(folds) == 3
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val.tolist()) == list(range(100))


def test_time_folds_never_train_on_the_future():
    y = np.array([0, 1] * 50)
    for train, val in make_folds(TIME, y, np.arange(100)):
        assert train.max() < val.min()


def test_time_folds_can_order_by_id():
    plan = TIME.model_copy(update={"order_by": "id"})
    ids = np.arange(100)[::-1]
    for train, val in make_folds(plan, np.array([0, 1] * 50), ids):
        assert ids[train].max() < ids[val].min()


def test_time_folds_can_order_by_the_drift_clock():
    plan = TIME.model_copy(update={"order_by": "drift_clock"})
    time_score = np.random.default_rng(0).normal(size=100)
    for train, val in make_folds(plan, np.array([0, 1] * 50), np.arange(100), time_score):
        assert time_score[train].max() < time_score[val].min()


def test_drift_clock_order_refuses_a_missing_score():
    plan = TIME.model_copy(update={"order_by": "drift_clock"})
    with pytest.raises(ValueError, match="clock"):
        make_folds(plan, np.array([0, 1] * 50), np.arange(100))


def test_recency_weights_follow_the_drift_clock():
    time_score = np.array([0.3, 0.1, 0.2])
    weights = recency_weights(np.arange(3), 0.5, "drift_clock", time_score)
    assert weights[0] > weights[2] > weights[1]


def test_experiment_scores_the_rows_nearest_the_test_period(shuffled_temporal_raw, tmp_path):
    raw, _ = shuffled_temporal_raw
    data = load_competition(raw, tmp_path / "work")
    build_profile(data, seed=0)
    plan = TIME.model_copy(update={"order_by": "drift_clock"})
    result = run_experiment(plan, PLAIN, model(), data, max_trees=30, patience=10, device="cpu")
    order = np.argsort(data.time_score)
    assert result.mask[order[-100:]].all()
    assert not result.mask[order[:100]].any()
    assert leak_canary(plan, PLAIN, data, seed=0)["passed"] is True


def test_time_folds_headline_score_is_the_block_nearest_the_test_period(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    result = run_experiment(TIME, PLAIN, model(), data, max_trees=30, patience=10, device="cpu")
    assert result.mask.sum() == 600  # one block of 4, not every block pooled
    assert result.cv == pytest.approx(result.fold_scores[-1])


def test_base_rate_drift_across_blocks_does_not_inflate_the_score():
    # Noise features and a target rate that rises with time: there is no skill to measure.
    rng = np.random.default_rng(0)
    n = 12_000
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (rng.random(n) < np.linspace(0.05, 0.6, n)).astype(np.int8)
    features = [f"f{i}" for i in range(5)]
    data = CompetitionData(Schema("id", "y", "p", features), X, y, X[:100], np.arange(n), np.arange(100), None,
                           {f: n for f in features}, True)
    result = run_experiment(TIME, PLAIN, model(), data, max_trees=40, patience=20, device="cpu")
    assert abs(result.cv - 0.5) < 0.03


def test_recency_weights_are_rescaled_for_each_fit(temporal_raw, tmp_path, monkeypatch):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    seen = []
    real = training._fit

    def spying(model, X, y, weights, *args, **kwargs):
        seen.append(weights.max())
        return real(model, X, y, weights, *args, **kwargs)

    monkeypatch.setattr(training, "_fit", spying)
    run_experiment(TIME, PLAIN, model(recency_half_life=0.1), data, max_trees=10, patience=5, device="cpu")
    assert len(seen) == 4  # three folds and the refit
    assert seen == pytest.approx([1.0] * 4)


def test_lightgbm_early_stopping_follows_auc_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 6)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=1.5, size=4000) > 0).astype(np.int8)
    fitted, best = training._fit(model(), X[:3000], y[:3000], None, X[3000:], y[3000:], 300, 30, "cpu")
    history = fitted.evals_result_["valid_0"]
    assert list(history) == ["auc"]
    assert best == int(np.argmax(history["auc"])) + 1


def test_refuses_too_many_folds_for_the_rows():
    with pytest.raises(ValueError, match="folds"):
        make_folds(TIME, np.array([0, 1]), np.arange(2))


def test_recency_weights_favour_recent_rows_in_time_order():
    weights = recency_weights(ids=np.arange(100), half_life=0.5, order_by="row_order")
    assert weights[-1] == pytest.approx(1.0)
    assert weights[49] == pytest.approx(0.5, abs=0.02)
    assert (np.diff(weights) > 0).all()


# --- target encoding ------------------------------------------------------------------------
def test_target_encode_unseen_and_null_fall_back_to_global_mean():
    train_col = np.array([1.0, 1.0, 2.0, 2.0, np.nan, np.nan])
    y = np.array([1, 1, 0, 0, 1, 0])
    encoded = target_encode(train_col, y, np.array([1.0, 2.0, 3.0, np.nan]), smoothing=0.0)
    assert encoded[0] == pytest.approx(1.0)
    assert encoded[1] == pytest.approx(0.0)
    assert encoded[2] == pytest.approx(y.mean())
    assert encoded[3] == pytest.approx(0.5)


def test_target_encoding_of_training_rows_does_not_see_their_own_label():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    every_row_is_its_own_category = np.arange(1000, dtype=np.float32)[:, None]
    encoded, _ = _encode_fold(every_row_is_its_own_category, y, every_row_is_its_own_category[:10], [0])
    assert roc_auc_score(y, encoded[:, 1]) < 0.6


class PairsOfDigits:
    """Splits a number the way GPT-2 often does, so these tests run with no vocabulary file."""

    def encode(self, text):
        whole, _, decimals = text.strip().partition(".")
        tokens = [" " + whole[:2]] + [whole[i:i + 2] for i in range(2, len(whole), 2)]
        return SimpleNamespace(tokens=tokens + [".", decimals] if decimals else tokens)


def one_row_of_tokens(values):
    column = np.array(values, dtype=np.float32)
    keys, _ = token_keys(column, column[:1], PairsOfDigits())
    return {kind: keys[:, i] for i, kind in enumerate(training.TOKEN_KEY_KINDS)}


def test_token_keys_cut_a_number_the_way_a_language_model_wrote_it():
    keys = one_row_of_tokens([1234.35, 1288.35, 1234.5, 12.5, 7712.35])
    assert keys["first"][0] == keys["first"][1] == keys["first"][2] == keys["first"][3] != keys["first"][4]
    assert keys["second"][0] == keys["second"][2] != keys["second"][1]
    assert keys["last"][0] == keys["last"][1] == keys["last"][4] != keys["last"][2]
    assert keys["last"][2] == keys["last"][3]
    assert keys["count"][0] == keys["count"][1] != keys["count"][3]
    assert keys["first_with_count"][0] == keys["first_with_count"][1] != keys["first_with_count"][3]
    assert keys["before_point"][0] == keys["before_point"][2] != keys["before_point"][1]
    assert keys["before_point"][3] != keys["before_point"][0]


def test_token_keys_mean_the_same_on_train_and_test():
    train, test = token_keys(np.array([1234.35, 77.5], dtype=np.float32), np.array([3.25, 1234.35], dtype=np.float32), PairsOfDigits())
    assert np.array_equal(train[0], test[1])
    assert not np.array_equal(train[0], test[0])


def test_token_keys_read_a_null_as_its_own_value():
    train, _ = token_keys(np.array([1234.35, np.nan, 77.5], dtype=np.float32), np.array([3.25], dtype=np.float32), PairsOfDigits())
    assert np.isfinite(train).all()
    assert not np.array_equal(train[1], train[0])


@pytest.mark.parametrize("value, written", [(12.0, "12.0"), (12.5, "12.5"), (1234.35, "1234.35"), (0.07, "0.07")])
def test_a_number_is_written_as_a_csv_writes_it(value, written):
    assert training.written_form(np.float32(value)) == written


@pytest.mark.skipif(not training.tokenizer_available(), reason="the GPT-2 vocabulary file is not here")
def test_the_real_tokenizer_splits_a_number_as_expected():
    assert training._load_tokenizer().encode(" 1234.35").tokens == ["\u012012", "34", ".", "35"]


def test_a_missing_tokenizer_file_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "TOKENIZER_FILE", tmp_path / "absent.json")
    assert training.tokenizer_available() is False
    plan = FeaturePlan(number_tokens=True, rationale="Look for signal in how numbers were tokenized.")
    with pytest.raises(FileNotFoundError, match="README"):
        apply_feature_plan(plan, digit_signal_data(200))


def test_number_tokens_are_named_keys_and_get_encoded(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    data = digit_signal_data(2000)
    plan = FeaturePlan(number_tokens=True, rationale="Look for signal in how numbers were tokenized.")
    X, X_test, names = apply_feature_plan(plan, data)
    keys = [n for n in names if n.startswith("key_")]
    assert len(keys) == 2 * len(training.TOKEN_KEY_KINDS)  # this data has two decimal columns
    assert X.shape[1] == len(names) and X_test.shape[1] == len(names)
    assert [names[i] for i in _target_encode_positions(plan, data, names)] == keys


def test_number_tokens_add_to_number_keys(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    data = digit_signal_data(2000)
    plan = FeaturePlan(number_keys=True, number_tokens=True, rationale="Look for signal in how numbers were written.")
    _, _, names = apply_feature_plan(plan, data)
    assert len([n for n in names if n.startswith("key_")]) == 2 * (len(NUMBER_KEY_KINDS) + len(training.TOKEN_KEY_KINDS))


def test_number_tokens_find_a_signal_hidden_in_the_cents(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    whole = digit_signal_data(8000)
    data = CompetitionData(whole.schema, whole.X_train[:6000], whole.y[:6000], whole.X_train[6000:], np.arange(6000),
                           np.arange(2000), None, whole.n_unique, True)
    plan = FeaturePlan(number_tokens=True, rationale="Look for signal in how numbers were tokenized.")
    result = run_experiment(STRATIFIED, plan, model(), data, max_trees=60, patience=20, device="cpu")
    assert result.cv > 0.68
    assert roc_auc_score(whole.y[6000:], result.test) > 0.68
    assert leak_canary(STRATIFIED, plan, data, seed=0)["passed"] is True


def test_number_keys_cut_a_value_the_way_it_is_written():
    keys = number_keys(np.array([1371.37, 7.01, 0.5, np.nan], dtype=np.float32))
    assert keys.dtype == np.float32
    assert keys[0].tolist() == [1371, 137, 13713, 37, 41, 413]  # size 4, first digit 1, first two 13
    assert keys[1].tolist() == [7, 0, 70, 1, 17, 107]
    assert keys[2].tolist() == [0, 0, 5, 50, 10, 100]
    assert np.isnan(keys[3]).all()


def digit_signal_data(n=6000):
    """The label lives only in the cents, which a tree on the raw value cannot isolate."""
    rng = np.random.default_rng(0)
    cents = rng.integers(0, 100, n)
    values = (rng.integers(0, 5000, n) + cents / 100).astype(np.float32)
    noise = rng.normal(size=n).astype(np.float32)
    y = (rng.random(n) < np.where(cents % 10 == 3, 0.8, 0.1)).astype(np.int8)
    X = np.column_stack([values, noise])
    features = ["amount", "noise"]
    return CompetitionData(Schema("id", "y", "p", features), X, y, X[:50], np.arange(n), np.arange(50), None,
                           {"amount": n, "noise": n}, True)


def test_number_keys_find_a_signal_hidden_in_the_cents():
    data = digit_signal_data()
    plain = run_experiment(STRATIFIED, PLAIN, model(), data, max_trees=60, patience=20, device="cpu")
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    with_keys = run_experiment(STRATIFIED, plan, model(), data, max_trees=60, patience=20, device="cpu")
    assert plain.cv < 0.6
    assert with_keys.cv > 0.68  # a perfect model tops out near 0.72 with these label rates


def test_number_keys_pass_the_leak_canary():
    plan = FeaturePlan(number_keys=True, target_encode_bags=2, rationale="Look for signal in how numbers are written.")
    assert leak_canary(STRATIFIED, plan, digit_signal_data(), seed=0)["passed"] is True


def pair_signal_data(n=6000):
    """The label lives in whether `left` and `right` agree, which no column shows alone. `steady` adds up with them."""
    rng = np.random.default_rng(0)
    left, right, steady = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(left == right, 0.7, 0.1) * np.where(steady == 1, 1.0, 0.8)).astype(np.int8)
    amount = (rng.integers(0, 5000, n) + rng.integers(0, 100, n) / 100).astype(np.float32)
    X = np.column_stack([left, right, steady, amount]).astype(np.float32)
    features = ["left", "right", "steady", "amount"]
    return CompetitionData(Schema("id", "y", "p", features), X, y, X[:50], np.arange(n), np.arange(50), None,
                           {"left": 2, "right": 2, "steady": 2, "amount": n}, True)


JOINT = FeaturePlan(joint_keys=1, rationale="Key the pair of columns whose effect does not add up.")


def test_the_additive_gap_is_big_for_a_pair_that_interacts_and_small_for_one_that_adds_up():
    data = pair_signal_data()
    left, right, steady = data.X_train[:, 0].astype(int), data.X_train[:, 1].astype(int), data.X_train[:, 2].astype(int)
    gap, free = training.additive_gap(left, right, data.y)
    assert free == 1 and gap > 500
    gap, free = training.additive_gap(left, steady, data.y)
    assert free == 1 and gap < 10


def test_the_additive_gap_reads_training_rows_only():
    data = pair_signal_data()
    left, right = data.X_train[:, 0].astype(int), data.X_train[:, 1].astype(int)
    alone = training.additive_gap(left, right, data.y)
    spread = np.append(left * 2, np.ones(50, dtype=int))  # code 1 sits between the others and only test rows have it
    assert training.additive_gap(spread, np.append(right, right[:50]), data.y) == pytest.approx(alone)


def two_pairs_data(n=8000):
    """`a` with `b` carries a strong interaction and `c` with `d` a weaker one, and the weaker pair comes first."""
    rng = np.random.default_rng(1)
    a, b, c, d = (rng.integers(0, 2, n) for _ in range(4))
    y = (rng.random(n) < 0.2 + 0.5 * (a == b) + 0.1 * (c == d)).astype(np.int8)
    features = ["c", "d", "a", "b"]
    X = np.column_stack([c, d, a, b]).astype(np.float32)
    return CompetitionData(Schema("id", "y", "p", features), X, y, X[:50], np.arange(n), np.arange(50), None,
                           dict.fromkeys(features, 2), True)


def test_the_scan_finds_the_pair_that_interacts_and_only_that_one():
    assert training.non_additive_pairs(pair_signal_data()) == [("left", "right")]


def test_the_scan_ranks_the_strongest_pair_first():
    assert training.non_additive_pairs(two_pairs_data())[:2] == [("a", "b"), ("c", "d")]


def test_the_scan_skips_a_pair_with_no_freedom_left():
    rng = np.random.default_rng(2)
    n = 8000
    a = rng.integers(0, 2, n)
    implied = a * rng.integers(0, 2, n)  # three cells only, so what each column says alone already fixes the fourth
    y = (rng.random(n) < 0.1 + 0.7 * implied + 0.1 * a).astype(np.int8)
    X = np.column_stack([a, implied]).astype(np.float32)
    data = CompetitionData(Schema("id", "y", "p", ["a", "implied"]), X, y, X[:50], np.arange(n), np.arange(50), None,
                           {"a": 2, "implied": 2}, True)
    assert training.additive_gap(a, implied, y)[0] > 0
    assert training.non_additive_pairs(data) == []


def test_the_scan_runs_once_per_competition(monkeypatch):
    data = pair_signal_data()
    training.non_additive_pairs(data)
    monkeypatch.setattr(training, "additive_gap", lambda *args: pytest.fail("scanned again"))
    assert training.non_additive_pairs(data) == [("left", "right")]


def test_only_a_continuous_column_is_scanned_in_deciles():
    rng = np.random.default_rng(3)
    n = 3000
    amount = rng.integers(0, 5000, n) + rng.integers(1, 100, n) / 100
    whole = rng.integers(0, 60, n)
    few = rng.integers(0, 12, n) + 0.5
    features = ["amount", "whole", "few"]
    X = np.column_stack([amount, whole, few]).astype(np.float32)
    data = CompetitionData(Schema("id", "y", "p", features), X, np.zeros(n, dtype=np.int8), X[:50], np.arange(n), np.arange(50), None,
                           {"amount": n, "whole": 60, "few": 12}, True)
    assert len(np.unique(training._scan_codes(data, "amount"))) == 10
    assert len(np.unique(training._scan_codes(data, "whole"))) == 60
    assert len(np.unique(training._scan_codes(data, "few"))) == 12


def test_joint_keys_are_named_keys_and_get_encoded():
    data = pair_signal_data()
    X, X_test, names = apply_feature_plan(JOINT, data)
    assert names == data.schema.features + ["key_joint_left_right"]
    assert X.shape[1] == X_test.shape[1] == len(names)
    assert _target_encode_positions(JOINT, data, names) == [4]


def test_joint_keys_mean_the_same_on_train_and_test():
    data = pair_signal_data()
    X, X_test, _ = apply_feature_plan(JOINT, data)
    assert np.array_equal(X_test[:, 4], X[:50, 4])  # the test rows repeat the first training rows here
    assert len(np.unique(X[:, 4])) == 4


def test_joint_keys_stop_at_what_the_scan_found():
    plan = FeaturePlan(joint_keys=5, rationale="Ask for more pairs than the data has.")
    assert [name for name in apply_feature_plan(plan, pair_signal_data())[2] if name.startswith("key_joint")] == ["key_joint_left_right"]


def test_joint_keys_take_the_strongest_pairs_up_to_what_the_plan_asks():
    assert apply_feature_plan(JOINT, two_pairs_data())[2][-1] == "key_joint_a_b"
    assert sum(name.startswith("key_joint") for name in apply_feature_plan(JOINT, two_pairs_data())[2]) == 1


def by_value_data(n=12000):
    """`flag` interacts with a decimal column of 100 values and with one of thousands."""
    rng = np.random.default_rng(4)
    flag = rng.integers(0, 2, n)
    few = rng.integers(0, 100, n) / 10 + 0.05
    many = rng.integers(0, 5000, n) + rng.integers(1, 100, n) / 100
    y = (rng.random(n) < 0.1 + 0.3 * ((few > 5) == flag) + 0.3 * ((many > 2500) == flag)).astype(np.int8)
    features = ["few", "many", "flag"]
    X = np.column_stack([few, many, flag]).astype(np.float32)
    return CompetitionData(Schema("id", "y", "p", features), X, y, X[:50], np.arange(n), np.arange(50), None,
                           {"few": 100, "many": n, "flag": 2}, True)


@pytest.mark.parametrize("by_value, few_cells", [(False, 20), (True, 200)])
def test_joint_by_value_keys_each_value_of_a_column_with_few_and_still_cuts_one_with_many(by_value, few_cells):
    plan = FeaturePlan(joint_keys=2, joint_by_value=by_value, rationale="Key the pairs with this column cut or by value.")
    X, _, names = apply_feature_plan(plan, by_value_data())
    assert {"key_joint_few_flag", "key_joint_many_flag"} <= set(names)
    assert len(np.unique(X[:, names.index("key_joint_few_flag")])) == few_cells
    assert len(np.unique(X[:, names.index("key_joint_many_flag")])) == 20


def test_joint_by_value_does_not_change_which_pairs_are_scanned(monkeypatch):
    data = by_value_data()
    pairs = training.non_additive_pairs(data)
    plan = FeaturePlan(joint_keys=1, joint_by_value=True, rationale="Key only the strongest pair, by value.")
    assert apply_feature_plan(plan, data)[2][-1] == f"key_joint_{pairs[0][0]}_{pairs[0][1]}"


def test_joint_keys_leave_out_a_pair_with_a_dropped_column():
    data = pair_signal_data()
    data.drift_rank = ["left", "right", "steady", "amount"]
    plan = FeaturePlan(joint_keys=1, drop_drifting_top_k=1, rationale="Drop the most drifting column first.")
    assert not any(name.startswith("key_joint") for name in apply_feature_plan(plan, data)[2])


def test_joint_keys_give_a_logistic_model_what_it_cannot_express():
    data = pair_signal_data()
    plain = run_experiment(STRATIFIED, PLAIN, model("logistic"), data, max_trees=60, patience=20, device="cpu")
    with_key = run_experiment(STRATIFIED, JOINT, model("logistic"), data, max_trees=60, patience=20, device="cpu")
    assert plain.cv < 0.6
    assert with_key.cv > 0.78


def test_joint_keys_pass_the_leak_canary():
    assert leak_canary(STRATIFIED, JOINT, pair_signal_data(), seed=0)["passed"] is True


COUNTED = FeaturePlan(number_keys=True, key_counts=True, rationale="Show trees how many rows stand behind each key.")


def test_key_counts_tell_how_many_rows_share_each_value_in_train_and_test():
    data = digit_signal_data()
    X, X_test, names = apply_feature_plan(COUNTED, data)
    key, count = names.index("key_amount_decimals"), names.index("count_key_amount_decimals")
    values, counts = np.unique(np.concatenate([X[:, key], X_test[:, key]]), return_counts=True)
    assert X[7, count] == counts[np.searchsorted(values, X[7, key])]
    assert X_test[7, count] == counts[np.searchsorted(values, X_test[7, key])]


def test_key_counts_are_not_encoded_and_a_linear_model_leaves_them_out():
    data = digit_signal_data()
    plan = FeaturePlan(number_keys=True, rationale="The same keys without their counts.")
    X, _, names = apply_feature_plan(plan, data)
    X_counted, _, counted_names = apply_feature_plan(COUNTED, data)
    assert len(counted_names) == 2 * len(names) - len(data.schema.features)
    encoded = _target_encode_positions(COUNTED, data, counted_names)
    assert encoded == _target_encode_positions(plan, data, names)
    assert _logistic_view(data, counted_names)(X_counted).shape[1] == _logistic_view(data, names)(X).shape[1]


def test_key_counts_cover_joint_keys():
    plan = FeaturePlan(joint_keys=1, key_counts=True, rationale="Count the rows behind the pair of columns too.")
    assert "count_key_joint_left_right" in apply_feature_plan(plan, pair_signal_data())[2]


def test_key_counts_with_no_keys_add_nothing():
    data = pair_signal_data()
    plan = FeaturePlan(key_counts=True, rationale="Counts asked for on a plan with no keys.")
    assert apply_feature_plan(plan, data)[2] == data.schema.features


def test_key_counts_skip_a_real_column_that_only_looks_like_a_key():
    data = digit_signal_data()
    data.schema.features = ["key_amount", "noise"]
    data.n_unique = {"key_amount": 6000, "noise": 6000}
    assert "count_key_amount" not in apply_feature_plan(FeaturePlan(key_counts=True, rationale="No keys were made here."), data)[2]


SHARED = FeaturePlan(shared_token_rates=True, rationale="Pool decimal tokens over fields written alike.")


def shared_tokens_of(monkeypatch, columns, few=()):
    """Columns are lists of numbers, written as a CSV writes them. Fields in `few` count as having few distinct values."""
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    features = list(columns)
    X = np.column_stack([np.array(columns[c], dtype=np.float32) for c in features])
    n = len(X)
    data = CompetitionData(Schema("id", "y", "p", features), X, np.zeros(n, dtype=np.int8), X[:1], np.arange(n), np.arange(1), None,
                           {c: 100 if c in few else 2000 for c in features}, True)
    return training._SharedTokens(data, features, X, X[:1])


def test_a_rare_token_leans_on_what_other_fields_say_about_it(monkeypatch):
    # "35" is rare in `a`, common and always positive in `b`. Everything else is "50".
    a = [1.35] + [1.5] * 999
    b = [2.35] * 500 + [2.5] * 500
    y = np.array([0] + [1] * 499 + [0] * 500)
    shared = shared_tokens_of(monkeypatch, {"a": a, "b": b})
    rows = np.arange(1000)
    rate = shared.rates(rows, y, rows[:1])[0, 0]
    prior = y.mean()
    pooled = (0 + 499 + 50 * prior) / (1 + 500 + 50)
    assert rate == pytest.approx((0 + 2000 * pooled) / (1 + 2000), rel=1e-5)
    assert rate > 1.5 * prior  # its one sighting in `a` was negative, and `b` still lifts it


def test_fields_with_few_values_pool_apart_from_those_with_many(monkeypatch):
    many = [2.35] * 500 + [2.5] * 500
    few = [1.35] + [1.5] * 999
    y = np.array([0] + [1] * 499 + [0] * 500)
    shared = shared_tokens_of(monkeypatch, {"many": many, "few": few}, few={"few"})
    rows = np.arange(1000)
    width = shared.rates(rows, y, rows[:1]).shape[1]
    assert width == 2 + 1 + 2  # decimals and first decimal, one chain, then decimals and last whole token
    few_rate = shared.rates(rows, y, rows[:1])[0, 2]
    assert few_rate < y.mean()  # its only sighting of "35" was negative, and `many` has no say


def test_a_token_chain_leans_on_its_fields_rate(monkeypatch):
    a = [12.35] * 10 + [34.35] * 490 + [34.5] * 500
    y = np.array([1] * 300 + [0] * 700)  # "35" is far above the base rate, so leaning on the field is not leaning on the base rate
    shared = shared_tokens_of(monkeypatch, {"a": a})
    rows = np.arange(1000)
    decimals, _, chain = shared.rates(rows, y, rows[:1])[0]
    assert decimals > 1.5 * y.mean()
    assert chain == pytest.approx((10 + 200 * decimals) / (10 + 200), rel=1e-5)


def test_the_first_decimal_pools_tokens_that_share_it(monkeypatch):
    a = [1.31] * 500 + [1.82] * 500
    y = np.array([1] * 250 + [0] * 750)
    shared = shared_tokens_of(monkeypatch, {"a": a, "b": [1.39] * 1000})
    assert shared.groups[1][0, 0] == shared.groups[1][0, 1]
    assert shared.groups[0][0, 0] != shared.groups[0][0, 1]
    assert shared.groups[1][0, 0] != shared.groups[1][999, 0]


def test_the_last_whole_token_pools_fields_with_few_values(monkeypatch):
    shared = shared_tokens_of(monkeypatch, {"a": [1234.5, 77.5], "b": [9934.25, 88.25]}, few={"a", "b"})
    assert shared.groups[1][0, 0] == shared.groups[1][0, 1]
    assert shared.groups[1][0, 0] != shared.groups[1][1, 0]


def test_shared_rates_ask_for_test_rows_after_the_train_rows(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    X = np.array([[1.35]] * 500 + [[1.5]] * 500, dtype=np.float32)
    X_test = np.array([[9.5], [9.35]], dtype=np.float32)
    data = CompetitionData(Schema("id", "y", "p", ["a"]), X, np.zeros(1000, dtype=np.int8), X_test, np.arange(1000), np.arange(2), None, {"a": 2000}, True)
    shared = training._SharedTokens(data, ["a"], X, X_test)
    y = np.array([1] * 500 + [0] * 500)
    fifty, thirty_five = shared.rates(np.arange(1000), y, np.array([1000, 1001]))[:, 0]
    assert thirty_five > 0.5 > fifty


def test_shared_rates_of_training_rows_cannot_see_their_own_label(monkeypatch):
    rng = np.random.default_rng(0)
    n = 600
    a = rng.integers(0, 90, n) + rng.integers(1, 100, n) / 100
    y = rng.integers(0, 2, n).astype(np.int8)
    shared = shared_tokens_of(monkeypatch, {"a": a})
    rows = np.arange(n)
    assert roc_auc_score(y, shared.rates(rows, y, rows)[:, 0]) > 0.6  # what seeing its own label would look like
    assert roc_auc_score(y, shared.out_of_fold(rows, y, 1, 0, 3)[:, 0]) < 0.53


def test_shared_rates_average_over_the_plans_bags_with_the_seed(monkeypatch):
    rng = np.random.default_rng(1)
    n = 300
    a = rng.integers(0, 90, n) + rng.integers(1, 100, n) / 100
    y = rng.integers(0, 2, n).astype(np.int8)
    shared = shared_tokens_of(monkeypatch, {"a": a})
    rows = np.arange(n)
    bagged = shared.out_of_fold(rows, y, 2, 5, 3)
    alone = (shared.out_of_fold(rows, y, 1, 5, 3) + shared.out_of_fold(rows, y, 1, 6, 3)) / 2
    assert np.allclose(bagged, alone, atol=1e-6)
    assert not np.allclose(shared.out_of_fold(rows, y, 1, 5, 3), shared.out_of_fold(rows, y, 1, 5, 2), atol=1e-6)


def decimal_signal_data(n=8000, held=2000):
    """The last two decimals decide the label in both fields, so nothing but how they are written can find it."""
    rng = np.random.default_rng(0)
    total = n + held
    cents = rng.integers(1, 100, (total, 2))
    values = (rng.integers(0, 5000, (total, 2)) + cents / 100).astype(np.float32)
    y = (rng.random(total) < 0.1 + 0.35 * (cents % 10 == 3).sum(axis=1)).astype(np.int8)
    features = ["amount", "balance"]
    data = CompetitionData(Schema("id", "y", "p", features), values[:n], y[:n], values[n:], np.arange(n), np.arange(held), None,
                           {"amount": n, "balance": n}, True)
    return data, y[n:]


@pytest.mark.parametrize("validation", [STRATIFIED, TIME])
@pytest.mark.parametrize("family", ["logistic", "logistic_boost", "lightgbm"])
def test_shared_rates_reach_the_models_and_the_test_side(monkeypatch, validation, family):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    data, y_test = decimal_signal_data()
    result = run_experiment(validation, SHARED, model(family), data, max_trees=60, patience=20, device="cpu")
    assert result.cv > 0.6
    assert roc_auc_score(y_test, result.test) > 0.6


def test_shared_rates_on_data_with_no_decimals_add_nothing(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    data = pair_signal_data()
    data.X_train, data.X_test = data.X_train[:, :3], data.X_test[:, :3]
    data.schema.features = ["left", "right", "steady"]
    shared = training._shared_tokens(SHARED, data, data.schema.features, data.X_train, data.X_test)
    rows = np.arange(len(data.y))
    assert shared.out_of_fold(rows, data.y, 1, 0, 3).shape == (len(rows), 0)


def test_shared_rates_leave_out_a_dropped_field(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    data, _ = decimal_signal_data()
    data.drift_rank = ["amount", "balance"]
    plan = FeaturePlan(shared_token_rates=True, drop_drifting_top_k=1, rationale="Drop the most drifting field first.")
    X, X_test, names = apply_feature_plan(plan, data)
    assert training._shared_tokens(plan, data, names, X, X_test).groups[0].shape[1] == 1


def test_a_plan_without_shared_rates_builds_none():
    assert training._shared_tokens(PLAIN, digit_signal_data(), ["amount", "noise"], None, None) is None


def test_shared_rates_pass_the_leak_canary(monkeypatch):
    monkeypatch.setattr(training, "_load_tokenizer", PairsOfDigits)
    data, _ = decimal_signal_data()
    seen = []
    real = training._add_shared_rates

    def spying(shared, *args, **kwargs):
        seen.append(shared)
        return real(shared, *args, **kwargs)

    monkeypatch.setattr(training, "_add_shared_rates", spying)
    assert leak_canary(STRATIFIED, SHARED, data, seed=0)["passed"] is True
    assert seen and all(shared is not None for shared in seen)  # the canary has to run what the plan will run


def test_number_keys_skip_categoricals_whole_numbers_and_low_cardinality_columns():
    rng = np.random.default_rng(0)
    features = ["amount", "many_codes", "few_values", "age"]
    X = rng.normal(size=(200, 4)).astype(np.float32)
    X[:, 3] = rng.integers(18, 90, 200)
    data = CompetitionData(Schema("id", "y", "p", features, categoricals=["many_codes"]), X,
                           rng.integers(0, 2, 200).astype(np.int8), X[:10], np.arange(200), np.arange(10), None,
                           {"amount": 200, "many_codes": 200, "few_values": 3, "age": 70}, True)
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    _, _, names = apply_feature_plan(plan, data)
    assert "key_amount_decimals" in names
    assert not any(name.startswith(("key_many_codes", "key_few_values", "key_age")) for name in names)


def test_number_keys_are_all_target_encoded(iid):
    data, _ = iid
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    _, _, names = apply_feature_plan(plan, data)
    keys = [i for i, name in enumerate(names) if name.startswith("key_")]
    assert keys and set(keys) <= set(_target_encode_positions(plan, data, names))


def test_number_keys_are_capped_on_wide_data():
    rng = np.random.default_rng(0)
    features = [f"f{i}" for i in range(50)]
    X = rng.normal(size=(200, 50)).astype(np.float32)
    data = CompetitionData(Schema("id", "y", "p", features), X, rng.integers(0, 2, 200).astype(np.int8), X[:10],
                           np.arange(200), np.arange(10), None, {f: 200 for f in features}, True)
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    _, _, names = apply_feature_plan(plan, data)
    assert sum(name.endswith("_decimals") for name in names) == training.MAX_KEYED_COLUMNS


def test_bagged_encoding_still_cannot_see_a_rows_own_label():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    unique = np.arange(1000, dtype=np.float32)[:, None]
    encoded, _ = _encode_fold(unique, y, unique[:10], [0], bags=3)
    assert encoded.shape == (1000, 2)
    assert roc_auc_score(y, encoded[:, 1]) < 0.6


def test_bagging_averages_over_different_inner_splits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    values = rng.integers(0, 20, 1000).astype(np.float32)[:, None]
    one, _ = _encode_fold(values, y, values[:10], [0], bags=1)
    three, _ = _encode_fold(values, y, values[:10], [0], bags=3)
    assert not np.allclose(one[:, 1], three[:, 1], atol=1e-4)
    assert np.corrcoef(one[:, 1], three[:, 1])[0, 1] > 0.5
    assert three[:, 1].max() <= 1  # still a mean of the target, not a sum over bags


def test_the_seed_changes_the_encoding_noise_of_training_rows_only():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    values = rng.integers(0, 20, 1000).astype(np.float32)[:, None]
    first, first_other = _encode_fold(values, y, values[:10], [0], seed=1)
    second, second_other = _encode_fold(values, y, values[:10], [0], seed=2)
    assert not np.allclose(first[:, 1], second[:, 1], atol=1e-4)
    assert np.array_equal(first_other, second_other)


@pytest.mark.parametrize("folds", [2, 3, 7])
def test_inner_folds_set_how_the_training_rows_are_encoded(folds):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    one_value = np.ones((1000, 1), dtype=np.float32)
    encoded, _ = _encode_fold(one_value, y, one_value[:10], [0], inner_folds=folds)
    assert len(np.unique(encoded[:, 1])) == folds  # each inner fold gives this one value one encoding


def test_inner_folds_reach_the_models(iid):
    data, _ = iid
    plan = FeaturePlan(target_encode_max_cardinality=100000, rationale="Encode every value.")
    fewer = plan.model_copy(update={"target_encode_inner_folds": 2})
    five = run_experiment(STRATIFIED, plan, model(), data, max_trees=30, patience=20, device="cpu")
    two = run_experiment(STRATIFIED, fewer, model(), data, max_trees=30, patience=20, device="cpu")
    assert not np.allclose(five.oof, two.oof)


def test_leak_canary_runs_the_plans_inner_folds(monkeypatch):
    seen = []
    real = training._encode_fold

    def recording(*args, **kwargs):
        seen.append(kwargs.get("inner_folds"))
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "_encode_fold", recording)
    plan = FeaturePlan(number_keys=True, target_encode_inner_folds=3, rationale="Look for signal in how numbers are written.")
    assert leak_canary(STRATIFIED, plan, digit_signal_data(), seed=0)["passed"] is True
    assert seen and set(seen) == {3}


def test_smoothing_sets_how_hard_encodings_lean_on_the_base_rate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    values = rng.integers(0, 20, 1000).astype(np.float32)[:, None]
    light, light_other = _encode_fold(values, y, values[:10], [0], smoothing=1.0)
    heavy, heavy_other = _encode_fold(values, y, values[:10], [0], smoothing=1e6)
    assert heavy[:, 1].std() < light[:, 1].std() / 2
    assert np.abs(heavy_other[:, 1] - y.mean()).max() < 1e-3 < np.abs(light_other[:, 1] - y.mean()).max()
    outside = training._encode_other(values, y, values[:10], [0], smoothing=1e6)
    assert np.array_equal(outside, heavy_other)


def test_smoothing_stays_at_its_old_default_and_must_be_sane():
    assert FeaturePlan(rationale="Encode every value.").target_encode_smoothing == 20.0
    with pytest.raises(ValueError):
        FeaturePlan(target_encode_smoothing=0.5, rationale="Encode every value.")
    with pytest.raises(ValueError):
        FeaturePlan(target_encode_smoothing=100001.0, rationale="Encode every value.")


@pytest.fixture
def smoothing_seen(monkeypatch):
    seen = set()
    real = training.target_encode

    def recording(train_col, y, other_col, smoothing=20.0):
        seen.add(smoothing)
        return real(train_col, y, other_col, smoothing)

    monkeypatch.setattr(training, "target_encode", recording)
    return seen


def test_every_encoding_in_a_run_uses_the_plans_smoothing(iid, smoothing_seen):
    data, _ = iid
    plan = FeaturePlan(target_encode_max_cardinality=100000, target_encode_smoothing=500.0, rationale="Encode every value.")
    run_experiment(STRATIFIED, plan, model(), data, max_trees=30, patience=20, device="cpu")
    assert smoothing_seen == {500.0}


def test_leak_canary_runs_the_plans_smoothing(smoothing_seen):
    plan = FeaturePlan(number_keys=True, target_encode_smoothing=500.0, rationale="Look for signal in how numbers are written.")
    assert leak_canary(STRATIFIED, plan, digit_signal_data(), seed=0)["passed"] is True
    assert smoothing_seen == {500.0}


def test_target_encode_smoothing_pulls_to_global_mean():
    train_col = np.array([1.0, 2.0, 2.0, 2.0])
    y = np.array([1, 0, 0, 0])
    encoded = target_encode(train_col, y, np.array([1.0]), smoothing=4.0)
    assert 0.25 < encoded[0] < 1.0


# --- features -------------------------------------------------------------------------------
def test_feature_plan_row_stats_and_frequency_add_columns(iid):
    data, _ = iid
    plan = FeaturePlan(row_stats=True, frequency_encode_max_cardinality=10, rationale="Add simple row features.")
    X_train, X_test, names = apply_feature_plan(plan, data)
    assert X_train.shape[0] == 2400 and X_test.shape[0] == 800
    assert "row_null_count" in names and "freq_cat" in names
    assert X_train.shape[1] == len(names)
    assert X_train.dtype == np.float32


def test_frequency_encoding_counts_train_and_test(iid):
    data, _ = iid
    plan = FeaturePlan(frequency_encode_max_cardinality=10, rationale="Count encode low cardinality.")
    X_train, X_test, names = apply_feature_plan(plan, data)
    column = names.index("freq_cat")
    assert X_train[:, column].max() <= 3200
    assert X_train[:, column].min() >= 1


def test_feature_plan_drops_most_drifting_first(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    data.drift_rank = ["f5", "f1"] + [f for f in data.schema.features if f not in ("f5", "f1")]
    plan = FeaturePlan(drop_drifting_top_k=1, rationale="Drop the most drifting feature.")
    _, _, names = apply_feature_plan(plan, data)
    assert "f5" not in names and "f1" in names


def test_feature_plan_never_uses_id_or_target(iid):
    data, _ = iid
    plan = FeaturePlan(row_stats=True, frequency_encode_max_cardinality=5000, rationale="Add everything we can.")
    _, _, names = apply_feature_plan(plan, data)
    assert data.schema.id_col not in names and data.schema.target_col not in names


# --- experiments ----------------------------------------------------------------------------
def test_logistic_family_handles_nulls_time_folds_and_weights(temporal_raw, tmp_path):
    raw, hidden = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    result = run_experiment(TIME, PLAIN, model("logistic", recency_half_life=0.5), data, max_trees=10, patience=5,
                            device="cpu")
    assert result.cv > 0.6
    assert roc_auc_score(hidden, result.test) > 0.6
    unweighted = run_experiment(TIME, PLAIN, model("logistic"), data, max_trees=10, patience=5, device="cpu")
    assert not np.allclose(result.test, unweighted.test)


def coded_data(features, categoricals, X_test=None):
    X = np.array([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)
    X_test = X[:1] if X_test is None else X_test
    return CompetitionData(Schema("id", "y", "p", features, categoricals=categoricals), X, np.array([0, 1], dtype=np.int8),
                           X_test, np.arange(2), np.arange(len(X_test)), None, {f: 2 for f in features}, True)


def test_logistic_view_adds_encodings_as_log_odds_and_leaves_labels_out():
    data = coded_data(["amount", "code"], ["code"])
    names = ["amount", "code", "key_amount_decimals"]
    X = np.array([[np.e - 1, 2.0, 37.0, 0.5, 0.1, 0.0],
                  [-(np.e - 1), np.nan, 12.0, 0.0, 0.1, 1.0]], dtype=np.float32)
    inputs = _logistic_view(data, names)(X)
    assert inputs.shape == (2, 7)  # one number, three encodings, three categories: no raw key, no category code
    assert inputs[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert inputs[1, 0] == pytest.approx(-1.0, abs=1e-5)
    assert inputs[0, 1] == pytest.approx(0.0, abs=1e-5)  # a rate of one half is even odds
    assert inputs[0, 2] == pytest.approx(np.log(0.1 / 0.9), abs=1e-5)
    assert np.isfinite(inputs).all()  # rates of 0 and 1 are clipped, not infinite
    assert inputs[0, 4:].tolist() == [0, 0, 1]  # code 2 of 3
    assert inputs[1, 4:].tolist() == [0, 0, 0]  # a null marks no category


def test_logistic_view_counts_categories_that_only_test_has():
    data = coded_data(["amount", "code"], ["code"], X_test=np.array([[1.0, 5.0]], dtype=np.float32))
    X = np.array([[1.0, 5.0, 0.5]], dtype=np.float32)
    assert _logistic_view(data, ["amount", "code"])(X)[0, -1] == 1


def test_logistic_view_leaves_a_huge_categorical_to_its_encoding():
    data = coded_data(["amount", "code"], ["code"], X_test=np.array([[1.0, float(training.MAX_ONE_HOT_LEVELS)]], dtype=np.float32))
    X = np.array([[1.0, 3.0, 0.5]], dtype=np.float32)
    assert _logistic_view(data, ["amount", "code"])(X).shape == (1, 2)  # the number and its encoding, no code


def test_logistic_view_keeps_a_real_column_that_only_looks_like_a_key():
    data = coded_data(["key_code", "amount"], [])
    X = np.array([[3.0, 1.0]], dtype=np.float32)
    assert _logistic_view(data, ["key_code", "amount"])(X).shape == (1, 2)


def test_the_logistic_family_reads_through_its_view(monkeypatch):
    data = digit_signal_data(3000)
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    X, _, names = apply_feature_plan(plan, data)
    untouched = X.shape[1] + len(_target_encode_positions(plan, data, names))
    widths = []
    real = training._fit

    def spying(model, X, y, *args, **kwargs):
        widths.append(X.shape[1])
        return real(model, X, y, *args, **kwargs)

    monkeypatch.setattr(training, "_fit", spying)
    run_experiment(STRATIFIED, plan, model("logistic"), data, max_trees=10, patience=5, device="cpu")
    assert len(widths) == 4
    assert all(width < untouched for width in widths)  # raw keys are labels, so they stay out


def test_logistic_family_reads_a_signal_only_the_encodings_carry():
    whole = digit_signal_data(8000)
    data = CompetitionData(whole.schema, whole.X_train[:6000], whole.y[:6000], whole.X_train[6000:], np.arange(6000),
                           np.arange(2000), None, whole.n_unique, True)
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    result = run_experiment(STRATIFIED, plan, model("logistic"), data, max_trees=10, patience=5, device="cpu")
    assert result.cv > 0.68
    assert roc_auc_score(whole.y[6000:], result.test) > 0.68
    plain = run_experiment(STRATIFIED, PLAIN, model("logistic"), data, max_trees=10, patience=5, device="cpu")
    assert plain.cv < 0.6


def additive_data(n=4000, columns=12):
    """A label that adds up many small effects: one small tree cannot hold it, and a linear model can."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, columns)).astype(np.float32)
    y = (X.sum(axis=1) + rng.normal(scale=1.0, size=n) > 0).astype(np.int8)
    features = [f"f{i}" for i in range(columns)]
    return CompetitionData(Schema("id", "y", "p", features), X[:n - 500], y[:n - 500], X[n - 500:], np.arange(n - 500),
                           np.arange(500), None, {f: n for f in features}, True), y[n - 500:]


def test_logistic_boost_starts_from_the_linear_model():
    data, hidden = additive_data()
    kept_small = {"max_trees": 1, "patience": 1, "device": "cpu"}
    alone = run_experiment(STRATIFIED, PLAIN, model("xgboost"), data, **kept_small)
    boosted = run_experiment(STRATIFIED, PLAIN, model("logistic_boost"), data, **kept_small)
    assert alone.cv < 0.7
    assert boosted.cv > 0.85
    assert roc_auc_score(hidden, boosted.test) > 0.85


def test_logistic_boost_starts_from_what_the_logistic_family_reads(monkeypatch):
    data = digit_signal_data(3000)
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    X, _, names = apply_feature_plan(plan, data)
    untouched = X.shape[1] + len(_target_encode_positions(plan, data, names))
    widths = []
    real = training._logistic

    def spying(reg_lambda, X, y, weights, tolerance):
        widths.append(X.shape[1])
        return real(reg_lambda, X, y, weights, tolerance)

    monkeypatch.setattr(training, "_logistic", spying)
    run_experiment(STRATIFIED, plan, model("logistic_boost"), data, max_trees=2, patience=1, device="cpu")
    assert len(widths) == 4
    assert all(width < untouched for width in widths)  # through the view: raw keys are labels, so they stay out


def test_logistic_boost_early_stops_on_scores_that_include_the_linear_model(monkeypatch):
    data, _ = additive_data()
    seen = []
    real = training.xgb.XGBClassifier.fit

    def spying(self, X, y, **kwargs):
        seen.append(kwargs)
        return real(self, X, y, **kwargs)

    monkeypatch.setattr(training.xgb.XGBClassifier, "fit", spying)
    run_experiment(STRATIFIED, PLAIN, model("logistic_boost"), data, max_trees=3, patience=2, device="cpu")
    assert len(seen) == 4
    assert all("base_margin" in kwargs for kwargs in seen)
    assert all("base_margin_eval_set" in kwargs for kwargs in seen[:3])
    assert "base_margin_eval_set" not in seen[3] and "eval_set" not in seen[3]


def test_boosted_trees_predict_through_the_logit():
    data, hidden = additive_data()
    fitted, _ = training._fit(model("logistic_boost"), data.X_train, data.y, None, None, None, 1, 1, "cpu", fixed_trees=1,
                              linear_view=_logistic_view(data, data.schema.features))
    assert isinstance(fitted, _BoostedLogistic)
    assert roc_auc_score(hidden, training._predict(fitted, data.X_test)) > 0.85


def test_logistic_tolerance_stays_at_the_solver_default_and_must_be_sane():
    assert FeaturePlan(rationale="Encode every value.").logistic_tolerance == 1e-4
    with pytest.raises(ValueError):
        FeaturePlan(logistic_tolerance=1e-8, rationale="Encode every value.")
    with pytest.raises(ValueError):
        FeaturePlan(logistic_tolerance=0.1, rationale="Encode every value.")


@pytest.mark.parametrize("family", ["logistic", "logistic_boost"])
@pytest.mark.parametrize("tolerance", [1e-4, 1e-6])
def test_logistic_tolerance_reaches_every_fit_of_a_logistic_family(iid, monkeypatch, family, tolerance):
    data, _ = iid
    seen = []
    real = training.LogisticRegression

    def spying(**kwargs):
        seen.append(kwargs["tol"])
        return real(**kwargs)

    monkeypatch.setattr(training, "LogisticRegression", spying)
    plan = FeaturePlan(logistic_tolerance=tolerance, rationale="Let the solver run to its optimum.")
    run_experiment(STRATIFIED, plan, model(family), data, max_trees=3, patience=2, device="cpu")
    assert seen == [tolerance] * 4


def test_logistic_start_penalty_stays_at_its_old_value_and_must_be_sane():
    assert FeaturePlan(rationale="Encode every value.").logistic_start_penalty == training.START_REG_LAMBDA == 1.0
    with pytest.raises(ValueError):
        FeaturePlan(logistic_start_penalty=0.001, rationale="Encode every value.")
    with pytest.raises(ValueError):
        FeaturePlan(logistic_start_penalty=2000.0, rationale="Encode every value.")


@pytest.mark.parametrize("family, penalty, expected_c", [("logistic_boost", 1.0, 1.0), ("logistic_boost", 10.0, 0.1), ("logistic", 10.0, 1.0)])
def test_logistic_start_penalty_reaches_every_start_and_leaves_the_plain_family_alone(iid, monkeypatch, family, penalty, expected_c):
    data, _ = iid
    seen = []
    real = training.LogisticRegression

    def spying(**kwargs):
        seen.append(kwargs["C"])
        return real(**kwargs)

    monkeypatch.setattr(training, "LogisticRegression", spying)
    plan = FeaturePlan(logistic_start_penalty=penalty, rationale="Penalize the start harder.")
    run_experiment(STRATIFIED, plan, model(family), data, max_trees=3, patience=2, device="cpu")
    assert seen == pytest.approx([expected_c] * 4)


def test_logistic_boost_reads_a_signal_only_the_encodings_carry():
    whole = digit_signal_data(8000)
    data = CompetitionData(whole.schema, whole.X_train[:6000], whole.y[:6000], whole.X_train[6000:], np.arange(6000),
                           np.arange(2000), None, whole.n_unique, True)
    plan = FeaturePlan(number_keys=True, rationale="Look for signal in how numbers are written.")
    result = run_experiment(STRATIFIED, plan, model("logistic_boost"), data, max_trees=60, patience=20, device="cpu")
    assert result.cv > 0.68
    assert roc_auc_score(whole.y[6000:], result.test) > 0.68


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost", "logistic", "logistic_boost"])
def test_each_family_learns_with_observed_scores(iid, family):
    data, hidden = iid
    result = run_experiment(STRATIFIED, PLAIN, model(family), data, max_trees=60, patience=20, device="cpu")
    assert result.cv > 0.75
    assert result.cv == pytest.approx(roc_auc_score(data.y, result.oof), abs=1e-9)
    assert len(result.fold_scores) == 3 and result.std >= 0
    assert result.mask.all()
    assert roc_auc_score(hidden, result.test) > 0.7


def test_random_folds_also_refit_on_every_row(iid, monkeypatch):
    data, hidden = iid
    fits = []
    real = training._fit

    def spying(model, X, y, weights, X_val, y_val, max_trees, patience, device, fixed_trees=None, linear_view=None, logistic_tolerance=1e-4, start_penalty=1.0):
        fits.append((len(y), fixed_trees))
        return real(model, X, y, weights, X_val, y_val, max_trees, patience, device, fixed_trees=fixed_trees, linear_view=linear_view,
                    logistic_tolerance=logistic_tolerance, start_penalty=start_penalty)

    monkeypatch.setattr(training, "_fit", spying)
    result = run_experiment(STRATIFIED, PLAIN, model(), data, max_trees=30, patience=10, device="cpu")
    assert len(fits) == 4
    assert fits[-1] == (len(data.y), max(int(np.mean(result.best_iters) * training.REFIT_TREE_FACTOR), 1))
    assert roc_auc_score(hidden, result.test) > 0.7


def test_random_folds_rank_test_rows_by_the_fold_models_and_the_refit_together(iid, monkeypatch):
    data, _ = iid
    n_test = len(data.X_test)

    def stub_fit(model, X, y, weights, X_val, y_val, max_trees, patience, device, fixed_trees=None, linear_view=None, logistic_tolerance=1e-4, start_penalty=1.0):
        return ("refit" if fixed_trees else "fold"), fixed_trees or 5

    def stub_predict(fitted, X):
        order = np.arange(len(X), dtype=np.float32)
        return order[::-1] if fitted == "refit" and len(X) == n_test else order

    monkeypatch.setattr(training, "_fit", stub_fit)
    monkeypatch.setattr(training, "_predict", stub_predict)
    result = run_experiment(STRATIFIED, PLAIN, model(), data, max_trees=30, patience=10, device="cpu")
    # The two sides say opposite orders, so only their average gives every row the same score.
    assert np.ptp(result.test) < 1e-5


def test_time_folds_score_only_rows_with_past_training(temporal_raw, tmp_path):
    raw, hidden = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    result = run_experiment(TIME, PLAIN, model(), data, max_trees=40, patience=20, device="cpu")
    assert not result.mask.all()
    assert result.mask[-1] and not result.mask[0]
    assert result.cv == pytest.approx(roc_auc_score(data.y[result.mask], result.oof[result.mask]))
    assert result.test.shape == (800,)
    # Test predictions must come from the refit on all rows, not from an empty accumulator.
    assert roc_auc_score(hidden, result.test) > 0.7


def test_recency_weights_run_on_time_folds(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    result = run_experiment(TIME, PLAIN, model(recency_half_life=1.0), data, max_trees=30, patience=10, device="cpu")
    assert result.cv > 0.6


def test_in_fold_target_encoding_learns(iid):
    data, _ = iid
    plan = FeaturePlan(target_encode_max_cardinality=10, rationale="Target encode the category.")
    result = run_experiment(STRATIFIED, plan, model(), data, max_trees=40, patience=20, device="cpu")
    assert result.cv > 0.75


# --- leak canary ----------------------------------------------------------------------------
def test_canary_passes_on_a_clean_pipeline(iid):
    data, _ = iid
    plan = FeaturePlan(target_encode_max_cardinality=10, rationale="Target encode the category.")
    canary = leak_canary(STRATIFIED, plan, data, seed=0)
    assert canary["passed"] is True
    assert abs(canary["auc"] - 0.5) < 0.08


def test_canary_catches_validation_rows_inside_training(iid, monkeypatch):
    data, _ = iid

    def overlapping_folds(plan, y, ids, time_score=None):
        rows = np.arange(len(y))
        half = len(y) // 2
        return [(rows, rows[:half]), (rows, rows[half:])]

    monkeypatch.setattr("slayer.training.make_folds", overlapping_folds)
    canary = leak_canary(STRATIFIED, PLAIN, data, seed=0)
    assert canary["passed"] is False
    assert canary["auc"] > 0.8


def test_experiment_result_round_trips_through_disk(iid, tmp_path):
    data, _ = iid
    result = run_experiment(STRATIFIED, PLAIN, model(), data, max_trees=20, patience=10, device="cpu")
    path = tmp_path / "artifacts" / "exp.npz"
    save_predictions(path, result)
    oof, test, mask = load_predictions(path)
    assert np.array_equal(oof, result.oof)
    assert np.array_equal(test, result.test)
    assert np.array_equal(mask, result.mask)
