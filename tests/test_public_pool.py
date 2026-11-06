import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

spec = importlib.util.spec_from_file_location("public_pool", Path(__file__).resolve().parents[1] / "tools" / "public_pool.py")
public_pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public_pool)


def members(n=6000):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    pool = {"a": y + rng.normal(scale=1.5, size=n), "b": y + rng.normal(scale=1.5, size=n), "noise": rng.normal(size=n)}
    return y, {name: public_pool.rank_normalize(values) for name, values in pool.items()}


def perfect_of(y):
    return y.astype(np.float64)


def test_folds_that_only_rename_the_same_groups_are_accepted():
    ours = np.array([0, 0, 1, 1, 2, 2])
    public_pool.require_same_folds(np.array([2, 2, 0, 0, 1, 1]), ours, "renamed")


def test_a_different_split_is_refused():
    ours = np.array([0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="other_split"):
        public_pool.require_same_folds(np.array([0, 1, 1, 0, 2, 2]), ours, "other_split")


def test_a_nested_stack_of_two_views_beats_either_one():
    y, pool = members()
    folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(y))
    scores = public_pool.nested_stack(public_pool.rank_logits(pool, list(pool)), y, folds)
    assert roc_auc_score(y, scores) > max(roc_auc_score(y, pool["a"]), roc_auc_score(y, pool["b"])) + 0.01


def test_a_nested_stack_scores_a_row_without_seeing_its_label():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    pool = {f"m{i}": public_pool.rank_normalize(rng.normal(size=600)) for i in range(60)}
    folds = list(KFold(n_splits=3, shuffle=True, random_state=0).split(y))
    X = public_pool.rank_logits(pool, list(pool))
    assert roc_auc_score(y, public_pool.fit_stack(X, y).decision_function(X)) > 0.65  # what seeing its own label looks like
    assert roc_auc_score(y, public_pool.nested_stack(X, y, folds)) < 0.56


def test_public_predictions_too_weak_are_left_out_and_folds_are_checked(tmp_path):
    y, pool = members(300)
    ours = np.arange(300) % 3
    gpt2 = tmp_path / public_pool.GPT2
    bank = tmp_path / public_pool.GOODPJW
    alex = tmp_path / public_pool.ALEX
    series = tmp_path / public_pool.SERIES
    queyrusi = tmp_path / public_pool.QUEYRUSI
    pooled = tmp_path / public_pool.POOLED
    for folder in (gpt2, bank, alex, series, queyrusi, pooled):
        folder.mkdir()
    np.savez(pooled / "pool_bank.npz", oof=np.column_stack([perfect_of(y), 1 - perfect_of(y)]), test=np.column_stack([perfect_of(y)[:10], 1 - perfect_of(y)[:10]]))
    (pooled / "pool_manifest.json").write_text(json.dumps({"names": ["first", "second"], "n_folds": 5, "fold_seed": 42}), encoding="utf-8")
    np.save(gpt2 / "fold_ids.npy", ours)
    np.save(gpt2 / "oof.npy", pool["a"])
    np.save(gpt2 / "test.npy", pool["a"][:10])
    (bank / "folds.csv").write_text("id,fold\n" + "".join(f"{i},{ours[i]}\n" for i in range(300)), encoding="utf-8")
    perfect = y.astype(np.float64)
    np.save(bank / "oof_strong.npy", perfect)
    np.save(bank / "test_strong.npy", perfect[:10])
    np.save(alex / "oof_weak.npy", pool["noise"])
    np.save(alex / "test_weak.npy", pool["noise"][:10])
    for bundle in ("predictions.npz", "gen_predictions.npz"):
        np.savez(queyrusi / bundle, folds=ours, oof=perfect, test=perfect[:10])
    np.save(series / "folded_oof.npy", perfect)
    np.save(series / "folded_test.npy", perfect[:10])
    np.save(series / "refit_oof.npy", perfect)
    np.save(series / "refit_full.npy", perfect[:10])
    np.save(series / "both_oof.npy", perfect)
    np.save(series / "both_test.npy", 1 - perfect[:10])
    np.save(series / "both_full.npy", perfect[:10])
    np.save(series / "no_test_side_oof.npy", perfect)
    np.save(series / "weak_oof.npy", pool["noise"])
    np.save(series / "weak_test.npy", pool["noise"][:10])
    np.save(series / "stage_glmoof.npy", pool["noise"])
    oof, test = public_pool.load_public(tmp_path, y, ours)
    assert set(oof) == set(test) == {"gpt2", "pj_strong", "qr_cnn", "qr_gen", "gs_folded", "gs_refit", "gs_both", "pl_first", "pl_second"}
    assert np.array_equal(test["gs_both"], public_pool.rank_normalize(1 - perfect[:10]))  # the average of the fold models comes before a single refit
    assert np.array_equal(oof["pl_second"], public_pool.rank_normalize(1 - perfect))
    assert np.array_equal(test["pl_first"], public_pool.rank_normalize(perfect[:10]))
    (pooled / "pool_manifest.json").write_text(json.dumps({"names": ["first", "second"], "n_folds": 5, "fold_seed": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match="pooled"):
        public_pool.load_public(tmp_path, y, ours)
    (pooled / "pool_manifest.json").write_text(json.dumps({"names": ["first", "second"], "n_folds": 5, "fold_seed": 42}), encoding="utf-8")
    for bundle in ("predictions.npz", "gen_predictions.npz"):
        np.savez(queyrusi / bundle, folds=(np.arange(300) * 7 + np.arange(300) // 2) % 3, oof=perfect, test=perfect[:10])
    with pytest.raises(ValueError, match="queyrusi"):
        public_pool.load_public(tmp_path, y, ours)
    (bank / "folds.csv").write_text("id,fold\n" + "".join(f"{i},{(i * 7) % 3}\n" for i in range(150)) + "".join(f"{i},0\n" for i in range(150, 300)), encoding="utf-8")
    with pytest.raises(ValueError, match="goodpjw"):
        public_pool.load_public(tmp_path, y, ours)


def test_the_stack_reaches_the_optimum_on_near_identical_members_of_a_rare_class():
    rng = np.random.default_rng(2)
    y = (rng.random(20000) < 0.017).astype(int)
    shared = 2 * y + rng.normal(size=20000)
    pool = {f"m{i}": public_pool.rank_normalize(shared + rng.normal(scale=0.05, size=20000)) for i in range(40)}
    X = public_pool.rank_logits(pool, list(pool))
    fit_exactly = make_pipeline(StandardScaler(), LogisticRegression(C=public_pool.STACK_C, max_iter=20000, tol=1e-11)).fit(X, y)
    fit_loosely = make_pipeline(StandardScaler(), LogisticRegression(C=public_pool.STACK_C, max_iter=20000)).fit(X, y)
    assert np.abs(fit_loosely[-1].coef_ - fit_exactly[-1].coef_).max() > 0.01  # the default tolerance really does stop short here
    assert np.abs(public_pool.fit_stack(X, y)[-1].coef_ - fit_exactly[-1].coef_).max() < 0.001


def write_run(artifacts, ledger_path, experiments, n=30):
    rng = np.random.default_rng(3)
    with open(ledger_path, "w", encoding="utf-8") as handle:
        for experiment in experiments:
            np.savez(artifacts / f"{experiment['id']}.npz", oof=rng.normal(size=n), test=rng.normal(size=n))
            handle.write(json.dumps({"kind": "experiment", "canary_passed": True, "params": {}, "recency_half_life": None, **experiment}) + "\n")


@pytest.mark.parametrize("smoothing, expected", [(299.0, False), (300.0, True), (1000.0, True)])
def test_heavily_smoothed_plans_become_a_member_of_their_own(tmp_path, smoothing, expected):
    write_run(tmp_path, tmp_path / "ledger.jsonl", [
        {"id": "plain", "family": "logistic_boost", "cv": 0.95, "features": {}},
        {"id": "smoothed", "family": "logistic_boost", "cv": 0.94, "features": {"target_encode_smoothing": smoothing}},
    ] + [{"id": family, "family": family, "cv": 0.93, "features": {}} for family in public_pool.FAMILIES])
    oof, test = public_pool.load_own(tmp_path, tmp_path / "ledger.jsonl")
    assert ("own_smoothed" in oof) is expected
    assert ("own_smoothed" in test) is expected
