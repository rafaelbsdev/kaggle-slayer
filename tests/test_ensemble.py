import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from slayer.ensemble import blend, hill_climb, rank_normalize


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    signal = y * 1.0
    a = signal + rng.normal(scale=1.2, size=4000)
    b = signal + rng.normal(scale=1.2, size=4000)
    noise = rng.normal(size=4000)
    return y, {"a": a, "b": b, "noise": noise}


def test_rank_normalize_stays_in_unit_interval_and_keeps_order():
    ranks = rank_normalize(np.array([10.0, -5.0, 3.0]))
    assert ranks.min() > 0 and ranks.max() <= 1
    assert ranks[0] > ranks[2] > ranks[1]


def test_hill_climb_combines_complementary_models_and_ignores_noise(predictions):
    y, oofs = predictions
    weights = hill_climb(oofs, y, np.ones(len(y), dtype=bool))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights.get("a", 0) > 0 and weights.get("b", 0) > 0
    assert weights.get("noise", 0) == 0


def test_hill_climb_stops_once_nothing_improves(monkeypatch):
    import slayer.ensemble as ensemble

    calls = []
    real = ensemble._fast_auc
    monkeypatch.setattr(ensemble, "_fast_auc", lambda y, scores: calls.append(1) or real(y, scores))
    y = np.array([1, 1, 0, 0])
    oofs = {"perfect": np.array([0.9, 0.8, 0.2, 0.1]), "noise": np.array([0.9, 0.1, 0.8, 0.2])}
    weights = hill_climb(oofs, y, np.ones(4, dtype=bool))
    assert weights == {"perfect": 1.0}
    # One scoring pass to pick the first model, one round of trials that brings no gain, then stop.
    assert len(calls) <= 3 * len(oofs)


def test_blend_reports_honest_crossfit_score(predictions):
    y, oofs = predictions
    tests = {k: v[:100] for k, v in oofs.items()}
    result = blend(oofs, tests, y, np.ones(len(y), dtype=bool), seed=0)
    best_single = max(roc_auc_score(y, v) for v in oofs.values())
    assert result["cv"] > best_single
    assert result["test"].shape == (100,)
    assert result["test"].min() >= 0 and result["test"].max() <= 1


def test_crossfit_score_does_not_reward_overfitted_weights():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1500)
    noise = {f"noise{i}": rng.normal(size=1500) for i in range(30)}
    mask = np.ones(1500, dtype=bool)
    result = blend(noise, {k: v[:10] for k, v in noise.items()}, y, mask, seed=0)
    in_sample = sum(weight * rank_normalize(noise[name]) for name, weight in hill_climb(noise, y, mask).items())
    assert roc_auc_score(y, in_sample) > 0.53
    assert result["cv"] < 0.52


def test_blend_respects_the_scored_mask(predictions):
    y, oofs = predictions
    mask = np.zeros(len(y), dtype=bool)
    mask[1000:] = True
    garbage = {k: np.where(mask, v, np.nan) for k, v in oofs.items()}
    result = blend(garbage, {k: v[:10] for k, v in oofs.items()}, y, mask, seed=0)
    assert np.isfinite(result["cv"])


def test_blend_of_one_model_is_that_model(predictions):
    y, oofs = predictions
    result = blend({"a": oofs["a"]}, {"a": oofs["a"][:10]}, y, np.ones(len(y), dtype=bool), seed=0)
    assert result["weights"] == {"a": 1.0}
    assert result["cv"] == pytest.approx(roc_auc_score(y, oofs["a"]), abs=1e-3)


def test_seeds_of_one_plan_blend_as_one_member(predictions):
    y, oofs = predictions
    rng = np.random.default_rng(1)
    seeds = {f"a{i}": oofs["a"] + rng.normal(scale=1.0, size=len(y)) for i in range(3)}
    members = {**seeds, "b": oofs["b"], "noise": oofs["noise"]}
    plans = {**{name: "plan_a" for name in seeds}, "b": "plan_b"}
    result = blend(members, {k: v[:10] for k, v in members.items()}, y, np.ones(len(y), dtype=bool), plans=plans)
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert result["weights"]["a0"] > 0 and result["weights"]["b"] > 0
    assert result["weights"]["a0"] == result["weights"]["a1"] == result["weights"]["a2"]
    assert "noise" not in result["weights"]
    assert result["cv"] > max(roc_auc_score(y, v) for v in members.values())


def test_the_search_sees_one_bag_per_plan(predictions, monkeypatch):
    import slayer.ensemble as ensemble

    y, oofs = predictions
    rng = np.random.default_rng(1)
    seeds = {f"a{i}": oofs["a"] + rng.normal(scale=1.0, size=len(y)) for i in range(3)}
    searched = []
    real = ensemble._hill_climb_ranks

    def spying(ranks, y_fit):
        searched.append(set(ranks))
        return real(ranks, y_fit)

    monkeypatch.setattr(ensemble, "_hill_climb_ranks", spying)
    blend(seeds, {k: v[:10] for k, v in seeds.items()}, y, np.ones(len(y), dtype=bool), plans=dict.fromkeys(seeds, "plan_a"))
    assert all(names == {"plan_a"} for names in searched)


def test_averaging_seeds_lifts_a_plan_above_its_best_seed(predictions):
    y, oofs = predictions
    rng = np.random.default_rng(1)
    seeds = {f"s{i}": oofs["a"] + rng.normal(scale=1.0, size=len(y)) for i in range(4)}
    plans = {name: "plan" for name in seeds}
    one_plan = blend(seeds, {k: v[:10] for k, v in seeds.items()}, y, np.ones(len(y), dtype=bool), plans=plans)
    assert set(one_plan["weights"]) == set(seeds)
    assert one_plan["cv"] > max(roc_auc_score(y, v) for v in seeds.values()) + 0.005


def test_only_the_strongest_plans_reach_the_search(predictions, monkeypatch):
    import slayer.ensemble as ensemble

    monkeypatch.setattr(ensemble, "MAX_PLANS", 1)
    y, oofs = predictions
    result = blend(oofs, {k: v[:10] for k, v in oofs.items()}, y, np.ones(len(y), dtype=bool), seed=0)
    assert len(result["weights"]) == 1
    assert "noise" not in result["weights"]


def strong_and_weak(n=20000):
    """`weak` knows something `strong` does not, but only about a third as clearly, so a small weight helps and half does not."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, n)
    return y, {"strong": y + rng.normal(scale=1.0, size=n), "weak": y + rng.normal(scale=3.0, size=n)}


def test_a_weak_model_can_join_at_a_small_weight():
    y, oofs = strong_and_weak()
    half = roc_auc_score(y, rank_normalize(oofs["strong"]) + rank_normalize(oofs["weak"]))
    assert half < roc_auc_score(y, oofs["strong"])  # at equal weight it would hurt
    weights = blend(oofs, {k: v[:10] for k, v in oofs.items()}, y, np.ones(len(y), dtype=bool), seed=0)["weights"]
    assert 0 < weights["weak"] < 0.3


def test_a_family_bag_reaches_the_search_however_weak(monkeypatch):
    import slayer.ensemble as ensemble

    monkeypatch.setattr(ensemble, "MAX_PLANS", 1)
    y, oofs = strong_and_weak()
    tests = {k: v[:10] for k, v in oofs.items()}
    mask = np.ones(len(y), dtype=bool)
    assert "weak" not in blend(oofs, tests, y, mask, seed=0)["weights"]
    with_families = blend(oofs, tests, y, mask, seed=0, families={"strong": "boost", "weak": "tree"})
    assert with_families["weights"]["weak"] > 0


def test_a_family_bag_takes_only_the_strongest_models_of_the_family(monkeypatch):
    import slayer.ensemble as ensemble

    monkeypatch.setattr(ensemble, "FAMILY_BAG", 1)
    monkeypatch.setattr(ensemble, "MAX_PLANS", 1)
    y, oofs = strong_and_weak()
    rng = np.random.default_rng(4)
    members = {**oofs, "weaker": y + rng.normal(scale=30.0, size=len(y))}
    families = {"strong": "boost", "weak": "tree", "weaker": "tree"}
    weights = blend(members, {k: v[:10] for k, v in members.items()}, y, np.ones(len(y), dtype=bool), seed=0, families=families)["weights"]
    assert "weaker" not in weights and weights["weak"] > 0


def test_a_model_in_a_plan_bag_and_a_family_bag_adds_up_its_weights():
    y, oofs = strong_and_weak()
    plans = {"strong": "plan_a", "weak": "plan_b"}
    families = {"strong": "boost", "weak": "tree"}
    result = blend(oofs, {k: v[:10] for k, v in oofs.items()}, y, np.ones(len(y), dtype=bool), seed=0, plans=plans, families=families)
    assert sum(result["weights"].values()) == pytest.approx(1.0)


def test_blend_refuses_no_models(predictions):
    y, _ = predictions
    with pytest.raises(ValueError, match="models"):
        blend({}, {}, y, np.ones(len(y), dtype=bool), seed=0)
