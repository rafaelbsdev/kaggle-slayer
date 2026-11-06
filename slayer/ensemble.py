"""Blends experiments on ranks, because AUC only cares about order."""
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

MAX_STEPS = 40
MIN_GAIN = 1e-5
START_PICKS = 3  # from one pick a second model can only join at half weight, and a useful weak one never will
FIT_SAMPLE = 1_000_000  # ponytail: greedy search on a sample past this; a smaller one lost measurable CV (decision 0009)
MAX_PLANS = 24          # ponytail: every step scores every plan, and past this many the weakest rarely earn a weight
FAMILY_BAG = 6          # the strongest models of a family, averaged, so a family can earn weight though no plan of it can
FAMILY = "family:"


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return rankdata(values) / len(values)


def _fast_auc(y: np.ndarray, scores: np.ndarray) -> float:
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    positives = y.sum()
    return (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * (len(y) - positives))


def _hill_climb_ranks(ranks: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Caruana's greedy selection with replacement."""
    names = list(ranks)
    scores = {name: _fast_auc(y, ranks[name]) for name in names}
    first = max(names, key=scores.get)
    picks = {first: START_PICKS}
    current = START_PICKS * ranks[first]
    best = scores[first]
    for step in range(START_PICKS + 1, MAX_STEPS + 1):
        trials = {name: _fast_auc(y, current + ranks[name]) for name in names}
        name = max(trials, key=trials.get)
        if trials[name] - best < MIN_GAIN:
            break
        best = trials[name]
        current += ranks[name]
        picks[name] = picks.get(name, 0) + 1
    total = sum(picks.values())
    return {name: count / total for name, count in picks.items()}


def hill_climb(oofs: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    ranks = {name: rank_normalize(oof[mask]) for name, oof in oofs.items()}
    return _hill_climb_ranks(ranks, y[mask])


def _combine(ranks: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weight * ranks[name] for name, weight in weights.items())


def _weigh_plans(bags: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    plans = [plan for plan in bags if not plan.startswith(FAMILY)]
    strongest = sorted(plans, key=lambda plan: -_fast_auc(y, bags[plan]))[:MAX_PLANS]
    searched = strongest + [plan for plan in bags if plan.startswith(FAMILY)]
    return _hill_climb_ranks({plan: bags[plan] for plan in searched}, y)


def blend(oofs: dict[str, np.ndarray], tests: dict[str, np.ndarray], y: np.ndarray, mask: np.ndarray,
          seed: int = 0, folds: int = 3, plans: dict[str, str] | None = None, families: dict[str, str] | None = None) -> dict:
    """Weights come from all scored rows, but `cv` is cross-fitted: each row is scored by weights
    fitted without it, so a blend cannot look better just by overfitting its own weights.

    `plans` names what each model runs apart from its seed. Seeds of a plan are averaged first, so the search
    weighs ideas instead of lucky seeds. A model with no plan stands alone.

    `families` adds one more bag per model family, from its strongest models across plans. These always reach the search:
    a family far below the top on CV can still be what a board wants (decision 0010).
    """
    if not oofs:
        raise ValueError("no models to blend")
    y_scored = y[mask]
    ranks = {name: rank_normalize(oof[mask]) for name, oof in oofs.items()}
    seeds_of = defaultdict(list)
    for name in ranks:
        seeds_of[(plans or {}).get(name, name)].append(name)
    by_family = defaultdict(list)
    for name, family in (families or {}).items():
        by_family[family].append(name)
    for family, names in by_family.items():
        strongest = sorted(names, key=lambda name: -_fast_auc(y_scored, ranks[name]))[:FAMILY_BAG]
        seeds_of[f"{FAMILY}{family}"] = strongest
    bags = {plan: np.mean([ranks[name] for name in names], axis=0) for plan, names in seeds_of.items()}
    rng = np.random.default_rng(seed)

    crossfit = np.empty(len(y_scored))
    for fit, hold in KFold(n_splits=folds, shuffle=True, random_state=seed).split(y_scored):
        sample = fit if len(fit) <= FIT_SAMPLE else rng.choice(fit, FIT_SAMPLE, replace=False)
        weights = _weigh_plans({plan: bag[sample] for plan, bag in bags.items()}, y_scored[sample])
        crossfit[hold] = _combine({plan: bag[hold] for plan, bag in bags.items()}, weights)

    everything = np.arange(len(y_scored))
    sample = everything if len(everything) <= FIT_SAMPLE else rng.choice(everything, FIT_SAMPLE, replace=False)
    weights = _weigh_plans({plan: bag[sample] for plan, bag in bags.items()}, y_scored[sample])
    members = defaultdict(float)
    for plan, weight in weights.items():
        for name in seeds_of[plan]:
            members[name] += weight / len(seeds_of[plan])  # a model can sit in its plan's bag and in its family's
    test_ranks = {name: rank_normalize(tests[name]) for name in members}
    return {
        "weights": dict(members),
        "cv": float(roc_auc_score(y_scored, crossfit)),
        "test": rank_normalize(_combine(test_ranks, members)),
    }
