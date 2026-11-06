"""Second phase, run by hand: what do other people's published predictions add to this team's models?

The agents take no part in this and never see these files. Scores go to a ledger of their own, so the team's numbers stay
the team's (decision 0011). The predictions must be downloaded first, each dataset into data/public-banks/<owner>__<name>.

    python tools/public_pool.py measure
    python tools/public_pool.py write --name pool-stack --method stack
    python tools/public_pool.py submit --name pool-stack --cv 0.95212
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slayer.contracts import ValidationPlan  # noqa: E402
from slayer.data import load_competition  # noqa: E402
from slayer.ensemble import _hill_climb_ranks  # noqa: E402
from slayer.kaggle_client import KaggleClient  # noqa: E402
from slayer.ledger import Ledger  # noqa: E402
from slayer.orchestrator import plan_key  # noqa: E402
from slayer.submission import validate_submission, write_submission  # noqa: E402
from slayer.training import make_folds  # noqa: E402

SLUG = "playground-series-s6e11"
BANKS = Path("data/public-banks")
GPT2 = "masayakawamata__s6e11-gpt2-target-first-oof"
GOODPJW = "goodpjw2008__s6e11-oof-bank"
ALEX = "alexchilton__s6e11-oof-predictions"
SERIES = "goodpjw2008__s6e11-fraud-series-artifacts"
QUEYRUSI = "queyrusi__fraud-shared-digit-cnn-weights"
POOLED = "malyshevdanil__s6e11-pooled-stack-v9"
OWN_PLANS = 12
FAMILY_BAG = 6
FOLDS_OF_THIS_TEAM, SEED_OF_THIS_TEAM = 5, 42
FAMILIES = ("lightgbm", "xgboost", "catboost")
SMOOTHED_FROM = 300      # heavily smoothed plans score too low to be among the strongest, yet add most as their own member
MIN_PUBLIC_AUC = 0.94    # one public member sits near 0.90, and a stack gains nothing from it
STACK_C = 0.05           # the value public stacks on this competition settled on; 0.01 and 0.2 both measured lower here
STACK_TOL = 1e-7         # with a rare class and near-identical columns the default 1e-4 stops far from the optimum, and adding members could lower CV


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return (rankdata(values) / len(values)).astype(np.float32)


def rank_logits(pool: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.column_stack([np.log(np.clip(pool[n], 1e-6, 1 - 1e-6) / (1 - np.clip(pool[n], 1e-6, 1 - 1e-6))) for n in names]).astype(np.float32)


def fit_stack(X: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), LogisticRegression(C=STACK_C, max_iter=400, tol=STACK_TOL)).fit(X, y)


def nested_stack(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Each row is scored by a stack fitted without it."""
    out = np.zeros(len(y))
    for fit, hold in folds:
        out[hold] = fit_stack(X[fit], y[fit]).decision_function(X[hold])
    return out


def nested_hill(pool: dict[str, np.ndarray], y: np.ndarray, folds) -> np.ndarray:
    out = np.zeros(len(y))
    for fit, hold in folds:
        weights = _hill_climb_ranks({n: rankdata(v[fit]) / len(fit) for n, v in pool.items()}, y[fit])
        out[hold] = sum(w * pool[n][hold] for n, w in weights.items())
    return out


def require_same_folds(theirs: np.ndarray, ours: np.ndarray, source: str) -> None:
    """Predictions from another split would let each stack fold see rows it is scored on, and CV would lie."""
    if len(set(zip(theirs.tolist(), ours.tolist()))) != len(set(ours.tolist())):
        raise ValueError(f"{source} does not use the folds of this team, so its predictions cannot be stacked")


def load_own(artifacts: Path, ledger_path: Path) -> tuple[dict, dict]:
    rows = Ledger(ledger_path).read("experiment")
    experiments = [e for e in rows if e["canary_passed"]]
    plans = defaultdict(list)
    for e in experiments:
        plans[plan_key(e)].append(e)
    groups = {f"own{i}": members for i, members in
              enumerate(sorted(plans.values(), key=lambda members: -max(e["cv"] for e in members))[:OWN_PLANS])}
    for family in FAMILIES:
        groups[f"own_{family}"] = sorted((e for e in experiments if e["family"] == family), key=lambda e: -e["cv"])[:FAMILY_BAG]
    smoothed = [e for e in experiments if e["features"].get("target_encode_smoothing", 20.0) >= SMOOTHED_FROM]
    if smoothed:
        groups["own_smoothed"] = sorted(smoothed, key=lambda e: -e["cv"])[:FAMILY_BAG]
    oof, test = {}, {}
    for name, members in groups.items():
        loaded = [np.load(artifacts / f"{e['id']}.npz", allow_pickle=False) for e in members]
        oof[name] = np.mean([rank_normalize(z["oof"]) for z in loaded], axis=0).astype(np.float32)
        test[name] = np.mean([rank_normalize(z["test"]) for z in loaded], axis=0).astype(np.float32)
    return oof, test


def load_public(banks: Path, y: np.ndarray, ours: np.ndarray) -> tuple[dict, dict]:
    require_same_folds(np.load(banks / GPT2 / "fold_ids.npy", allow_pickle=False), ours, GPT2)
    with open(banks / GOODPJW / "folds.csv", encoding="utf-8", newline="") as handle:
        require_same_folds(np.array([int(row["fold"]) for row in csv.DictReader(handle)]), ours, GOODPJW)
    oof = {"gpt2": rank_normalize(np.load(banks / GPT2 / "oof.npy", allow_pickle=False))}
    test = {"gpt2": rank_normalize(np.load(banks / GPT2 / "test.npy", allow_pickle=False))}
    for folder, tag in ((banks / GOODPJW, "pj"), (banks / ALEX, "ac")):
        for path in sorted(folder.rglob("oof*.npy")):
            values = np.load(path, allow_pickle=False)
            if roc_auc_score(y, values) >= MIN_PUBLIC_AUC:
                name = f"{tag}_{path.name[4:-4]}"
                oof[name] = rank_normalize(values)
                test[name] = rank_normalize(np.load(path.with_name(path.name.replace("oof", "test", 1)), allow_pickle=False))
    for bundle, tag in (("predictions.npz", "qr_cnn"), ("gen_predictions.npz", "qr_gen")):
        with np.load(banks / QUEYRUSI / bundle, allow_pickle=False) as saved:
            require_same_folds(saved["folds"], ours, QUEYRUSI)
            oof[tag] = rank_normalize(saved["oof"])
            test[tag] = rank_normalize(saved["test"])
    # This dataset names our split but ships no fold ids. Its fold means spread 5 to 139 times wider under our folds than under any of 20 random ones.
    for path in sorted((banks / SERIES).glob("*_oof.npy")):
        stem = path.name[:-len("_oof.npy")]
        partner = next((p for p in (path.with_name(f"{stem}_test.npy"), path.with_name(f"{stem}_full.npy")) if p.exists()), None)
        values = np.load(path, allow_pickle=False)
        if partner is not None and roc_auc_score(y, values) >= MIN_PUBLIC_AUC:
            oof[f"gs_{stem}"] = rank_normalize(values)
            test[f"gs_{stem}"] = rank_normalize(np.load(partner, allow_pickle=False))
    manifest = json.loads((banks / POOLED / "pool_manifest.json").read_text(encoding="utf-8"))
    if (manifest["n_folds"], manifest["fold_seed"]) != (FOLDS_OF_THIS_TEAM, SEED_OF_THIS_TEAM):
        raise ValueError(f"{POOLED} does not use the folds of this team, so its predictions cannot be stacked")
    with np.load(banks / POOLED / "pool_bank.npz", allow_pickle=False) as saved:
        for i, name in enumerate(manifest["names"]):
            oof[f"pl_{name}"] = rank_normalize(saved["oof"][:, i])
            test[f"pl_{name}"] = rank_normalize(saved["test"][:, i])
    return oof, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure", "write", "submit"))
    parser.add_argument("--name")
    parser.add_argument("--method", choices=("stack", "hill", "both"))
    parser.add_argument("--cv", type=float)
    args = parser.parse_args()
    ledger = Ledger(f"runs/{SLUG}-public-pool/ledger.jsonl")

    if args.action == "submit":
        client = KaggleClient()
        result = client.submit(SLUG, f"submissions/{SLUG}/{args.name}.csv", f"{args.name} cv={args.cv:.5f} with public predictions")
        score = result["public_score"]
        ledger.append("submission", ref=result["ref"], candidate=args.name, cv=args.cv, public_score=score,
                      gap=None if score is None else args.cv - score, error=result.get("error"))
        print(json.dumps({"candidate": args.name, "cv": args.cv, "public": score}))
        return

    data = load_competition(f"data/{SLUG}/raw", f"data/{SLUG}/work")
    y = data.y
    folds = make_folds(ValidationPlan(scheme="stratified_kfold", n_splits=5, rationale="The folds of every model here."), y, data.train_ids)
    ours = np.zeros(len(y), dtype=np.int8)
    for k, (_, hold) in enumerate(folds):
        ours[hold] = k
    own_oof, own_test = load_own(Path(f"artifacts/{SLUG}"), Path(f"runs/{SLUG}/ledger.jsonl"))
    pub_oof, pub_test = load_public(BANKS, y, ours)
    oof, test = {**own_oof, **pub_oof}, {**own_test, **pub_test}
    names = list(oof)

    def report(label, scores):
        entry = {"mix": label, "cv": round(float(roc_auc_score(y, scores)), 6), "folds": [round(float(roc_auc_score(y[h], scores[h])), 5) for _, h in folds]}
        print(json.dumps(entry), flush=True)
        return entry

    if args.action == "measure":
        report("own, hill", nested_hill(own_oof, y, folds))
        report("own and gpt2, hill", nested_hill({**own_oof, "gpt2": pub_oof["gpt2"]}, y, folds))
        report("own, stack", nested_stack(rank_logits(own_oof, list(own_oof)), y, folds))
        report("public, stack", nested_stack(rank_logits(pub_oof, list(pub_oof)), y, folds))
        entry = report("own and public, stack", nested_stack(rank_logits(oof, names), y, folds))
        ledger.append("measure", members=len(names), own=len(own_oof), public=len(pub_oof), cv=entry["cv"], folds=entry["folds"])
        return

    hill_pool = {**own_oof, "gpt2": pub_oof["gpt2"]}
    weights = _hill_climb_ranks({n: rankdata(v) / len(v) for n, v in hill_pool.items()}, y)
    hill_test = rank_normalize(sum(w * test[n] for n, w in weights.items()))
    stack_test = rank_normalize(fit_stack(rank_logits(oof, names), y).decision_function(rank_logits(test, names)))
    predicted = {"hill": hill_test, "stack": stack_test, "both": rank_normalize(hill_test + stack_test)}[args.method]
    path = write_submission(data.sample_path, data.test_ids, predicted, f"submissions/{SLUG}/{args.name}.csv")
    problems = validate_submission(path, data.sample_path)
    ledger.append("written", name=args.name, method=args.method, hill_weights=weights, members=len(names), problems=problems)
    print(json.dumps({"wrote": str(path), "problems": problems, "hill_weights": {k: round(v, 3) for k, v in weights.items()}}))


if __name__ == "__main__":
    main()
