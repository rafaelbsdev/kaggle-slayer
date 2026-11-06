"""Turns the competition's CSVs into matrices and a profile with no row values in it."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.metrics import roc_auc_score

TEMPORAL_WORDS = re.compile(r"temporal|chronolog|time[- ]based|strictly after|out[- ]of[- ]time", re.I)
ADVERSARIAL_SAMPLE = 100_000
TOP_K = 30
DROP_CURVE = (1, 2, 5, 10, 20, 50, 100)
MAX_DROP_SHARE = 0.8   # mirrors the verifier: a plan must keep at least 20% of the features
NO_DRIFT_AUC = 0.55    # below this, train and test look alike and the curve stops
CLOCK_MIN_BEYOND = 0.5 # share of test past train's 1st-99th percentile range that makes a column a clock
CLOCK_MAX_NULLS = 0.05 # a clock with many missing readings cannot order rows


@dataclass
class Schema:
    id_col: str
    target_col: str
    pred_col: str
    features: list[str]
    categoricals: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)


@dataclass
class CompetitionData:
    schema: Schema
    X_train: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    sample_path: Path
    n_unique: dict[str, int]
    target_is_binary: bool
    drift_rank: list[str] = field(default_factory=list)
    time_score: np.ndarray | None = None
    joint_pairs: list[tuple[str, str]] | None = None


def infer_schema(train_cols: list[str], test_cols: list[str], sample_cols: list[str]) -> Schema:
    """Reads the competition's shape from its files, so no competition needs its own code."""
    if len(sample_cols) != 2:
        raise ValueError(f"sample submission must have 2 columns, found {sample_cols}")
    id_col, pred_col = sample_cols
    targets = [c for c in train_cols if c not in test_cols]
    if len(targets) != 1:
        raise ValueError(f"expected one target column in train only, found {targets}")
    if id_col not in train_cols or id_col not in test_cols:
        raise ValueError(f"id column {id_col!r} must be in both train and test")
    return Schema(id_col, targets[0], pred_col, [c for c in test_cols if c != id_col])


def temporal_hint(overview: str) -> bool:
    return bool(TEMPORAL_WORDS.search(overview or ""))


def _to_parquet(csv_path: Path, parquet_path: Path) -> None:
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    lazy = pl.scan_csv(csv_path, infer_schema_length=None)
    partial = parquet_path.with_name(parquet_path.name + ".partial")
    lazy.with_columns(cs.float().cast(pl.Float32)).sink_parquet(partial)
    os.replace(partial, parquet_path)  # so an interrupted write can never pass for a finished file


def _encode_strings(train: pl.DataFrame, test: pl.DataFrame, columns: list[str]):
    """Same code for the same value in train and test. Nulls stay null, so they become NaN."""
    for column in columns:
        values = pl.concat([train[column].cast(pl.String), test[column].cast(pl.String)]).drop_nulls().unique().sort()
        codes = pl.Series(range(len(values)), dtype=pl.Float32)
        encode = pl.col(column).cast(pl.String).replace_strict(values, codes, default=None, return_dtype=pl.Float32)
        train, test = train.with_columns(encode), test.with_columns(encode)
    return train, test


def _to_binary(target: pl.Series) -> tuple[np.ndarray, bool]:
    values = target.cast(pl.String).to_numpy() if target.dtype == pl.String else target.to_numpy()
    unique = np.unique(values)
    if len(unique) == 2:
        return (values == unique[1]).astype(np.int8), True
    return values, False


def load_competition(raw_dir, work_dir) -> CompetitionData:
    raw_dir, work_dir = Path(raw_dir), Path(work_dir)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        if not (raw_dir / name).exists():
            raise FileNotFoundError(f"{name} not found in {raw_dir}")

    _to_parquet(raw_dir / "train.csv", work_dir / "train.parquet")
    _to_parquet(raw_dir / "test.csv", work_dir / "test.parquet")
    train = pl.read_parquet(work_dir / "train.parquet")
    test = pl.read_parquet(work_dir / "test.parquet")

    sample_cols = pl.read_csv(raw_dir / "sample_submission.csv", n_rows=1).columns
    schema = infer_schema(train.columns, test.columns, sample_cols)
    # A column with no values in one file teaches nothing usable, and CSV inference reads it as text there.
    schema.empty = [c for c in schema.features
                    if train[c].null_count() == train.height or test[c].null_count() == test.height]
    schema.features = [c for c in schema.features if c not in schema.empty]
    if not schema.features:
        raise ValueError("no feature has values in both train and test")
    schema.categoricals = [c for c in schema.features if train[c].dtype == pl.String or test[c].dtype == pl.String]
    train, test = _encode_strings(train, test, schema.categoricals)

    y, is_binary = _to_binary(train[schema.target_col])
    n_unique = train.select(schema.features).select(pl.all().n_unique()).row(0, named=True)
    return CompetitionData(
        schema=schema,
        X_train=train.select(schema.features).cast(pl.Float32).to_numpy(order="c"),
        y=y,
        X_test=test.select(schema.features).cast(pl.Float32).to_numpy(order="c"),
        train_ids=train[schema.id_col].to_numpy(),
        test_ids=test[schema.id_col].to_numpy(),
        sample_path=raw_dir / "sample_submission.csv",
        n_unique=n_unique,
        target_is_binary=is_binary,
    )


def _adversarial(X_a: np.ndarray, X_b: np.ndarray, seed: int, columns: list[int] | None = None) -> tuple[float, np.ndarray]:
    """How well a model tells two sets apart: about 0.5 means they look the same."""
    rng = np.random.default_rng(seed)
    a = rng.choice(len(X_a), min(len(X_a), ADVERSARIAL_SAMPLE), replace=False)
    b = rng.choice(len(X_b), min(len(X_b), ADVERSARIAL_SAMPLE), replace=False)
    X = np.vstack([X_a[a], X_b[b]])
    if columns is not None:  # after the row sample, so the column copy stays small on wide data
        X = X[:, columns]
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    order = rng.permutation(len(y))
    half = len(y) // 2
    fit, score = order[:half], order[half:]
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15, colsample_bytree=0.5,
                               subsample=0.8, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1,
                               importance_type="gain")
    model.fit(X[fit], y[fit])
    auc = float(roc_auc_score(y[score], model.predict_proba(X[score])[:, 1]))
    return auc, model.feature_importances_.astype(float)


def find_drift_clock(X_train: np.ndarray, X_test: np.ndarray, drift_order: np.ndarray) -> tuple[int, int, float] | None:
    """A column that grows with time leaves test beyond everything train has seen.

    Returns (column, direction, share of test beyond train's range) for the most drifting such column.
    When rows and ids say nothing about time, sorting by this column stands in for it (decision 0005).
    """
    for column in drift_order[:TOP_K]:
        values = X_train[:, column]
        future = X_test[:, column]
        future = future[~np.isnan(future)]
        if np.isnan(values).mean() > CLOCK_MAX_NULLS or not future.size:
            continue
        low, high = np.nanquantile(values, [0.01, 0.99])
        above, below = float((future > high).mean()), float((future < low).mean())
        if max(above, below) >= CLOCK_MIN_BEYOND:
            return int(column), 1 if above >= below else -1, max(above, below)
    return None


def build_profile(data: CompetitionData, seed: int = 0) -> dict:
    """Everything the agents may know about the data: derived statistics only (decision 0002)."""
    features = data.schema.features
    n_train = len(data.y)

    auc, importance = _adversarial(data.X_train, data.X_test, seed)
    drift_order = np.argsort(-importance, kind="stable")
    data.drift_rank = [features[i] for i in drift_order]
    total = importance.sum() or 1.0

    # Does the order of the rows carry time? The first and last rows differ only if it does.
    fifth = max(n_train // 5, 1)
    row_auc, _ = _adversarial(data.X_train[:fifth], data.X_train[-fifth:], seed)
    id_auc = row_auc
    if not (np.diff(data.train_ids) >= 0).all():
        by_id = np.argsort(data.train_ids, kind="stable")
        id_auc, _ = _adversarial(data.X_train[by_id[:fifth]], data.X_train[by_id[-fifth:]], seed)

    clock = find_drift_clock(data.X_train, data.X_test, drift_order)
    data.time_score = None
    if clock is not None:
        values = data.X_train[:, clock[0]]
        data.time_score = np.where(np.isnan(values), -np.inf, values * clock[1])  # no reading counts as earliest

    # How much drift is left after dropping the k most drifting features: the Feature Engineer
    # chooses k from this curve instead of guessing.
    drift_after_drop = []
    for k in DROP_CURVE:
        if k > MAX_DROP_SHARE * len(features):
            break
        dropped = set(drift_order[:k].tolist())
        keep = [i for i in range(len(features)) if i not in dropped]
        auc_k, _ = _adversarial(data.X_train, data.X_test, seed, columns=keep)
        drift_after_drop.append({"k": k, "adversarial_auc": round(auc_k, 4)})
        if auc_k < NO_DRIFT_AUC:
            break

    null_rates = np.isnan(data.X_train).mean(axis=0)
    null_order = np.argsort(-null_rates, kind="stable")[:TOP_K]
    cardinality = np.array([data.n_unique[f] for f in features])

    profile = {
        "n_train": n_train,
        "n_test": len(data.test_ids),
        "n_features": len(features),
        "n_categorical": len(data.schema.categoricals),
        "n_features_dropped_as_empty": len(data.schema.empty),
        "target_is_binary": bool(data.target_is_binary),
        "adversarial_auc": round(auc, 4),
        "row_order_drift_auc": round(row_auc, 4),
        "id_order_drift_auc": round(id_auc, 4),
        "drift_clock": None if clock is None else {"name": features[clock[0]],
                                                   "share_of_test_beyond_train": round(clock[2], 4)},
        "top_drifting": [{"name": features[i], "share": round(float(importance[i] / total), 4)}
                         for i in drift_order[:TOP_K] if importance[i] > 0],
        "drift_after_drop": drift_after_drop,
        "features_with_nulls": int((null_rates > 0).sum()),
        "top_null": [{"name": features[i], "null_rate": round(float(null_rates[i]), 4)}
                     for i in null_order if null_rates[i] > 0],
        "cardinality_buckets": {
            "constant": int((cardinality <= 1).sum()),
            "binary": int((cardinality == 2).sum()),
            "up_to_20": int(((cardinality > 2) & (cardinality <= 20)).sum()),
            "up_to_1000": int(((cardinality > 20) & (cardinality <= 1000)).sum()),
            "above_1000": int((cardinality > 1000).sum()),
        },
    }
    if data.target_is_binary:
        blocks = np.array_split(data.y, 5)
        profile["target_rate"] = round(float(data.y.mean()), 4)
        profile["target_rate_by_row_order"] = [round(float(block.mean()), 4) for block in blocks]
    return profile
