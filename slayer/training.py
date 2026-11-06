"""Runs what the agents chose and measures it. Scores here are observed, never declared by an agent."""
import gc
import itertools
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

from slayer.contracts import FeaturePlan, ModelParams, ModelPlan, ValidationPlan

MAX_ENCODED_COLUMNS = 200  # ponytail: caps memory on wide data; lift it if encoding ever proves its value
MAX_KEYED_COLUMNS = 30     # ponytail: six keys per column; on wide anonymised data they are unlikely to pay
MIN_KEYED_CARDINALITY = 50
JOINT_MIN_SIGMAS = 4.0     # noise gives a gap near its degrees of freedom, with a spread of sqrt(2 * df)
JOINT_CUTS = np.linspace(0.1, 0.9, 9)
SHARED_POOL_PRIOR = 50.0     # how hard a token's rate over all fields leans on the base rate
SHARED_FIELD_PRIOR = 2000.0  # how hard a field's rate leans on the token's rate over all fields
SHARED_CHAIN_PRIOR = 200.0   # how hard a pair of neighbouring tokens leans on its field's rate
SHARED_MANY_VALUES = 1000    # past this many distinct values a field is written unlike the scores below it
NUMBER_KEY_KINDS = ("whole", "no_units", "one_decimal", "decimals", "size_first", "size_first_two")
TOKEN_KEY_KINDS = ("first", "second", "last", "count", "first_with_count", "before_point")
TOKENIZER_FILE = Path(__file__).resolve().parent.parent / "data" / "gpt2-tokenizer" / "tokenizer.json"
LOGISTIC_MAX_ITER = 1000
ENCODING_CLIP = 1e-5
MAX_ONE_HOT_LEVELS = 50    # ponytail: more categories than this reach a linear model through their encoding only
START_REG_LAMBDA = 1.0     # what a plan starts from, and `logistic_start_penalty` replaces it
CANARY_SAMPLE = 20_000
CANARY_TREES = 80
REFIT_TREE_FACTOR = 1.1    # the refit sees more rows than any fold, so it can use slightly more trees


@dataclass
class ExperimentResult:
    oof: np.ndarray
    test: np.ndarray
    mask: np.ndarray
    cv: float
    std: float
    fold_scores: list[float]
    best_iters: list[int]
    seconds: float


# --- folds and weights ----------------------------------------------------------------------
def _time_order(ids: np.ndarray, order_by: str, time_score: np.ndarray | None = None) -> np.ndarray:
    if order_by == "row_order":
        return np.arange(len(ids))
    if order_by == "id":
        return np.argsort(ids, kind="stable")
    if time_score is None:
        raise ValueError("ordering by the drift clock needs a clock found by the profile")
    return np.argsort(time_score, kind="stable")


def make_folds(plan: ValidationPlan, y: np.ndarray, ids: np.ndarray,
               time_score: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(y) < (plan.n_splits + 1) * 2:
        raise ValueError(f"{len(y)} rows are too few for {plan.n_splits} folds")
    if plan.scheme == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=plan.n_splits, shuffle=True, random_state=42)
        return list(splitter.split(np.zeros(len(y)), y))
    blocks = np.array_split(_time_order(ids, plan.order_by, time_score), plan.n_splits + 1)
    return [(np.concatenate(blocks[:k]), blocks[k]) for k in range(1, plan.n_splits + 1)]


def recency_weights(ids: np.ndarray, half_life: float, order_by: str, time_score: np.ndarray | None = None) -> np.ndarray:
    """Weight halves every `half_life` fractions of the training period, counted back from the end."""
    order = _time_order(ids, order_by, time_score)
    position = np.empty(len(ids), dtype=np.int64)
    position[order] = np.arange(len(ids))
    age = (len(ids) - 1 - position) / len(ids)
    return 0.5 ** (age / half_life)


# --- features -------------------------------------------------------------------------------
def _frequency(train_col: np.ndarray, test_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    both = np.nan_to_num(np.concatenate([train_col, test_col]), nan=np.inf)
    values, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    encoded = counts[inverse].astype(np.float32)
    return encoded[:len(train_col)], encoded[len(train_col):]


def _row_stats(X: np.ndarray) -> np.ndarray:
    present = ~np.isnan(X)
    count = present.sum(axis=1)
    safe = np.maximum(count, 1)
    total = np.nansum(X, axis=1, dtype=np.float64)
    mean = total / safe
    variance = np.maximum(np.nansum(np.square(X, dtype=np.float64), axis=1) / safe - mean ** 2, 0)
    return np.column_stack([X.shape[1] - count, mean, np.sqrt(variance)]).astype(np.float32)


def number_keys(values: np.ndarray) -> np.ndarray:
    """How a number is written, as keys to target-encode. Synthetic data often carries signal there.

    1371.37 gives 1371, 137, 13713 (cut to one decimal), 37 (the two decimals), 41 (4 digits, first one 1) and
    413 (4 digits, first two 13). Computed in float64 from rounded hundredths, because float32 stores 123.45 as
    123.44999...
    """
    hundredths = np.round(np.abs(values.astype(np.float64)) * 100)
    whole = np.floor(hundredths / 100)
    spare_digits = np.floor(np.log10(np.where(whole > 0, whole, 1.0)))
    size = spare_digits + 1
    keys = np.column_stack([whole, np.floor(whole / 10), np.floor(hundredths / 10), hundredths % 100,
                            size * 10 + np.floor(whole / 10.0 ** spare_digits),
                            size * 100 + np.floor(whole / 10.0 ** np.maximum(spare_digits - 1, 0))])
    return keys.astype(np.float32)


def tokenizer_available() -> bool:
    return TOKENIZER_FILE.exists()


def _load_tokenizer():
    if not tokenizer_available():
        raise FileNotFoundError(f"number_tokens needs {TOKENIZER_FILE}, and the README says how to get it")
    return Tokenizer.from_file(str(TOKENIZER_FILE))


def written_form(value: float) -> str:
    """A number the way a CSV writes it: 12.5 and 12.0, never 12.50 or 12."""
    return np.format_float_positional(value, unique=True, trim="0")


def token_keys(train_col: np.ndarray, test_col: np.ndarray, tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """How a language model splits each number into tokens, as keys to target-encode.

    GPT-2 reads 1234.35 as " 12", "34", ".", "35". When a language model wrote the rows, the label can leak through
    these pieces. Codes are shared by train and test, so a token means the same on both sides.
    """
    both = np.nan_to_num(np.concatenate([train_col, test_col]), nan=np.inf)
    values, rows = np.unique(both, return_inverse=True)
    labels = []
    for value in values:
        tokens = tokenizer.encode(" " + written_form(value)).tokens
        before_point = tokens[:tokens.index(".")] if "." in tokens else tokens
        labels.append((tokens[0], tokens[1] if len(tokens) > 1 else "", tokens[-1], str(len(tokens)),
                       f"{tokens[0]}|{len(tokens)}", "".join(before_point)))
    codes = np.column_stack([np.unique(kind, return_inverse=True)[1] for kind in zip(*labels)])
    keys = codes[rows].astype(np.float32)
    return keys[:len(train_col)], keys[len(train_col):]


def _has_decimals(column: np.ndarray) -> bool:
    return bool(np.nanmax(np.abs(column - np.rint(column))) > 0)


def _decimal_columns(data, kept: list[str], X_train: np.ndarray) -> list[str]:
    continuous = [c for c in kept if c not in data.schema.categoricals and data.n_unique[c] >= MIN_KEYED_CARDINALITY
                  and _has_decimals(X_train[:, kept.index(c)])]
    return continuous[:MAX_KEYED_COLUMNS]


class _SharedTokens:
    """Label rates of decimal tokens, pooled over the fields a language model wrote alike.

    One model wrote every field, so a token such as "35" pulls the label partly the same way in each of them. One field
    estimates a rare token poorly, so its rate leans on the token's rate over the whole group. Rows are numbered over
    train then test, so these rates can be asked for rows of either (decision 0010).
    """

    def __init__(self, data, kept: list[str], X_train: np.ndarray, X_test: np.ndarray):
        tokenizer = _load_tokenizer()
        parts = {}
        for column in _decimal_columns(data, kept, X_train):
            position = kept.index(column)
            values, rows = np.unique(np.nan_to_num(np.concatenate([X_train[:, position], X_test[:, position]]), nan=np.inf),
                                     return_inverse=True)
            decimals, last_whole = [], []
            for value in values:
                text = written_form(value)
                decimals.append(text.split(".")[1] if "." in text else "")
                tokens = tokenizer.encode(" " + text).tokens
                last_whole.append(tokens[tokens.index(".") - 1] if "." in tokens else tokens[-1])
            parts[column] = (decimals, last_whole, rows)

        def codes(columns, label):
            table = {}
            return np.column_stack([np.array([table.setdefault(text, len(table)) for text in label(*parts[c][:2])],
                                             dtype=np.int32)[parts[c][2]] for c in columns])

        many = [c for c in parts if data.n_unique[c] > SHARED_MANY_VALUES]
        few = [c for c in parts if data.n_unique[c] <= SHARED_MANY_VALUES]
        self.chains = [codes([c], lambda d, w: [f"{a}|{b}" for a, b in zip(w, d)])[:, 0] for c in many]
        self.groups = []
        if many:  # first, because the chains lean on this group's field rates
            self.groups += [codes(many, lambda d, w: d), codes(many, lambda d, w: [text[:1] for text in d])]
        if few:
            self.groups += [codes(few, lambda d, w: d), codes(few, lambda d, w: w)]

    def rates(self, fit_rows: np.ndarray, y_fit: np.ndarray, other_rows: np.ndarray) -> np.ndarray:
        target = y_fit.astype(np.float64)
        prior = target.mean()
        out, leaned_on = [], []
        for g, codes in enumerate(self.groups):
            levels = int(codes.max()) + 1
            hits = [np.bincount(codes[fit_rows, j], weights=target, minlength=levels) for j in range(codes.shape[1])]
            sizes = [np.bincount(codes[fit_rows, j], minlength=levels) for j in range(codes.shape[1])]
            pooled = (sum(hits) + prior * SHARED_POOL_PRIOR) / (sum(sizes) + SHARED_POOL_PRIOR)
            for j in range(codes.shape[1]):
                out.append(((hits[j] + SHARED_FIELD_PRIOR * pooled) / (sizes[j] + SHARED_FIELD_PRIOR))[codes[other_rows, j]])
                if g == 0:
                    leaned_on.append(out[-1])
        for j, chain in enumerate(self.chains):
            levels = int(chain.max()) + 1
            hits, sizes = np.bincount(chain[fit_rows], weights=target, minlength=levels), np.bincount(chain[fit_rows], minlength=levels)
            out.append((hits[chain[other_rows]] + SHARED_CHAIN_PRIOR * leaned_on[j]) / (sizes[chain[other_rows]] + SHARED_CHAIN_PRIOR))
        return np.column_stack(out).astype(np.float32) if out else np.empty((len(other_rows), 0), dtype=np.float32)

    def out_of_fold(self, fit_rows: np.ndarray, y_fit: np.ndarray, bags: int, seed: int, inner_folds: int) -> np.ndarray:
        """Rates for training rows that cannot see their own label, the same way `_encode_fold` does it."""
        width = self.rates(fit_rows[:1], y_fit[:1], fit_rows[:1]).shape[1]
        encoded = np.zeros((len(fit_rows), width), dtype=np.float32)
        for bag in range(bags):
            for fit, hold in KFold(n_splits=inner_folds, shuffle=True, random_state=seed + bag).split(fit_rows):
                encoded[hold] += self.rates(fit_rows[fit], y_fit[fit], fit_rows[hold]) / bags
        return encoded


def _add_shared_rates(shared: _SharedTokens | None, fit_rows, y_fit, other_rows, X_fit, X_other, features: FeaturePlan, seed: int = 0):
    if shared is None:
        return X_fit, X_other
    fit_side = shared.out_of_fold(fit_rows, y_fit, features.target_encode_bags, seed, features.target_encode_inner_folds)
    return np.hstack([X_fit, fit_side]), np.hstack([X_other, shared.rates(fit_rows, y_fit, other_rows)])


def _shared_tokens(features: FeaturePlan, data, names: list[str], X: np.ndarray, X_test: np.ndarray) -> _SharedTokens | None:
    if not features.shared_token_rates:
        return None
    kept = [n for n in names if n in data.n_unique]
    return _SharedTokens(data, kept, X[:, [names.index(n) for n in kept]], X_test[:, [names.index(n) for n in kept]])


def _low_cardinality(data, limit: int, kept: list[str]) -> list[str]:
    candidates = sorted((c for c in kept if 1 < data.n_unique[c] <= limit), key=lambda c: data.n_unique[c])
    return candidates[:MAX_ENCODED_COLUMNS]


def _logit(rate: np.ndarray) -> np.ndarray:
    rate = np.clip(rate, ENCODING_CLIP, 1 - ENCODING_CLIP)
    return np.log(rate / (1 - rate))


def _pair_codes(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.unique(first * (second.max() + 1) + second, return_inverse=True)[1]


def additive_gap(first: np.ndarray, second: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """How far a pair's label rates sit from what each column says alone, added up in log-odds.

    Returns a likelihood ratio and its degrees of freedom. `first` and `second` are integer codes whose training rows
    come first. Encodings add up in a linear model, so a big gap is what such a model cannot see by itself.
    """
    rows = len(y)
    expected = np.full(rows, -_logit(y.mean()))
    free = 0
    for codes in (first[:rows], second[:rows]):
        counts = np.bincount(codes)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected += _logit(np.bincount(codes, weights=y) / counts)[codes]
        free += np.count_nonzero(counts) - 1
    key = _pair_codes(first, second)[:rows]
    counts = np.bincount(key)
    seen = counts > 0
    hits, counts = np.bincount(key, weights=y)[seen], counts[seen]
    rates = np.bincount(key, weights=1 / (1 + np.exp(-expected)))[seen] / counts
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(hits > 0, hits * np.log(hits / (counts * rates)), 0)
        gap += np.where(counts > hits, (counts - hits) * np.log((counts - hits) / (counts * (1 - rates))), 0)
    return 2 * float(gap.sum()), int(seen.sum() - 1 - free)


def _scan_codes(data, column: str, cut_above: int = MIN_KEYED_CARDINALITY - 1) -> np.ndarray:
    """Integer codes shared by train and test: deciles for a decimal column past `cut_above` values, one code per value otherwise."""
    position = data.schema.features.index(column)
    values = np.concatenate([data.X_train[:, position], data.X_test[:, position]])
    if data.n_unique[column] > cut_above and _has_decimals(values):
        return np.searchsorted(np.unique(np.nanquantile(values, JOINT_CUTS)), values, side="right")
    return np.unique(np.nan_to_num(values, nan=np.inf), return_inverse=True)[1]


def non_additive_pairs(data) -> list[tuple[str, str]]:
    """Column pairs whose label effect does not add up, strongest first. Scanned once per competition.

    ponytail: the scan reads every training label, held-out rows included. It only picks which pairs get a key and
    their encodings stay out of fold (decision 0010). Scanning inside each fold is the way out if a board ever disagrees.
    ponytail: the ratio runs high on cells with about one positive row, so two columns of many whole values can pass as
    noise. Comparing with the same scan on shuffled labels is the way out if such a pair ever gets picked.
    """
    if data.joint_pairs is None:
        codes = {column: _scan_codes(data, column) for column in data.schema.features}
        found = []
        for first, second in itertools.combinations(data.schema.features, 2):
            gap, free = additive_gap(codes[first], codes[second], data.y)
            if free > 0 and gap - free > JOINT_MIN_SIGMAS * np.sqrt(2 * free):
                found.append((gap - free, first, second))
        data.joint_pairs = [(first, second) for _, first, second in sorted(found, reverse=True)]
    return data.joint_pairs


def apply_feature_plan(plan: FeaturePlan, data) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = data.schema.features
    if plan.drop_drifting_top_k and not data.drift_rank:
        raise ValueError("dropping drifting features needs the profile's drift rank first")
    dropped = set(data.drift_rank[:plan.drop_drifting_top_k])
    kept = [f for f in features if f not in dropped]
    positions = [features.index(f) for f in kept]
    X_train = data.X_train[:, positions]
    X_test = data.X_test[:, positions]
    names = list(kept)

    extra_train, extra_test = [], []
    if plan.row_stats:
        extra_train.append(_row_stats(X_train))
        extra_test.append(_row_stats(X_test))
        names += ["row_null_count", "row_mean", "row_std"]
    for column in _low_cardinality(data, plan.frequency_encode_max_cardinality, kept):
        position = kept.index(column)
        train_col, test_col = _frequency(X_train[:, position], X_test[:, position])
        extra_train.append(train_col[:, None])
        extra_test.append(test_col[:, None])
        names.append(f"freq_{column}")
    if plan.number_keys or plan.number_tokens:
        tokenizer = _load_tokenizer() if plan.number_tokens else None
        for column in _decimal_columns(data, kept, X_train):
            position = kept.index(column)
            if plan.number_keys:
                extra_train.append(number_keys(X_train[:, position]))
                extra_test.append(number_keys(X_test[:, position]))
                names += [f"key_{column}_{kind}" for kind in NUMBER_KEY_KINDS]
            if plan.number_tokens:
                train_keys, test_keys = token_keys(X_train[:, position], X_test[:, position], tokenizer)
                extra_train.append(train_keys)
                extra_test.append(test_keys)
                names += [f"key_{column}_{kind}" for kind in TOKEN_KEY_KINDS]
    if plan.joint_keys:
        pairs = [pair for pair in non_additive_pairs(data) if set(pair) <= set(kept)][:plan.joint_keys]
        cut_above = SHARED_MANY_VALUES if plan.joint_by_value else MIN_KEYED_CARDINALITY - 1
        for first, second in pairs:
            codes = _pair_codes(_scan_codes(data, first, cut_above), _scan_codes(data, second, cut_above)).astype(np.float32)
            extra_train.append(codes[:len(X_train), None])
            extra_test.append(codes[len(X_train):, None])
            names.append(f"key_joint_{first}_{second}")
    if extra_train:
        X_train = np.hstack([X_train, *extra_train])
        X_test = np.hstack([X_test, *extra_test])
    keys = [i for i, name in enumerate(names) if name.startswith("key_") and name not in data.n_unique]
    if plan.key_counts and keys:  # an encoding from 3 rows and one from 30,000 look alike until a model sees how many
        counts = [_frequency(X_train[:, i], X_test[:, i]) for i in keys]
        X_train = np.hstack([X_train, np.column_stack([train for train, _ in counts])])
        X_test = np.hstack([X_test, np.column_stack([test for _, test in counts])])
        names += [f"count_{names[i]}" for i in keys]
    return X_train, X_test, names


def target_encode(train_col: np.ndarray, y: np.ndarray, other_col: np.ndarray, smoothing: float = 20.0) -> np.ndarray:
    """Smoothed target mean per value. Null counts as a value and unseen values get the global mean."""
    train_key = np.nan_to_num(train_col, nan=np.inf)
    other_key = np.nan_to_num(other_col, nan=np.inf)
    values, inverse = np.unique(train_key, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(values))
    positives = np.bincount(inverse, weights=y, minlength=len(values))
    prior = float(y.mean())
    means = (positives + smoothing * prior) / (counts + smoothing)
    spot = np.clip(np.searchsorted(values, other_key), 0, len(values) - 1)
    return np.where(values[spot] == other_key, means[spot], prior).astype(np.float32)


def _encode_fold(X: np.ndarray, y: np.ndarray, X_other: np.ndarray, columns: list[int], bags: int = 1, seed: int = 0,
                 inner_folds: int = 5, smoothing: float = 20.0):
    """Target encoding that cannot see its own row: training rows are encoded out of fold.

    Averaging over `bags` different inner splits steadies the encoding of the training rows. `seed` picks the
    splits, so models with different seeds carry different encoding noise and blending them averages it away.
    Fewer `inner_folds` leave more noise in the training encodings, which can keep a model from trusting them.
    """
    if not columns:
        return X, X_other
    encoded = np.zeros((len(X), len(columns)), dtype=np.float32)
    for bag in range(bags):
        inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + bag)
        for i, column in enumerate(columns):
            for fit, hold in inner.split(X):
                encoded[hold, i] += target_encode(X[fit, column], y[fit], X[hold, column], smoothing) / bags
    other = np.column_stack([target_encode(X[:, c], y, X_other[:, c], smoothing) for c in columns])
    return np.hstack([X, encoded]), np.hstack([X_other, other])


def _encode_other(X: np.ndarray, y: np.ndarray, X_other: np.ndarray, columns: list[int], smoothing: float = 20.0) -> np.ndarray:
    """Encoding for rows outside the fit, such as test: they can use every training row directly."""
    if not columns:
        return X_other
    return np.hstack([X_other, np.column_stack([target_encode(X[:, c], y, X_other[:, c], smoothing) for c in columns])])


def _target_encode_positions(plan: FeaturePlan, data, names: list[str]) -> list[int]:
    kept = [n for n in names if n in data.n_unique]
    positions = [names.index(c) for c in _low_cardinality(data, plan.target_encode_max_cardinality, kept)]
    if plan.number_keys or plan.number_tokens or plan.joint_keys:
        positions += [i for i, name in enumerate(names) if name.startswith("key_") and name not in data.n_unique]
    return positions


def _one_hot(codes: np.ndarray, levels: int) -> np.ndarray:
    """One column per category. A null marks none of them."""
    hot = np.zeros((len(codes), levels), dtype=np.float32)
    present = ~np.isnan(codes)
    hot[np.flatnonzero(present), codes[present].astype(np.int64)] = 1
    return hot


def _logistic_view(data, names: list[str]):
    """Turns a model matrix into what a linear model can add up: encodings as log-odds, numbers on a log scale,
    categories one-hot. Raw keys and category codes are labels, not quantities, so they stay out.
    """
    features = data.schema.features
    levels = {c: int(max(np.nanmax(data.X_train[:, features.index(c)]), np.nanmax(data.X_test[:, features.index(c)]))) + 1
              for c in data.schema.categoricals if c in names}
    labels = [i for i, name in enumerate(names)
              if name in levels or (name.startswith(("key_", "count_key_")) and name not in data.n_unique)]
    hots = [(names.index(c), n) for c, n in levels.items() if n <= MAX_ONE_HOT_LEVELS]

    def view(X: np.ndarray) -> np.ndarray:
        numbers = np.delete(X[:, :len(names)], labels, axis=1)
        rates = np.clip(X[:, len(names):], ENCODING_CLIP, 1 - ENCODING_CLIP)
        return np.hstack([np.sign(numbers) * np.log1p(np.abs(numbers)), np.log(rates / (1 - rates)),
                          *[_one_hot(X[:, i], n) for i, n in hots]])
    return view


class _LogisticStart:
    """A logistic model whose logit is where trees start. `reg_lambda` belongs to the trees, so its own penalty is set apart."""

    def __init__(self, view, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None, tolerance: float, penalty: float):
        self.view = view
        self.model = _logistic(penalty, view(X), y, weights, tolerance)

    def margin(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(self.view(X))


class _BoostedLogistic:
    """Trees that start from a logistic model's logit instead of the base rate, so they fit only what it misses."""

    def __init__(self, start: _LogisticStart, trees: xgb.XGBClassifier):
        self.start, self.trees = start, trees

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.trees.predict_proba(X, base_margin=self.start.margin(X))


# --- models ---------------------------------------------------------------------------------
def _logistic(reg_lambda: float, X, y, weights, tolerance: float):
    fitted = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(C=1.0 / max(reg_lambda, 1e-3), max_iter=LOGISTIC_MAX_ITER, tol=tolerance))
    return fitted.fit(X, y, logisticregression__sample_weight=weights)


def _fit(model: ModelPlan, X, y, weights, X_val, y_val, max_trees, patience, device, fixed_trees=None, linear_view=None,
         logistic_tolerance: float = 1e-4, start_penalty: float = START_REG_LAMBDA):
    """Early-stops on the validation fold, or trains a fixed number of trees when refitting."""
    p: ModelParams = model.params
    trees = fixed_trees or max_trees
    evaluate = fixed_trees is None

    if model.family == "logistic":
        return _logistic(p.reg_lambda, X, y, weights, logistic_tolerance), 1

    if model.family == "lightgbm":
        fitted = lgb.LGBMClassifier(
            n_estimators=trees, learning_rate=p.learning_rate, max_depth=p.max_depth, num_leaves=p.num_leaves,
            subsample=p.subsample, subsample_freq=1, colsample_bytree=p.colsample, reg_lambda=p.reg_lambda,
            min_child_samples=p.min_child, metric="auc", random_state=model.seed, n_jobs=-1, verbose=-1)
        extra = {"eval_X": X_val, "eval_y": y_val,
                 "callbacks": [lgb.early_stopping(patience, verbose=False)]} if evaluate else {}
        fitted.fit(X, y, sample_weight=weights, **extra)
        return fitted, (fitted.best_iteration_ or trees) if evaluate else trees

    if model.family in ("xgboost", "logistic_boost"):
        start = _LogisticStart(linear_view, X, y, weights, logistic_tolerance, start_penalty) if model.family == "logistic_boost" else None
        fitted = xgb.XGBClassifier(
            n_estimators=trees, learning_rate=p.learning_rate, max_depth=p.max_depth, subsample=p.subsample,
            colsample_bytree=p.colsample, reg_lambda=p.reg_lambda, min_child_weight=p.min_child, tree_method="hist",
            device=device, eval_metric="auc", early_stopping_rounds=patience if evaluate else None,
            random_state=model.seed, n_jobs=-1, verbosity=0)
        extra = {"eval_set": [(X_val, y_val)], "verbose": False} if evaluate else {}
        if start is not None:
            extra["base_margin"] = start.margin(X)
            if evaluate:
                extra["base_margin_eval_set"] = [start.margin(X_val)]
        fitted.fit(X, y, sample_weight=weights, **extra)
        best = (fitted.best_iteration + 1) if evaluate else trees
        return (fitted if start is None else _BoostedLogistic(start, fitted)), best

    settings = dict(
        iterations=trees, learning_rate=p.learning_rate, depth=p.max_depth, l2_leaf_reg=max(p.reg_lambda, 1e-3),
        bootstrap_type="Bernoulli", subsample=p.subsample, random_seed=model.seed, eval_metric="AUC",
        verbose=False, allow_writing_files=False)
    if device == "cuda":
        settings.update(task_type="GPU", devices="0")  # catboost has no column sampling on GPU for this loss
    else:
        settings.update(rsm=p.colsample, thread_count=-1)
    if evaluate:
        settings.update(od_type="Iter", od_wait=patience, use_best_model=True)
    fitted = catboost.CatBoostClassifier(**settings)
    fitted.fit(X, y, sample_weight=weights, **({"eval_set": (X_val, y_val)} if evaluate else {}))
    return fitted, (fitted.get_best_iteration() + 1) if evaluate else trees


def _predict(fitted, X) -> np.ndarray:
    return fitted.predict_proba(X)[:, 1].astype(np.float32)


def _fit_weights(weights: np.ndarray | None, rows: np.ndarray) -> np.ndarray | None:
    """Early folds hold only old rows: rescaled, their weights stay usable instead of nearly zero."""
    return None if weights is None else weights[rows] / weights[rows].max()


def detect_device() -> str:
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def run_experiment(validation: ValidationPlan, features: FeaturePlan, model: ModelPlan, data,
                   max_trees: int = 8000, patience: int = 200, device: str = "auto") -> ExperimentResult:
    started = time.monotonic()
    device = detect_device() if device == "auto" else device
    X, X_test, names = apply_feature_plan(features, data)
    encoded = _target_encode_positions(features, data, names)
    view = _logistic_view(data, names) if model.family == "logistic" else (lambda matrix: matrix)
    linear_view = _logistic_view(data, names) if model.family == "logistic_boost" else None
    shared = _shared_tokens(features, data, names, X, X_test)
    y = data.y
    test_rows = np.arange(len(y), len(y) + len(X_test))
    weights = None
    if model.recency_half_life is not None:
        weights = recency_weights(data.train_ids, model.recency_half_life, validation.order_by, data.time_score)

    oof = np.full(len(y), np.nan, dtype=np.float32)
    mask = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(X_test), dtype=np.float64)
    fold_scores, best_iters = [], []
    folds = make_folds(validation, y, data.train_ids, data.time_score)
    # Time folds are scored on the block nearest the test period only. Pooling blocks would rank rows
    # across periods, so a drifting target rate would look like skill.
    scored_folds = folds[-1:] if validation.scheme == "time_kfold" else folds
    for _, val_rows in scored_folds:
        mask[val_rows] = True
    for train_rows, val_rows in folds:
        X_fit, X_val = _encode_fold(X[train_rows], y[train_rows], X[val_rows], encoded, features.target_encode_bags,
                                    model.seed, features.target_encode_inner_folds, features.target_encode_smoothing)
        X_fit, X_val = _add_shared_rates(shared, train_rows, y[train_rows], val_rows, X_fit, X_val, features, model.seed)
        X_fit, X_val = view(X_fit), view(X_val)
        fitted, best = _fit(model, X_fit, y[train_rows], _fit_weights(weights, train_rows),
                            X_val, y[val_rows], max_trees, patience, device, linear_view=linear_view,
                            logistic_tolerance=features.logistic_tolerance, start_penalty=features.logistic_start_penalty)
        oof[val_rows] = _predict(fitted, X_val)
        fold_scores.append(float(roc_auc_score(y[val_rows], oof[val_rows])))
        best_iters.append(int(best))
        if validation.scheme == "stratified_kfold":
            X_test_fold = _encode_other(X[train_rows], y[train_rows], X_test, encoded, features.target_encode_smoothing)
            if shared is not None:
                X_test_fold = np.hstack([X_test_fold, shared.rates(train_rows, y[train_rows], test_rows)])
            test += _predict(fitted, view(X_test_fold)) / len(folds)
        del fitted, X_fit, X_val
        gc.collect()

    X_all, X_test_all = _encode_fold(X, y, X_test, encoded, features.target_encode_bags, model.seed,
                                     features.target_encode_inner_folds, features.target_encode_smoothing)
    X_all, X_test_all = _add_shared_rates(shared, np.arange(len(y)), y, test_rows, X_all, X_test_all, features, model.seed)
    X_all, X_test_all = view(X_all), view(X_test_all)
    trees = max(int(np.mean(best_iters) * REFIT_TREE_FACTOR), 1)
    fitted, _ = _fit(model, X_all, y, _fit_weights(weights, np.arange(len(y))), None, None, max_trees, patience,
                     device, fixed_trees=trees, linear_view=linear_view, logistic_tolerance=features.logistic_tolerance,
                     start_penalty=features.logistic_start_penalty)
    refit = _predict(fitted, X_test_all)
    # On time folds only the refit saw the newest rows, and test comes after those. On random folds the fold models
    # are a bag and the refit saw every row: on a holdout their ranks together beat either one (decision 0009).
    test = refit if validation.scheme == "time_kfold" else (rankdata(refit) + rankdata(test)) / (2 * len(refit))

    return ExperimentResult(
        oof=oof, test=np.asarray(test, dtype=np.float32), mask=mask,
        cv=float(roc_auc_score(y[mask], oof[mask])), std=float(np.std(fold_scores)),
        fold_scores=fold_scores, best_iters=best_iters, seconds=round(time.monotonic() - started, 1))


def save_predictions(path, result: ExperimentResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, oof=result.oof, test=result.test, mask=result.mask)


def load_predictions(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return saved["oof"], saved["test"], saved["mask"]


# --- leak canary ----------------------------------------------------------------------------
def leak_canary(validation: ValidationPlan, features: FeaturePlan, data, seed: int = 0) -> dict:
    """Runs the real pipeline on shuffled labels. Any AUC above chance means information is leaking."""
    rng = np.random.default_rng(seed)
    n = len(data.y)
    rows = np.sort(rng.choice(n, min(n, CANARY_SAMPLE), replace=False))
    y = rng.permutation(data.y[rows])
    X, X_test, names = apply_feature_plan(features, data)
    shared = _shared_tokens(features, data, names, X, X_test)
    X = X[rows]
    encoded = _target_encode_positions(features, data, names)
    model = ModelPlan(
        family="lightgbm", seed=seed, hypothesis="Canary model for the shuffled label check.",
        params=ModelParams(learning_rate=0.1, max_depth=6, num_leaves=31, subsample=0.9, colsample=0.5,
                           reg_lambda=1.0, min_child=5))

    oof = np.full(len(y), np.nan)
    time_score = None if data.time_score is None else data.time_score[rows]
    for train_rows, val_rows in make_folds(validation, y, data.train_ids[rows], time_score):
        X_fit, X_val = _encode_fold(X[train_rows], y[train_rows], X[val_rows], encoded, features.target_encode_bags,
                                    inner_folds=features.target_encode_inner_folds, smoothing=features.target_encode_smoothing)
        X_fit, X_val = _add_shared_rates(shared, rows[train_rows], y[train_rows], rows[val_rows], X_fit, X_val, features)
        fitted, _ = _fit(model, X_fit, y[train_rows], None, None, None, 0, 0, "cpu", fixed_trees=CANARY_TREES)
        oof[val_rows] = _predict(fitted, X_val)
    scored = ~np.isnan(oof)
    auc = float(roc_auc_score(y[scored], oof[scored]))

    positives = int(y[scored].sum())
    negatives = int(scored.sum()) - positives
    chance_sd = np.sqrt((positives + negatives + 1) / (12 * positives * negatives))
    return {"auc": round(auc, 4), "passed": bool(abs(auc - 0.5) <= max(0.01, 4 * chance_sd))}
