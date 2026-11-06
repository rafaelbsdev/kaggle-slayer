"""Breaks the code on purpose and confirms the tests notice. Works on a throwaway copy of the repo."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TIMEOUT = 1500  # the end-to-end tests run real training, and a busy machine can take many minutes

MUTATIONS = [
    ("verifier", "ok = drift < DRIFT_AUC and not (temporal and any(_order_evidence(ctx).values()))", "ok = True", "tests/test_verifier.py"),
    ("verifier", '("order_is_time", _order_evidence(ctx)[out.order_by],', '("order_is_time", True,', "tests/test_verifier.py"),
    ("verifier", "ok = temporal or drift >= TEMPORAL_EVIDENCE_AUC", "ok = True", "tests/test_verifier.py"),
    ("verifier", 'checks.append(("no_repeated_experiment", exp_id not in seen,', 'checks.append(("no_repeated_experiment", True,', "tests/test_verifier.py"),
    ("verifier", 'params.num_leaves <= 2 ** params.max_depth,', 'True,', "tests/test_verifier.py"),
    ("verifier", 'params.max_depth <= CATBOOST_MAX_DEPTH,', 'True,', "tests/test_verifier.py"),
    ("verifier", 'ctx["validation"].scheme == "time_kfold",', 'True,', "tests/test_verifier.py"),
    ("verifier", '("candidate_exists", candidate is not None,', '("candidate_exists", True,', "tests/test_verifier.py"),
    ("verifier", '("quota_left", ctx["quota_left"] > 0,', '("quota_left", True,', "tests/test_verifier.py"),
    ("verifier", '("leak_canary_passed", ctx["canary_passed"],', '("leak_canary_passed", True,', "tests/test_verifier.py"),
    ("verifier", '("submission_file_valid", ctx["file_ok"],', '("submission_file_valid", True,', "tests/test_verifier.py"),
    ("verifier", '("cv_gain_above_noise", gain_ok,', '("cv_gain_above_noise", True,', "tests/test_verifier.py"),
    ("verifier", 'ok = cited is not None and abs(cited - observed) <= GAP_TOLERANCE', 'ok = True', "tests/test_verifier.py"),
    ("verifier", 'kept >= MIN_FEATURES_KEPT * n_features', 'True', "tests/test_verifier.py"),
    ("verifier", '("target_is_binary", ctx["target_is_binary"],', '("target_is_binary", True,', "tests/test_verifier.py"),
    ("verifier", 'except Exception as exc:  # a provider outage', 'except ZeroDivisionError as exc:  # a provider outage', "tests/test_verifier.py"),
    ("privacy", 'if str(key).lower() in FORBIDDEN_KEYS:', 'if False:', "tests/test_privacy.py"),
    ("privacy", 'if size > MAX_PAYLOAD_CHARS:', 'if False:', "tests/test_privacy.py"),
    ("ledger", 'if handle.read(1) != b"\\n":', 'if False:', "tests/test_ledger.py"),
    ("ledger", '.decode("utf-8", errors="replace").split("\\n"):', '.decode("utf-8", errors="replace").splitlines():', "tests/test_ledger.py"),
    ("ledger", '.decode("utf-8", errors="replace")', '.decode("utf-8")', "tests/test_ledger.py"),
    ("submission", 'if np.isnan(values).any():', 'if False:', "tests/test_submission.py"),
    ("submission", 'if finite.size and (finite.min() < 0 or finite.max() > 1):', 'if False:', "tests/test_submission.py"),
    ("submission", 'if submission[id_col].n_unique() != submission.height or set(submission[id_col]) != set(sample[id_col]):', 'if False:', "tests/test_submission.py"),
    ("kaggle_client", 'if not SLUG.match(slug):', 'if False:', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'timeout=SCORING_WAIT + 120, retries=1)', 'timeout=SCORING_WAIT + 120)', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'if RULES.search(last_error):', 'if False:', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'if UNAUTHENTICATED.search(last_error):', 'if False:', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'if not SUBMISSION_REF.search(exc.output):', 'if True:', "tests/test_kaggle_client.py"),
    ("kaggle_client", '"-p", str(partial), "--unzip"', '"-p", str(dest), "--unzip"', "tests/test_kaggle_client.py"),
    ("kaggle_client", '            shutil.rmtree(partial)\n', '            pass\n', "tests/test_kaggle_client.py"),
    ("kaggle_client", '"user_rank": rank or None,', '"user_rank": rank,', "tests/test_kaggle_client.py"),
    ("data", 'if len(targets) != 1:', 'if False:', "tests/test_data.py"),
    ("data", 'data.drift_rank = [features[i] for i in drift_order]', 'data.drift_rank = list(features)', "tests/test_data.py"),
    ("data", 'return bool(TEMPORAL_WORDS.search(overview or ""))', 'return False', "tests/test_data.py"),
    ("data", 'sink_parquet(partial)', 'sink_parquet(parquet_path)', "tests/test_data.py"),
    ("data", 'if max(above, below) >= CLOCK_MIN_BEYOND:', 'if True:', "tests/test_data.py"),
    ("data", 'if np.isnan(values).mean() > CLOCK_MAX_NULLS or not future.size:', 'if not future.size:', "tests/test_data.py"),
    ("data", 'return int(column), 1 if above >= below else -1, max(above, below)', 'return int(column), 1, max(above, below)', "tests/test_data.py"),
    ("data", 'np.where(np.isnan(values), -np.inf, values * clock[1])', 'values * clock[1]', "tests/test_data.py"),
    ("data", 'if auc_k < NO_DRIFT_AUC:', 'if False:', "tests/test_data.py"),
    ("training", 'scored_folds = folds[-1:] if validation.scheme == "time_kfold" else folds', 'scored_folds = folds', "tests/test_training.py"),
    ("training", 'return None if weights is None else weights[rows] / weights[rows].max()', 'return None if weights is None else weights[rows]', "tests/test_training.py"),
    ("training", 'min_child_samples=p.min_child, metric="auc", random_state', 'min_child_samples=p.min_child, random_state', "tests/test_training.py"),
    ("training", 'return np.argsort(time_score, kind="stable")', 'return np.arange(len(ids))', "tests/test_training.py"),
    ("training", 'return [(np.concatenate(blocks[:k]), blocks[k]) for k in range(1, plan.n_splits + 1)]',
     'return [(np.concatenate(blocks[:k + 1]), blocks[k]) for k in range(1, plan.n_splits + 1)]', "tests/test_training.py"),
    ("training", 'encoded[hold, i] += target_encode(X[fit, column], y[fit], X[hold, column], smoothing) / bags',
     'encoded[hold, i] += target_encode(X[:, column], y, X[hold, column], smoothing) / bags', "tests/test_training.py"),
    ("training", 'return {"auc": round(auc, 4), "passed": bool(abs(auc - 0.5) <= max(0.01, 4 * chance_sd))}',
     'return {"auc": round(auc, 4), "passed": True}', "tests/test_training.py"),
    ("training", 'y = rng.permutation(data.y[rows])', 'y = data.y[rows]', "tests/test_training.py"),
    ("training", 'dropped = set(data.drift_rank[:plan.drop_drifting_top_k])', 'dropped = set()', "tests/test_training.py"),
    ("training", 'test = refit if validation.scheme == "time_kfold" else (rankdata(refit) + rankdata(test)) / (2 * len(refit))', 'test = rankdata(test)', "tests/test_training.py"),
    ("training", 'test = refit if validation.scheme == "time_kfold" else (rankdata(refit) + rankdata(test)) / (2 * len(refit))', 'test = refit', "tests/test_training.py"),
    ("training", 'unique=True, trim="0")', 'unique=True, trim="-")', "tests/test_training.py"),
    ("training", 'before_point = tokens[:tokens.index(".")] if "." in tokens else tokens', 'before_point = tokens', "tests/test_training.py"),
    ("training", 'both = np.nan_to_num(np.concatenate([train_col, test_col]), nan=np.inf)\n    values, rows', 'both = np.nan_to_num(train_col, nan=np.inf)\n    values, rows', "tests/test_training.py"),
    ("training", 'if plan.number_keys or plan.number_tokens or plan.joint_keys:\n        positions', 'if plan.number_keys or plan.joint_keys:\n        positions', "tests/test_training.py"),
    ("verifier", 'checks.append(("tokenizer_present", ctx["tokenizer_available"],', 'checks.append(("tokenizer_present", True,', "tests/test_verifier.py"),
    ("verifier", 'checks.append(("joint_scan_fits", n_features <= MAX_JOINT_SCAN_COLUMNS,', 'checks.append(("joint_scan_fits", True,', "tests/test_verifier.py"),
    ("verifier", 'if out.joint_keys:', 'if True:', "tests/test_verifier.py"),
    ("training", 'expected = np.full(rows, -_logit(y.mean()))', 'expected = np.zeros(rows)', "tests/test_training.py"),
    ("training", 'for codes in (first[:rows], second[:rows]):', 'for codes in (first, second):', "tests/test_training.py"),
    ("training", 'free += np.count_nonzero(counts) - 1', 'free += len(counts) - 1', "tests/test_training.py"),
    ("training", 'key = _pair_codes(first, second)[:rows]', 'key = _pair_codes(first, second)', "tests/test_training.py"),
    ("training", 'return 2 * float(gap.sum()), int(seen.sum() - 1 - free)', 'return 2 * float(gap.sum()), int(seen.sum() - 1)', "tests/test_training.py"),
    ("training", 'if free > 0 and gap - free > JOINT_MIN_SIGMAS * np.sqrt(2 * free):', 'if gap - free > JOINT_MIN_SIGMAS * np.sqrt(2 * abs(free)):', "tests/test_training.py"),
    ("training", 'if free > 0 and gap - free > JOINT_MIN_SIGMAS * np.sqrt(2 * free):', 'if free > 0 and gap - free > 0:', "tests/test_training.py"),
    ("training", 'sorted(found, reverse=True)', 'found', "tests/test_training.py"),
    ("training", 'if data.joint_pairs is None:', 'if True:', "tests/test_training.py"),
    ("training", 'if data.n_unique[column] > cut_above and _has_decimals(values):', 'if _has_decimals(values):', "tests/test_training.py"),
    ("training", 'if data.n_unique[column] > cut_above and _has_decimals(values):', 'if data.n_unique[column] > cut_above:', "tests/test_training.py"),
    ("training", 'cut_above = SHARED_MANY_VALUES if plan.joint_by_value else MIN_KEYED_CARDINALITY - 1', 'cut_above = MIN_KEYED_CARDINALITY - 1', "tests/test_training.py"),
    ("training", 'cut_above = SHARED_MANY_VALUES if plan.joint_by_value else MIN_KEYED_CARDINALITY - 1', 'cut_above = SHARED_MANY_VALUES', "tests/test_training.py"),
    ("training", 'if set(pair) <= set(kept)][:plan.joint_keys]', 'if set(pair) <= set(kept)]', "tests/test_training.py"),
    ("training", 'if set(pair) <= set(kept)][:plan.joint_keys]', '][:plan.joint_keys]', "tests/test_training.py"),
    ("training", 'if plan.number_keys or plan.number_tokens or plan.joint_keys:\n        positions', 'if plan.number_keys or plan.number_tokens:\n        positions', "tests/test_training.py"),
    ("training", 'extra_test.append(codes[len(X_train):, None])', 'extra_test.append(codes[:len(X_test), None] * 0)', "tests/test_training.py"),
    ("training", 'if plan.key_counts and keys:', 'if keys:', "tests/test_training.py"),
    ("training", 'if plan.key_counts and keys:', 'if plan.key_counts:', "tests/test_training.py"),
    ("training", 'keys = [i for i, name in enumerate(names) if name.startswith("key_") and name not in data.n_unique]\n    if plan.key_counts', 'keys = [i for i, name in enumerate(names) if name.startswith("key_")]\n    if plan.key_counts', "tests/test_training.py"),
    ("training", 'name.startswith(("key_", "count_key_"))', 'name.startswith("key_")', "tests/test_training.py"),
    ("training", 'pooled = (sum(hits) + prior * SHARED_POOL_PRIOR) / (sum(sizes) + SHARED_POOL_PRIOR)', 'pooled = np.full(levels, prior)', "tests/test_training.py"),
    ("training", '((hits[j] + SHARED_FIELD_PRIOR * pooled) / (sizes[j] + SHARED_FIELD_PRIOR))', 'pooled', "tests/test_training.py"),
    ("training", 'out.append((hits[chain[other_rows]] + SHARED_CHAIN_PRIOR * leaned_on[j]) / (sizes[chain[other_rows]] + SHARED_CHAIN_PRIOR))', 'out.append((hits[chain[other_rows]] + SHARED_CHAIN_PRIOR * prior) / (sizes[chain[other_rows]] + SHARED_CHAIN_PRIOR))', "tests/test_training.py"),
    ("training", 'many = [c for c in parts if data.n_unique[c] > SHARED_MANY_VALUES]', 'many = list(parts)', "tests/test_training.py"),
    ("training", 'codes(many, lambda d, w: [text[:1] for text in d])', 'codes(many, lambda d, w: d)', "tests/test_training.py"),
    ("training", 'codes(few, lambda d, w: w)', 'codes(few, lambda d, w: d)', "tests/test_training.py"),
    ("training", 'encoded[hold] += self.rates(fit_rows[fit], y_fit[fit], fit_rows[hold]) / bags', 'encoded[hold] += self.rates(fit_rows, y_fit, fit_rows[hold]) / bags', "tests/test_training.py"),
    ("training", 'for fit, hold in KFold(n_splits=inner_folds, shuffle=True, random_state=seed + bag).split(fit_rows):', 'for fit, hold in KFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(fit_rows):', "tests/test_training.py"),
    ("training", 'for fit, hold in KFold(n_splits=inner_folds, shuffle=True, random_state=seed + bag).split(fit_rows):', 'for fit, hold in KFold(n_splits=3, shuffle=True, random_state=seed + bag).split(fit_rows):', "tests/test_training.py"),
    ("training", 'test_rows = np.arange(len(y), len(y) + len(X_test))', 'test_rows = np.arange(len(X_test))', "tests/test_training.py"),
    ("training", 'if not features.shared_token_rates:\n        return None', 'if False:\n        return None', "tests/test_training.py"),
    ("training", 'kept = [n for n in names if n in data.n_unique]\n    return _SharedTokens', 'kept = data.schema.features\n    return _SharedTokens', "tests/test_training.py"),
    ("training", 'X_fit, X_val = _add_shared_rates(shared, train_rows, y[train_rows], val_rows, X_fit, X_val, features, model.seed)\n', '', "tests/test_training.py"),
    ("training", 'X_all, X_test_all = _add_shared_rates(shared, np.arange(len(y)), y, test_rows, X_all, X_test_all, features, model.seed)\n', '', "tests/test_training.py"),
    ("training", 'shared = _shared_tokens(features, data, names, X, X_test)\n    X = X[rows]', 'shared = None\n    X = X[rows]', "tests/test_training.py"),
    ("verifier", 'if out.number_tokens or out.shared_token_rates:', 'if out.number_tokens:', "tests/test_verifier.py"),
    ("ensemble", 'START_PICKS = 3', 'START_PICKS = 1', "tests/test_ensemble.py"),
    ("ensemble", 'current = START_PICKS * ranks[first]', 'current = ranks[first].copy()', "tests/test_ensemble.py"),
    ("ensemble", 'searched = strongest + [plan for plan in bags if plan.startswith(FAMILY)]', 'searched = strongest', "tests/test_ensemble.py"),
    ("ensemble", 'key=lambda name: -_fast_auc(y_scored, ranks[name]))[:FAMILY_BAG]', 'key=lambda name: -_fast_auc(y_scored, ranks[name]))', "tests/test_ensemble.py"),
    ("ensemble", 'strongest = sorted(names, key=lambda name: -_fast_auc(y_scored, ranks[name]))[:FAMILY_BAG]', 'strongest = names[-FAMILY_BAG:]', "tests/test_ensemble.py"),
    ("orchestrator", 'plans=plans, families=families)', 'plans=plans)', "tests/test_orchestrator.py"),
    ("training", 'extra["base_margin"] = start.margin(X)', 'pass', "tests/test_training.py"),
    ("training", 'extra["base_margin_eval_set"] = [start.margin(X_val)]', 'pass', "tests/test_training.py"),
    ("training", 'self.trees.predict_proba(X, base_margin=self.start.margin(X))', 'self.trees.predict_proba(X)', "tests/test_training.py"),
    ("training", 'linear_view = _logistic_view(data, names) if model.family == "logistic_boost" else None', 'linear_view = lambda matrix: matrix', "tests/test_training.py"),
    ("kaggle_client", '        renew_expiring_token()\n', '', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'return datetime.fromisoformat(expiry) - datetime.now(timezone.utc) < TOKEN_MARGIN', 'return True', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'return datetime.fromisoformat(expiry) - datetime.now(timezone.utc) < TOKEN_MARGIN', 'return False', "tests/test_kaggle_client.py"),
    ("kaggle_client", 'except (OSError, ValueError, KeyError, TypeError):', 'except OSError:', "tests/test_kaggle_client.py"),
    ("kaggle_client", "except OSError:  # requests' errors", "except ZeroDivisionError:  # requests' errors", "tests/test_kaggle_client.py"),
    ("kaggle_client", '            if credentials:\n', '            if True:\n', "tests/test_kaggle_client.py"),
    ("kaggle_client", '|credentials|authentication required"', '|credentials"', "tests/test_kaggle_client.py"),
    ("data", 'if train[c].null_count() == train.height or test[c].null_count() == test.height]', 'if train[c].null_count() == train.height and test[c].null_count() == test.height]', "tests/test_data.py"),
    ("data", 'schema.features = [c for c in schema.features if c not in schema.empty]', 'pass', "tests/test_data.py"),
    ("data", '    if not schema.features:\n', '    if False:\n', "tests/test_data.py"),
    ("data", '"n_features_dropped_as_empty": len(data.schema.empty),', '"n_features_dropped_as_empty": 0,', "tests/test_data.py"),
    ("training", 'target_encode(X[fit, column], y[fit], X[hold, column], smoothing) / bags', 'target_encode(X[fit, column], y[fit], X[hold, column], smoothing)', "tests/test_training.py"),
    ("training", 'inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + bag)', 'inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)', "tests/test_training.py"),
    ("training", 'positions += [i for i, name in enumerate(names) if name.startswith("key_") and name not in data.n_unique]', 'pass', "tests/test_training.py"),
    ("training", 'return continuous[:MAX_KEYED_COLUMNS]', 'return continuous', "tests/test_training.py"),
    ("training", 'if c not in data.schema.categoricals and data.n_unique[c] >= MIN_KEYED_CARDINALITY', 'if c not in data.schema.categoricals', "tests/test_training.py"),
    ("training", 'and _has_decimals(X_train[:, kept.index(c)])]', ']', "tests/test_training.py"),
    ("training", 'hundredths % 100,', 'hundredths % 10,', "tests/test_training.py"),
    ("training", 'np.floor(hundredths / 10), ', 'np.floor(hundredths / 10) % 10, ', "tests/test_training.py"),
    ("training", 'size * 100 + ', '', "tests/test_training.py"),
    ("training", 'inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed + bag)', 'inner = KFold(n_splits=inner_folds, shuffle=True, random_state=7 + bag)', "tests/test_training.py"),
    ("training", 'inner = KFold(n_splits=inner_folds,', 'inner = KFold(n_splits=5,', "tests/test_training.py"),
    ("training", """model.seed, features.target_encode_inner_folds, features.target_encode_smoothing)
        X_fit, X_val""", """model.seed, smoothing=features.target_encode_smoothing)
        X_fit, X_val""", "tests/test_training.py"),
    ("training", """inner_folds=features.target_encode_inner_folds, smoothing=features.target_encode_smoothing)
        X_fit, X_val = _add_shared_rates(shared, rows[train_rows]""", """smoothing=features.target_encode_smoothing)
        X_fit, X_val = _add_shared_rates(shared, rows[train_rows]""", "tests/test_training.py"),
    ("training", 'encoded[hold, i] += target_encode(X[fit, column], y[fit], X[hold, column], smoothing) / bags', 'encoded[hold, i] += target_encode(X[fit, column], y[fit], X[hold, column]) / bags', "tests/test_training.py"),
    ("training", 'other = np.column_stack([target_encode(X[:, c], y, X_other[:, c], smoothing) for c in columns])', 'other = np.column_stack([target_encode(X[:, c], y, X_other[:, c]) for c in columns])', "tests/test_training.py"),
    ("training", 'return np.hstack([X_other, np.column_stack([target_encode(X[:, c], y, X_other[:, c], smoothing) for c in columns])])', 'return np.hstack([X_other, np.column_stack([target_encode(X[:, c], y, X_other[:, c]) for c in columns])])', "tests/test_training.py"),
    ("training", 'model.seed, features.target_encode_inner_folds, features.target_encode_smoothing)\n        X_fit, X_val', 'model.seed, features.target_encode_inner_folds)\n        X_fit, X_val', "tests/test_training.py"),
    ("training", 'X_test, encoded, features.target_encode_smoothing)', 'X_test, encoded)', "tests/test_training.py"),
    ("training", """features.target_encode_inner_folds, features.target_encode_smoothing)
    X_all, X_test_all""", """features.target_encode_inner_folds)
    X_all, X_test_all""", "tests/test_training.py"),
    ("training", 'inner_folds=features.target_encode_inner_folds, smoothing=features.target_encode_smoothing)', 'inner_folds=features.target_encode_inner_folds)', "tests/test_training.py"),
    ("training", 'max_iter=LOGISTIC_MAX_ITER, tol=tolerance))', 'max_iter=LOGISTIC_MAX_ITER, tol=1e-4))', "tests/test_training.py"),
    ("training", '_logistic(p.reg_lambda, X, y, weights, logistic_tolerance)', '_logistic(p.reg_lambda, X, y, weights, 1e-4)', "tests/test_training.py"),
    ("training", '_LogisticStart(linear_view, X, y, weights, logistic_tolerance, start_penalty)', '_LogisticStart(linear_view, X, y, weights, 1e-4, start_penalty)', "tests/test_training.py"),
    ("training", '_LogisticStart(linear_view, X, y, weights, logistic_tolerance, start_penalty)', '_LogisticStart(linear_view, X, y, weights, logistic_tolerance, START_REG_LAMBDA)', "tests/test_training.py"),
    ("training", 'self.model = _logistic(penalty, view(X), y, weights, tolerance)', 'self.model = _logistic(penalty, view(X), y, weights, 1e-4)', "tests/test_training.py"),
    ("training", 'self.model = _logistic(penalty, view(X), y, weights, tolerance)', 'self.model = _logistic(START_REG_LAMBDA, view(X), y, weights, tolerance)', "tests/test_training.py"),
    ("training", 'linear_view=linear_view,\n                            logistic_tolerance=features.logistic_tolerance, start_penalty=features.logistic_start_penalty)', 'linear_view=linear_view,\n                            start_penalty=features.logistic_start_penalty)', "tests/test_training.py"),
    ("training", 'linear_view=linear_view,\n                            logistic_tolerance=features.logistic_tolerance, start_penalty=features.logistic_start_penalty)', 'linear_view=linear_view,\n                            logistic_tolerance=features.logistic_tolerance)', "tests/test_training.py"),
    ("training", 'fixed_trees=trees, linear_view=linear_view, logistic_tolerance=features.logistic_tolerance,\n                     start_penalty=features.logistic_start_penalty)', 'fixed_trees=trees, linear_view=linear_view,\n                     start_penalty=features.logistic_start_penalty)', "tests/test_training.py"),
    ("training", 'fixed_trees=trees, linear_view=linear_view, logistic_tolerance=features.logistic_tolerance,\n                     start_penalty=features.logistic_start_penalty)', 'fixed_trees=trees, linear_view=linear_view, logistic_tolerance=features.logistic_tolerance)', "tests/test_training.py"),
    ("training", 'view = _logistic_view(data, names) if model.family == "logistic" else (lambda matrix: matrix)', 'view = lambda matrix: matrix', "tests/test_training.py"),
    ("training", 'np.log(rates / (1 - rates)),', 'rates,', "tests/test_training.py"),
    ("training", 'if name in levels or (name.startswith(("key_", "count_key_")) and name not in data.n_unique)]', 'if False]', "tests/test_training.py"),
    ("training", 'hots = [(names.index(c), n) for c, n in levels.items() if n <= MAX_ONE_HOT_LEVELS]', 'hots = []', "tests/test_training.py"),
    ("training", 'hots = [(names.index(c), n) for c, n in levels.items() if n <= MAX_ONE_HOT_LEVELS]', 'hots = [(names.index(c), n) for c, n in levels.items()]', "tests/test_training.py"),
    ("training", 'continuous = [c for c in kept if c not in data.schema.categoricals and', 'continuous = [c for c in kept if', "tests/test_training.py"),
    ("training", 'make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),', 'make_pipeline(StandardScaler(),', "tests/test_training.py"),
    ("training", 'fitted.fit(X, y, logisticregression__sample_weight=weights)', 'fitted.fit(X, y)', "tests/test_training.py"),
    ("ensemble", 'sample = fit if len(fit) <= FIT_SAMPLE else rng.choice(fit, FIT_SAMPLE, replace=False)',
     'sample = np.arange(len(y_scored))', "tests/test_ensemble.py"),
    ("ensemble", 'if trials[name] - best < MIN_GAIN:', 'if False:', "tests/test_ensemble.py"),
    ("ensemble", 'seeds_of[(plans or {}).get(name, name)].append(name)', 'seeds_of[name].append(name)', "tests/test_ensemble.py"),
    ("ensemble", 'for plan in searched}, y)', 'for plan in bags}, y)', "tests/test_ensemble.py"),
    ("ensemble", 'members[name] += weight / len(seeds_of[plan])', 'members[name] += weight', "tests/test_ensemble.py"),
    ("brains", 'node["required"] = sorted(node["properties"])', 'pass', "tests/test_brains.py"),
    ("brains", '"schema": _answer_schema(schema)}', '"schema": anthropic.transform_schema(schema)}', "tests/test_brains.py"),
    ("brains", """            for value in node.values():
                require_all(value)
""", "", "tests/test_brains.py"),
    ("brains", 'if step >= HEURISTIC_SCHEDULE_ROUNDS and measured:', 'if False:', "tests/test_brains.py"),
    ("brains", 'if step >= HEURISTIC_SCHEDULE_ROUNDS and measured:', 'if measured:', "tests/test_brains.py"),
    ("brains", 'best = max(measured, key=lambda experiment: experiment["cv"])', 'best = measured[-1]', "tests/test_brains.py"),
    ("brains", '        assert_no_rows(payload)\n', '', "tests/test_brains.py"),
    ("brains", '        self.budget.check()\n', '', "tests/test_brains.py"),
    ("brains", 'self.budget.charge(self.model, response.usage)', 'pass', "tests/test_brains.py"),
    ("brains", 'if response.stop_reason == "refusal":', 'if False:', "tests/test_brains.py"),
    ("brains", 'if response.stop_reason == "max_tokens":', 'if False:', "tests/test_brains.py"),
    ("brains", 'return schema.model_validate_json(texts[-1])', 'return schema.model_construct()', "tests/test_brains.py"),
    ("brains", '            self.on_charge(self.spent)\n', '            pass\n', "tests/test_brains.py"),
    ("brains", 'go = gain_ok and payload["quota_left"] > 0 and payload["canary_passed"] and payload["file_ok"]', 'go = gain_ok', "tests/test_brains.py"),
    ("orchestrator", 'if not self.config.allow_submit:', 'if False:', "tests/test_orchestrator.py"),
    ("orchestrator", 'if s.get("blend_version") == BLEND_VERSION]', ']', "tests/test_orchestrator.py"),
    ("orchestrator", '                           blend_version=BLEND_VERSION)', '                           )', "tests/test_orchestrator.py"),
    ("orchestrator", 'if set(e["features"]) <= set(FeaturePlan.model_fields) and e["family"] in families]', 'if set(e["features"]) <= set(FeaturePlan.model_fields)]', "tests/test_orchestrator.py"),
    ("orchestrator", 'features = {**FEATURE_DEFAULTS, **experiment["features"]}', 'features = experiment["features"]', "tests/test_orchestrator.py"),
    ("orchestrator", 'since = entries if start is None else entries[start + 1:]', 'since = entries', "tests/test_orchestrator.py"),
    ("orchestrator", 'return notes + lessons[-LESSONS_SHOWN:]', 'return (notes + lessons)[-LESSONS_SHOWN:]', "tests/test_orchestrator.py"),
    ("orchestrator", 'if set(e["features"]) <= set(FeaturePlan.model_fields) and e["family"] in families]', 'if e["family"] in families]', "tests/test_orchestrator.py"),
    ("orchestrator", 'lessons=self.config.notes, source="operator")', 'lessons=self.config.notes)', "tests/test_orchestrator.py"),
    ("orchestrator", 'self.ledger.append("stop", reason="crashed", detail=repr(exc)[:300])', 'pass', "tests/test_orchestrator.py"),
    ("orchestrator", 'budget.on_charge = lambda _: self._record_spend()', 'pass', "tests/test_orchestrator.py"),
    ("orchestrator", 'reason=reason, error=result.get("error"),', 'reason=reason, error=None,', "tests/test_orchestrator.py"),
    ("orchestrator", 'feedback = str(exc)[:500]', 'feedback = None', "tests/test_orchestrator.py"),
    ("orchestrator", 'if not info["entered"]:', 'if False:', "tests/test_orchestrator.py"),
    ("orchestrator", 'request = {**payload, "rejected_because": feedback} if feedback else payload', 'request = payload', "tests/test_orchestrator.py"),
    ("orchestrator", '"canary_passed": bool(clean)', '"canary_passed": True', "tests/test_orchestrator.py"),
    ("orchestrator", '"file_ok": not problems', '"file_ok": True', "tests/test_orchestrator.py"),
    ("orchestrator", 'gap=None if score is None else cv - score', 'gap=None if score is None else 0.0', "tests/test_orchestrator.py"),
    ("orchestrator", 'except (BudgetExceeded, BrainUnavailable, PayloadRejected):', 'except BudgetExceeded:', "tests/test_orchestrator.py"),
    ("orchestrator", 'self.ledger.append("lesson", round=step, diagnosis=report.diagnosis, lessons=report.lessons)', 'pass', "tests/test_orchestrator.py"),
    ("orchestrator", 'if rounds_without_gain >= self.config.no_gain_rounds:', 'if False:', "tests/test_orchestrator.py"),
    ("orchestrator", 'return CONTRACTS[role].model_validate(entry["output"])', 'pass', "tests/test_orchestrator.py"),
    ("report", 'if snapshots:  # the board shows', 'if False:  # the board shows', "tests/test_cli_report.py"),
]


def main():
    selected = sys.argv[1:]
    survivors, broken, timed_out = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "repo"
        copy.mkdir()
        for item in ("slayer", "tests"):
            shutil.copytree(REPO / item, copy / item, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(REPO / "pyproject.toml", copy / "pyproject.toml")

        for index, (module, old, new, test) in enumerate(MUTATIONS):
            if selected and str(index) not in selected:
                continue
            path = copy / "slayer" / f"{module}.py"
            original = path.read_text(encoding="utf-8")
            if original.count(old) != 1:
                broken.append((index, module, old[:60]))
                print(f"[{index}] NOT APPLIED ({original.count(old)} matches): {module} {old[:60]}", flush=True)
                continue
            try:
                path.write_text(original.replace(old, new), encoding="utf-8")
                try:
                    run = subprocess.run([PYTHON, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", test], cwd=copy,
                                         capture_output=True, text=True, timeout=TIMEOUT)
                    outcome = "killed" if run.returncode != 0 else "SURVIVED"
                except subprocess.TimeoutExpired:  # a hang can be the mutant or a busy machine, so it proves nothing
                    outcome = "TIMEOUT"
            finally:
                path.write_text(original, encoding="utf-8")
            print(f"[{index}] {outcome}: {module}: {old[:70]}", flush=True)
            if outcome == "SURVIVED":
                survivors.append((index, module, old))
            elif outcome == "TIMEOUT":
                timed_out.append((index, module, old))
    print(f"\n{len(MUTATIONS) if not selected else len(selected)} mutations, {len(survivors)} survived, "
          f"{len(timed_out)} timed out, {len(broken)} not applied", flush=True)
    for item in survivors + timed_out + broken:
        print(" -", item[0], item[1], item[2][:90], flush=True)


if __name__ == "__main__":
    main()
