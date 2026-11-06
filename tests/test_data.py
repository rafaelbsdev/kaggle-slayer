import json

import numpy as np
import polars as pl
import pytest

from fakes import SENTINEL_CATEGORY, SENTINEL_NUMBER
from slayer.data import build_profile, find_drift_clock, infer_schema, load_competition, temporal_hint


def test_infer_schema_from_file_columns():
    schema = infer_schema(["id", "a", "b", "label"], ["id", "a", "b"], ["id", "prediction"])
    assert (schema.id_col, schema.target_col, schema.pred_col) == ("id", "label", "prediction")
    assert schema.features == ["a", "b"]


@pytest.mark.parametrize("train,test,sample,message", [
    (["id", "a", "y1", "y2"], ["id", "a"], ["id", "p"], "target"),
    (["id", "a"], ["id", "a"], ["id", "p"], "target"),
    (["x", "a", "y"], ["x", "a"], ["id", "p"], "id"),
    (["id", "a", "y"], ["id", "a"], ["id"], "sample"),
])
def test_infer_schema_refuses_ambiguity(train, test, sample, message):
    with pytest.raises(ValueError, match=message):
        infer_schema(train, test, sample)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_competition(tmp_path, tmp_path / "work")


def test_load_produces_aligned_float32_matrices(iid_raw, tmp_path):
    raw, _ = iid_raw
    data = load_competition(raw, tmp_path / "work")
    assert data.X_train.dtype == np.float32
    assert data.X_train.shape == (2400, 9)
    assert data.X_test.shape == (800, 9)
    assert data.schema.features[-1] == "cat"
    assert set(np.unique(data.y)) == {0, 1}
    assert data.train_ids[0] == 1000 and data.test_ids[0] == 3400
    assert np.isnan(data.X_train[:, 4]).any()


def test_categorical_codes_are_shared_between_train_and_test(iid_raw, tmp_path):
    raw, _ = iid_raw
    data = load_competition(raw, tmp_path / "work")
    train_codes = set(np.unique(data.X_train[:, -1]).tolist())
    test_codes = set(np.unique(data.X_test[:, -1]).tolist())
    assert test_codes <= train_codes
    assert "cat" in data.schema.categoricals


def blank_columns(raw, file_name, columns):
    frame = pl.read_csv(raw / file_name)
    frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in columns).write_csv(raw / file_name)


@pytest.mark.parametrize("file_name", ["train.csv", "test.csv"])
def test_column_with_no_values_in_one_file_is_dropped_not_read_as_text(iid_raw, tmp_path, file_name):
    raw, _ = iid_raw
    blank_columns(raw, file_name, ["f2"])
    data = load_competition(raw, tmp_path / "work")
    assert data.schema.empty == ["f2"]
    assert "f2" not in data.schema.features and "f2" not in data.schema.categoricals
    assert data.X_train.shape[1] == data.X_test.shape[1] == 8
    assert build_profile(data)["n_features_dropped_as_empty"] == 1


def test_no_feature_with_values_is_a_clear_error(iid_raw, tmp_path):
    raw, _ = iid_raw
    blank_columns(raw, "test.csv", [f"f{i}" for i in range(8)] + ["cat"])
    with pytest.raises(ValueError, match="no feature"):
        load_competition(raw, tmp_path / "work")


def test_interrupted_parquet_write_leaves_no_file_behind(iid_raw, tmp_path, monkeypatch):
    raw, _ = iid_raw
    work = tmp_path / "work"

    def dies_halfway(self, path, **kwargs):
        with open(path, "wb") as handle:
            handle.write(b"PAR1 cut short")
        raise KeyboardInterrupt

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", dies_halfway)
    with pytest.raises(KeyboardInterrupt):
        load_competition(raw, work)
    assert not (work / "train.parquet").exists()
    monkeypatch.undo()
    assert load_competition(raw, work).X_train.shape == (2400, 9)


def test_parquet_cache_is_reused(iid_raw, tmp_path):
    raw, _ = iid_raw
    work = tmp_path / "work"
    load_competition(raw, work)
    stamp = (work / "train.parquet").stat().st_mtime_ns
    load_competition(raw, work)
    assert (work / "train.parquet").stat().st_mtime_ns == stamp


def test_profile_carries_no_row_values(iid_raw, tmp_path):
    raw, _ = iid_raw
    data = load_competition(raw, tmp_path / "work")
    assert (data.X_train == SENTINEL_NUMBER).any()  # the sentinel really is in the data the profile reads
    profile = build_profile(data, seed=0)
    text = json.dumps(profile)
    assert SENTINEL_CATEGORY not in text
    assert str(SENTINEL_NUMBER) not in text


def test_profile_reports_shape_target_and_no_drift_on_iid_data(iid_raw, tmp_path):
    raw, hidden = iid_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert profile["n_train"] == 2400 and profile["n_features"] == 9
    assert 0.2 < profile["target_rate"] < 0.8
    assert profile["target_is_binary"] is True
    assert profile["adversarial_auc"] < 0.65


def test_profile_sees_drift(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert profile["adversarial_auc"] > 0.75
    assert profile["top_drifting"][0]["name"] == "f5"


def test_profile_measures_the_drift_left_after_dropping_top_features(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    curve = profile["drift_after_drop"]
    assert curve[0]["k"] == 1
    assert curve[0]["adversarial_auc"] < 0.6  # f5 carries all the drift in this data
    assert all(point["k"] <= 0.8 * data.X_train.shape[1] for point in curve)


def test_drift_curve_is_short_when_there_is_no_drift(iid_raw, tmp_path):
    raw, _ = iid_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert len(profile["drift_after_drop"]) == 1


def test_profile_shows_row_order_is_time_when_it_is(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert profile["row_order_drift_auc"] > 0.75
    assert profile["id_order_drift_auc"] > 0.75


def test_profile_shows_row_order_is_not_time_when_rows_are_shuffled(shuffled_temporal_raw, tmp_path):
    raw, _ = shuffled_temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert profile["adversarial_auc"] > 0.75
    assert profile["row_order_drift_auc"] < 0.58
    assert profile["id_order_drift_auc"] < 0.58


def test_drift_clock_recovers_time_from_shuffled_rows(shuffled_temporal_raw, tmp_path):
    raw, _ = shuffled_temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert profile["drift_clock"]["name"] == "f5"
    assert profile["drift_clock"]["share_of_test_beyond_train"] >= 0.5
    assert np.array_equal(data.time_score, data.X_train[:, data.schema.features.index("f5")])


def test_no_drift_clock_without_drift(iid_raw, tmp_path):
    raw, _ = iid_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert profile["drift_clock"] is None
    assert data.time_score is None


def test_drift_clock_reads_a_column_that_runs_downwards():
    rng = np.random.default_rng(0)
    X_train = np.column_stack([rng.normal(size=2000), -np.linspace(0, 10, 2000)]).astype(np.float32)
    X_test = np.column_stack([rng.normal(size=500), -np.linspace(11, 12, 500)]).astype(np.float32)
    column, direction, share = find_drift_clock(X_train, X_test, np.array([1, 0]))
    assert (column, direction) == (1, -1)
    assert share > 0.9


def test_drift_clock_skips_columns_with_many_nulls():
    X_train = np.linspace(0, 10, 2000, dtype=np.float32)[:, None]
    X_train[:600, 0] = np.nan
    X_test = np.linspace(11, 12, 500, dtype=np.float32)[:, None]
    assert find_drift_clock(X_train, X_test, np.array([0])) is None


def test_drift_clock_ignores_columns_whose_test_stays_inside_train():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(2000, 1)).astype(np.float32)
    X_test = rng.normal(size=(500, 1)).astype(np.float32) + 0.5
    assert find_drift_clock(X_train, X_test, np.array([0])) is None


def test_rows_with_a_null_clock_count_as_earliest(shuffled_temporal_raw, tmp_path):
    raw, _ = shuffled_temporal_raw
    data = load_competition(raw, tmp_path / "work")
    column = data.schema.features.index("f5")
    data.X_train[0, column] = np.nan
    build_profile(data, seed=0)
    assert data.time_score[0] == -np.inf


def test_profile_never_exposes_the_clock_values(shuffled_temporal_raw, tmp_path):
    raw, _ = shuffled_temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert all(not isinstance(value, np.ndarray) for value in profile.values())
    assert set(profile["drift_clock"]) == {"name", "share_of_test_beyond_train"}
    assert len(json.dumps(profile)) < 10_000


def test_drift_rank_lists_every_feature(temporal_raw, tmp_path):
    raw, _ = temporal_raw
    data = load_competition(raw, tmp_path / "work")
    profile = build_profile(data, seed=0)
    assert sorted(data.drift_rank) == sorted(data.schema.features)
    assert data.drift_rank[0] == "f5"
    assert profile["top_drifting"][0]["name"] == data.drift_rank[0]


@pytest.mark.parametrize("text,expected", [
    ("All loans in the test set occurred strictly after the training set.", True),
    ("The data has a temporal split.", True),
    ("Use a time-based validation scheme.", True),
    ("Predict credit card fraud on synthetic data.", False),
    ("", False),
])
def test_temporal_hint_reads_the_overview(text, expected):
    assert temporal_hint(text) is expected
