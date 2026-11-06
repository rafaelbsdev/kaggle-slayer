import numpy as np
import polars as pl
import pytest

from slayer.submission import validate_submission, write_submission


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample_submission.csv"
    pl.DataFrame({"id": [10, 11, 12], "prediction": [0.5, 0.5, 0.5]}).write_csv(path)
    return path


def write(tmp_path, ids, preds, id_col="id", pred_col="prediction"):
    path = tmp_path / "out.csv"
    pl.DataFrame({id_col: ids, pred_col: preds}).write_csv(path)
    return path


def test_write_follows_sample_order_and_validates(tmp_path, sample):
    out = tmp_path / "sub" / "candidate.csv"
    write_submission(sample, np.array([12, 10, 11]), np.array([0.3, 0.1, 0.2]), out)
    assert pl.read_csv(out)["id"].to_list() == [10, 11, 12]
    assert pl.read_csv(out)["prediction"].to_list() == pytest.approx([0.1, 0.2, 0.3])
    assert validate_submission(out, sample) == []


def test_write_refuses_length_mismatch(tmp_path, sample):
    with pytest.raises(ValueError, match="predictions"):
        write_submission(sample, np.array([10, 11, 12]), np.array([0.1, 0.2]), tmp_path / "x.csv")


@pytest.mark.parametrize("ids,preds,problem", [
    ([10, 11], [0.1, 0.2], "rows"),
    ([10, 11, 99], [0.1, 0.2, 0.3], "ids"),
    ([10, 11, 11], [0.1, 0.2, 0.3], "ids"),
    ([10, 11, 12], [0.1, None, 0.3], "null"),
    ([10, 11, 12], [0.1, float("nan"), 0.3], "null"),
    ([10, 11, 12], [0.1, 1.5, 0.3], "range"),
    ([10, 11, 12], [-0.1, 0.5, 0.3], "range"),
    ([10, 11, 12], [0.5, 0.5, 0.5], "constant"),
])
def test_validation_catches_each_problem(tmp_path, sample, ids, preds, problem):
    problems = validate_submission(write(tmp_path, ids, preds), sample)
    assert any(problem in p for p in problems), problems


def test_validation_catches_wrong_columns(tmp_path, sample):
    problems = validate_submission(write(tmp_path, [10, 11, 12], [0.1, 0.2, 0.3], pred_col="score"), sample)
    assert any("columns" in p for p in problems)
