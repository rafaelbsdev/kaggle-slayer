"""Builds submission files and checks them independently before they spend a daily submission."""
from pathlib import Path

import numpy as np
import polars as pl


def write_submission(sample_path, test_ids, predictions, out_path) -> Path:
    if len(predictions) != len(test_ids):
        raise ValueError(f"{len(predictions)} predictions for {len(test_ids)} ids")
    sample = pl.read_csv(sample_path)
    id_col, pred_col = sample.columns
    frame = pl.DataFrame({id_col: np.asarray(test_ids), pred_col: np.asarray(predictions, dtype=float)})
    ordered = sample.select(id_col).join(frame, on=id_col, how="left", maintain_order="left")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered.write_csv(out_path)
    return out_path


def validate_submission(path, sample_path) -> list[str]:
    """Reads the file back from disk, the way Kaggle will. An empty list means it is valid."""
    sample = pl.read_csv(sample_path)
    submission = pl.read_csv(path)
    if submission.columns != sample.columns:
        return [f"columns {submission.columns}, expected {sample.columns}"]

    id_col, pred_col = sample.columns
    problems = []
    if submission.height != sample.height:
        problems.append(f"{submission.height} rows, expected {sample.height}")
    if submission[id_col].n_unique() != submission.height or set(submission[id_col]) != set(sample[id_col]):
        problems.append("ids differ from the sample submission")

    values = submission[pred_col].cast(pl.Float64, strict=False).to_numpy()
    if np.isnan(values).any():
        problems.append("null or NaN predictions")
    finite = values[~np.isnan(values)]
    if finite.size and (finite.min() < 0 or finite.max() > 1):
        problems.append("predictions out of the [0, 1] range")
    if finite.size and finite.std() == 0:
        problems.append("constant predictions")
    return problems
