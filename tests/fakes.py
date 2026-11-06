"""Offline stand-ins: a synthetic competition and a Kaggle that scores for real on hidden labels."""
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

SENTINEL_CATEGORY = "SENTINEL_CAT_VALUE"
SENTINEL_NUMBER = 987654.5  # exactly representable in float32, so it survives loading unchanged


def make_competition(dest: Path, temporal: bool = False, seed: int = 0, n_train: int = 2400, n_test: int = 800,
                     shuffle_train: bool = False):
    """Writes train/test/sample CSVs and returns the hidden test labels."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x = rng.normal(size=(n, 8)).astype(float)
    drift = np.linspace(0, 12, n) if temporal else np.zeros(n)  # big enough to put test beyond train's range
    x[:, 5] += drift
    score = 1.3 * x[:, 0] - 1.1 * x[:, 1] + 0.9 * x[:, 2] * x[:, 3]
    category = rng.choice(["a", "b", "c"], size=n)
    score += np.where(category == "a", 0.8, 0.0)
    y = (score + rng.normal(scale=1.0, size=n) > 0).astype(int)
    x[rng.random(n) < 0.05, 4] = np.nan
    x[7, 6] = SENTINEL_NUMBER
    category[11] = SENTINEL_CATEGORY

    columns = {f"f{i}": x[:, i] for i in range(8)}
    frame = pl.DataFrame({"id": np.arange(1000, 1000 + n), **columns, "cat": category})
    train = frame[:n_train].with_columns(pl.Series("label", y[:n_train]))
    test = frame[n_train:]
    if shuffle_train:
        ids = train["id"].to_numpy()
        train = train[rng.permutation(n_train).tolist()].with_columns(pl.Series("id", ids))

    dest.mkdir(parents=True, exist_ok=True)
    train.write_csv(dest / "train.csv")
    test.write_csv(dest / "test.csv")
    test.select("id").with_columns(pl.lit(0.5).alias("prediction")).write_csv(dest / "sample_submission.csv")
    return y[n_train:]


class FakeKaggle:
    def __init__(self, raw_dir: Path, hidden, overview="A tabular binary classification scored by AUC.",
                 daily_limit=5, entered=True):
        self.raw_dir, self.hidden = Path(raw_dir), np.asarray(hidden)
        self.overview_text, self.daily_limit, self.entered = overview, daily_limit, entered
        self.submitted: list[dict] = []
        self.board = np.linspace(0.55, 0.97, 100)

    def info(self, slug):
        scores = [s["public_score"] for s in self.submitted]
        rank = None
        if scores:
            rank = int((self.board > max(scores)).sum()) + 1
        return {"entered": self.entered, "team_count": 101, "user_rank": rank, "category": "playground"}

    def overview(self, slug):
        return self.overview_text

    def download(self, slug, dest):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copy(self.raw_dir / name, dest / name)
        return dest

    def quota_left(self, slug):
        return self.daily_limit - len(self.submitted)

    def submit(self, slug, path, message):
        frame = pl.read_csv(path).sort("id")
        public = len(self.hidden) // 3
        score = float(roc_auc_score(self.hidden[:public], frame["prediction"].to_numpy()[:public]))
        self.submitted.append({"ref": str(len(self.submitted) + 1), "public_score": score, "message": message})
        return dict(self.submitted[-1])

    def submissions(self, slug):
        return [dict(s) for s in self.submitted]

    def leaderboard(self, slug, dest):
        ordered = sorted(self.board, reverse=True)
        return [{"rank": i + 1, "team": f"team{i}", "score": float(s)} for i, s in enumerate(ordered)]
