import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import make_competition


@pytest.fixture(autouse=True)
def no_real_kaggle_login(tmp_path, monkeypatch):
    """No test may read or renew the login of whoever runs the suite."""
    monkeypatch.setattr("slayer.kaggle_client.CREDENTIALS_FILE", tmp_path / "kaggle-login.json")
    return tmp_path / "kaggle-login.json"


@pytest.fixture
def iid_raw(tmp_path):
    raw = tmp_path / "raw"
    hidden = make_competition(raw, temporal=False)
    return raw, hidden


@pytest.fixture
def temporal_raw(tmp_path):
    raw = tmp_path / "raw"
    hidden = make_competition(raw, temporal=True)
    return raw, hidden


@pytest.fixture
def shuffled_temporal_raw(tmp_path):
    """Test comes after train in time, but train's rows and ids say nothing about when: like CloudWalk's data."""
    raw = tmp_path / "raw"
    hidden = make_competition(raw, temporal=True, shuffle_train=True)
    return raw, hidden
