import pytest

from slayer.privacy import PayloadRejected, assert_no_rows


def test_aggregates_pass():
    assert_no_rows({"n_rows": 100, "columns": [{"name": "f1", "null_rate": 0.1, "drift": 0.02}]})


@pytest.mark.parametrize("key", ["min", "max", "rows", "sample", "head", "values", "q05"])
def test_keys_that_carry_row_values_are_rejected(key):
    with pytest.raises(PayloadRejected, match=key):
        assert_no_rows({"columns": [{"name": "f1", key: 1.5}]})


def test_long_numeric_list_is_rejected():
    with pytest.raises(PayloadRejected, match="numeric list"):
        assert_no_rows({"something": list(range(500))})


def test_long_string_is_rejected():
    with pytest.raises(PayloadRejected, match="long string"):
        assert_no_rows({"note": "x" * 20000})


def test_oversized_payload_is_rejected():
    columns = [{"name": f"feature_{i}", "null_rate": 0.1} for i in range(5000)]
    with pytest.raises(PayloadRejected, match="too large"):
        assert_no_rows({"columns": columns})
