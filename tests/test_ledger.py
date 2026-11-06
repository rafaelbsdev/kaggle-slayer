import pytest

from slayer.ledger import Ledger


def test_append_then_read_back(tmp_path):
    ledger = Ledger(tmp_path / "runs" / "ledger.jsonl")
    ledger.append("experiment", id="abc", cv=0.81)
    ledger.append("submission", ref="1")
    entries = ledger.read()
    assert [e["kind"] for e in entries] == ["experiment", "submission"]
    assert entries[0]["cv"] == 0.81
    assert "ts" in entries[0]


def test_read_filters_by_kind(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("experiment", id="abc")
    ledger.append("submission", ref="1")
    assert [e["ref"] for e in ledger.read("submission")] == ["1"]


def test_read_on_missing_file_is_empty(tmp_path):
    assert Ledger(tmp_path / "none.jsonl").read() == []


def test_partial_last_line_from_a_crash_is_ignored(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.append("experiment", id="ok")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "experiment", "id": "cut')
    assert [e["id"] for e in ledger.read()] == ["ok"]


@pytest.mark.parametrize("separator", [" ", " ", "\u0085", "\x0b", "\x0c"])
def test_unusual_line_separators_in_agent_text_do_not_destroy_an_entry(tmp_path, separator):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append("lesson", text=f"first{separator}second")
    ledger.append("lesson", text="after")
    assert [e["text"] for e in ledger.read()] == [f"first{separator}second", "after"]


def test_append_survives_a_tail_cut_inside_a_multibyte_character(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes('{"kind": "lesson", "text": "cora'.encode() + "ç".encode()[:1])
    ledger = Ledger(path)
    ledger.append("experiment", id="after")
    assert [e["id"] for e in ledger.read()] == ["after"]


def test_append_survives_a_partial_last_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write('{"kind": "experiment", "id": "cut')
    ledger.append("experiment", id="after")
    assert [e["id"] for e in ledger.read()] == ["after"]
