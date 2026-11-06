"""Last gate before anything reaches an LLM provider (see decision 0002).

The profile builder never collects row values. This guard is the second line, so that a future
change that adds them to a payload fails loudly instead of leaking.
"""
import json

FORBIDDEN_KEYS = {
    "min", "max", "median", "q05", "q25", "q50", "q75", "q95",
    "rows", "sample", "samples", "head", "tail", "values", "examples",
}
MAX_NUMERIC_LIST = 200
MAX_STRING = 8_000
MAX_PAYLOAD_CHARS = 100_000


class PayloadRejected(ValueError):
    pass


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                raise PayloadRejected(f"key {key!r} can carry row values")
            _walk(value)
    elif isinstance(node, (list, tuple)):
        if len(node) > MAX_NUMERIC_LIST and all(isinstance(v, (int, float)) for v in node):
            raise PayloadRejected(f"numeric list with {len(node)} items looks like a data column")
        for value in node:
            _walk(value)
    elif isinstance(node, str) and len(node) > MAX_STRING:
        raise PayloadRejected(f"long string with {len(node)} chars")


def assert_no_rows(payload: dict) -> None:
    _walk(payload)
    size = len(json.dumps(payload, default=str))
    if size > MAX_PAYLOAD_CHARS:
        raise PayloadRejected(f"payload too large: {size} chars")
