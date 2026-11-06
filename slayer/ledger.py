"""Append-only record of a run. It is the evidence and the state used to resume after a crash."""
import json
from datetime import datetime, timezone
from pathlib import Path


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, kind: str, **fields) -> dict:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        with self.path.open("ab+") as handle:
            if handle.seek(0, 2) > 0:  # a crash may have left a partial line, even mid-character
                handle.seek(-1, 2)
                if handle.read(1) != b"\n":
                    line = b"\n" + line
            handle.write(line)
        return entry

    def read(self, kind: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        # Only "\n" ends a line: agent text can hold other separators, which str.splitlines would cut on.
        for line in self.path.read_bytes().decode("utf-8", errors="replace").split("\n"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or entry.get("kind") == kind:
                entries.append(entry)
        return entries
