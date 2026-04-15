"""JSONL helpers with stable serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def stable_json_dumps(data: Any) -> str:
    """Serialize data to stable ASCII JSON."""

    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """Write iterable rows to JSONL using stable serialization."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json_dumps(row))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} in {path}.")
            rows.append(payload)
    return rows
