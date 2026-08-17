# Imports

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any


# Persistent I/O

def locked_append(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def register_artifact(cache: Path, record: dict[str, Any]) -> None:
    path = cache / "artifacts.json"
    lock_path = cache / "artifacts.json.lock"
    cache.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(path.read_text()) if path.exists() else []
        except json.JSONDecodeError:
            records = []
        if not any(item.get("content_key") == record.get("content_key") for item in records):
            records.append(record)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(records, indent=2) + "\n")
            os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
