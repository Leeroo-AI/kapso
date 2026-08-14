# Campaign I/O

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any


def locked_append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def register_artifact(cache_dir: Path, entry: dict[str, Any]) -> None:
    path = cache_dir / "artifacts.json"
    lock_path = cache_dir / "artifacts.json.lock"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(path.read_text()) if path.exists() else []
        except json.JSONDecodeError:
            entries = []
        if not any(item.get("name") == entry.get("name") for item in entries):
            entries.append(entry)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(entries, indent=2) + "\n")
            os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
