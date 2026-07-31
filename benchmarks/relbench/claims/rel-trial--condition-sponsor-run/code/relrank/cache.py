import fcntl
import hashlib
import json
import os
from pathlib import Path


SCHEMA_VERSION = "lane0_retrieve_rank_v3"


def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / SCHEMA_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def content_hash(parts) -> str:
    value = "|".join(map(str, parts)).encode()
    return hashlib.sha256(value).hexdigest()[:16]


def register_artifact(name: str, path: Path, description: str, content_key: str, rebuild_hint: str) -> None:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    shared.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if registry.exists():
                try:
                    records = json.loads(registry.read_text())
                except json.JSONDecodeError:
                    records = []
            else:
                records = []
            relative = str(path.resolve().relative_to(shared.resolve()))
            record = {
                "name": name,
                "path": relative,
                "description": description,
                "content_key": content_key,
                "rebuild_hint": rebuild_hint,
            }
            if not any(x.get("path") == relative and x.get("content_key") == content_key for x in records):
                records.append(record)
                temporary = registry.with_suffix(".tmp")
                temporary.write_text(json.dumps(records, indent=2))
                os.replace(temporary, registry)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
