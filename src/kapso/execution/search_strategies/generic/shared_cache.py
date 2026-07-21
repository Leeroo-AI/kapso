"""Campaign shared cache + artifact registry for the generic strategy.

The shared cache is a campaign-persistent directory (surviving experiments and
resumes) where sessions store expensive reusable artifacts — precomputed
tables, embeddings, feature matrices, per-model predictions. Sessions discover
it via the ``KAPSO_SHARED_CACHE_DIR`` env var (injected set-if-absent into
every session env) and describe what they stored in ``artifacts.json`` so
later experiments — and later campaigns, when the cache is pointed at a
persistent path — get an OPTIONAL offer of what already exists. The offer is
information, not instruction: agents are told to verify before trusting, and
to ignore or rebuild when unhelpful.

Registry format (``artifacts.json`` in the cache root): a JSON list of
entries, each with required keys ``name``/``path``/``description`` and
optional ``producer``/``content_key``/``rebuild_hint``. ``path`` is relative
to the cache root. A missing registry means "nothing registered" (documented
default); a corrupt or malformed one raises — fail loud, never skip.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SHARED_CACHE_ENV_VAR = "KAPSO_SHARED_CACHE_DIR"
REGISTRY_FILENAME = "artifacts.json"

_REQUIRED_KEYS = ("name", "path", "description")
_OPTIONAL_KEYS = ("producer", "content_key", "rebuild_hint")

NO_ARTIFACTS_BRIEF = "No shared-cache artifacts registered yet."

_OFFER_PREAMBLE = (
    "Artifacts below were built by previous experiments/campaigns and live in "
    "$KAPSO_SHARED_CACHE_DIR. They are an OFFER, not an instruction: verify "
    "before trusting (check the content key against current task data; "
    "spot-check correctness against the live task), use them if they help, "
    "ignore or rebuild them if not."
)


def resolve_shared_cache_dir(workspace_dir: str, override: Optional[str]) -> Path:
    """Resolve (and create) the campaign shared-cache directory.

    Default: ``<workspace>/.kapso/shared_cache``. An absolute ``override``
    (config: ``search_strategy.params.shared_cache_dir``) points it at a
    persistent task-level path to carry artifacts across campaigns.
    """
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("shared_cache_dir override must be a non-empty string")
        cache_dir = Path(override).expanduser().absolute()
    else:
        cache_dir = Path(workspace_dir).absolute() / ".kapso" / "shared_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_artifact_registry(cache_dir: Path) -> List[Dict[str, Any]]:
    """Load and validate ``artifacts.json``; missing file -> empty list."""
    registry_path = cache_dir / REGISTRY_FILENAME
    if not registry_path.is_file():
        return []
    entries = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(
            f"{registry_path} must contain a JSON list, got "
            f"{type(entries).__name__}"
        )
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{registry_path}[{i}] must be an object")
        missing = [k for k in _REQUIRED_KEYS if not isinstance(entry.get(k), str) or not entry[k].strip()]
        if missing:
            raise ValueError(
                f"{registry_path}[{i}] missing required keys: {', '.join(missing)}"
            )
        unknown = sorted(set(entry) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
        if unknown:
            raise ValueError(
                f"{registry_path}[{i}] has unknown keys: {', '.join(unknown)}"
            )
    return entries


def render_artifacts_brief(cache_dir: Path, entries: List[Dict[str, Any]]) -> str:
    """Render the optional-offer brief injected into ideation/implementation."""
    if not entries:
        return NO_ARTIFACTS_BRIEF
    lines = [_OFFER_PREAMBLE, ""]
    for entry in entries:
        artifact_path = cache_dir / entry["path"]
        if artifact_path.exists():
            size = artifact_path.stat().st_size if artifact_path.is_file() else sum(
                f.stat().st_size for f in artifact_path.rglob("*") if f.is_file()
            )
            status = f"present, {size / 1_048_576:.1f} MiB"
        else:
            status = "recorded but MISSING on disk"
        line = (
            f"- **{entry['name']}** (`$KAPSO_SHARED_CACHE_DIR/{entry['path']}`, "
            f"{status}): {entry['description']}"
        )
        for key, label in (
            ("producer", "producer"),
            ("content_key", "content key"),
            ("rebuild_hint", "rebuild"),
        ):
            if entry.get(key):
                line += f" [{label}: {entry[key]}]"
        lines.append(line)
    return "\n".join(lines)


def build_shared_artifacts_brief(workspace_dir: str, override: Optional[str]) -> tuple:
    """One-call helper for the strategy: (cache_dir, rendered brief)."""
    cache_dir = resolve_shared_cache_dir(workspace_dir, override)
    return cache_dir, render_artifacts_brief(cache_dir, load_artifact_registry(cache_dir))
