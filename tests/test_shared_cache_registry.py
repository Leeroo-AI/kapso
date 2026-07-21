"""Contract tests for the campaign shared cache + artifact registry.

What must hold: the cache resolves to a campaign-persistent default (or the
config override), the env contract reaches session env_defaults set-if-absent,
a missing registry is the documented empty default while a corrupt one raises,
and the optional-offer brief faithfully reports presence/absence on disk. Both
prompt templates must carry the offer slot so the information reaches agents.
"""

import json

import pytest

from kapso.core.prompt_loader import load_prompt
from kapso.execution.search_strategies.generic.shared_cache import (
    NO_ARTIFACTS_BRIEF,
    REGISTRY_FILENAME,
    SHARED_CACHE_ENV_VAR,
    build_shared_artifacts_brief,
    load_artifact_registry,
    render_artifacts_brief,
    resolve_shared_cache_dir,
)


def entry(**overrides):
    base = {
        "name": "oracle-answer-table",
        "path": "table/answers.npz",
        "description": "1472x559 bits + margins",
    }
    base.update(overrides)
    return base


def test_default_resolution_and_override(tmp_path):
    default = resolve_shared_cache_dir(str(tmp_path / "ws"), None)
    assert default == tmp_path / "ws" / ".kapso" / "shared_cache"
    assert default.is_dir()
    override = resolve_shared_cache_dir(str(tmp_path / "ws"), str(tmp_path / "task_cache"))
    assert override == tmp_path / "task_cache"
    assert override.is_dir()
    with pytest.raises(ValueError, match="non-empty"):
        resolve_shared_cache_dir(str(tmp_path / "ws"), "  ")


def test_missing_registry_is_empty_and_corrupt_raises(tmp_path):
    cache = resolve_shared_cache_dir(str(tmp_path), None)
    assert load_artifact_registry(cache) == []
    (cache / REGISTRY_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_artifact_registry(cache)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "not-a-list"},
        ["not-an-object"],
        [{"name": "x", "path": "p"}],  # missing description
        [dict(name="x", path="p", description="d", surprise="y")],  # unknown key
    ],
)
def test_malformed_registry_raises(tmp_path, bad):
    cache = resolve_shared_cache_dir(str(tmp_path), None)
    (cache / REGISTRY_FILENAME).write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_artifact_registry(cache)


def test_brief_reports_presence_and_absence(tmp_path):
    cache = resolve_shared_cache_dir(str(tmp_path), None)
    assert render_artifacts_brief(cache, []) == NO_ARTIFACTS_BRIEF

    present = cache / "table"
    present.mkdir()
    (present / "answers.npz").write_bytes(b"x" * 2048)
    entries = [
        entry(),
        entry(name="ghost", path="gone.npz", description="vanished",
              rebuild_hint="python build.py"),
    ]
    brief = render_artifacts_brief(cache, entries)
    assert "oracle-answer-table" in brief and "present" in brief
    assert "ghost" in brief and "MISSING on disk" in brief
    assert "rebuild: python build.py" in brief
    assert "OFFER, not an instruction" in brief


def test_one_call_helper_round_trip(tmp_path):
    cache = resolve_shared_cache_dir(str(tmp_path), None)
    (cache / "answers.npz").write_bytes(b"x")
    (cache / REGISTRY_FILENAME).write_text(
        json.dumps([entry(path="answers.npz")]), encoding="utf-8"
    )
    cache_dir, brief = build_shared_artifacts_brief(str(tmp_path), str(cache))
    assert cache_dir == cache
    assert "oracle-answer-table" in brief


def test_templates_carry_the_offer_and_contract():
    implementation = load_prompt(
        "execution/search_strategies/generic/prompts/implementation_claude_code.md"
    )
    ideation = load_prompt(
        "execution/search_strategies/generic/prompts/ideation_claude_code.md"
    )
    for template in (implementation, ideation):
        assert "{{shared_artifacts_brief}}" in template
    assert SHARED_CACHE_ENV_VAR in implementation
    assert "artifacts.json" in implementation  # register-what-you-store contract
    assert "Check-before-compute" in implementation
