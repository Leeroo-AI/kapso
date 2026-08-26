"""Two-layer config resolution: platform `defaults:` under every mode.

load_mode_config() resolves deep_merge(platform defaults, modes[mode]) so
benchmark configs carry overrides only. These tests pin the merge semantics
(nested dicts merge; scalars AND lists replace wholesale), the fail-loud
paths (unknown mode, missing defaults section), and the real-file contract:
the enumerated gap-fills (retry.request_timeout_seconds, models.embedding)
reach every mode that does not override them — the drift class where
request_timeout_seconds reached relbench's copy but never ioai2026's.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from kapso.core import config as config_module
from kapso.core.config import deep_merge, load_mode_config


# ---------------------------------------------------------------------------
# deep_merge semantics
# ---------------------------------------------------------------------------

def test_deep_merge_nested_dicts_merge_key_by_key():
    base = {"retry": {"max_attempts": 2, "request_timeout_seconds": 600}}
    override = {"retry": {"max_attempts": 5}}
    merged = deep_merge(base, override)
    assert merged["retry"] == {"max_attempts": 5, "request_timeout_seconds": 600}


def test_deep_merge_scalar_replaces_and_new_keys_pass_through():
    base = {"models": {"embedding": "text-embedding-3-small"}}
    override = {"models": {"utility": "gpt-5.6-luna"}, "budget": 7}
    merged = deep_merge(base, override)
    assert merged["models"] == {
        "embedding": "text-embedding-3-small",
        "utility": "gpt-5.6-luna",
    }
    assert merged["budget"] == 7


def test_deep_merge_lists_replace_wholesale_not_concat():
    base = {"params": {"gates": ["research", "repo_memory"]}}
    override = {"params": {"gates": ["repo_memory"]}}
    assert deep_merge(base, override)["params"]["gates"] == ["repo_memory"]


def test_deep_merge_dict_override_replaces_scalar_base():
    base = {"models": {"utility": "gpt-4.1-mini"}}
    override = {"models": {"utility": {"model": "gpt-5.6-luna"}}}
    assert deep_merge(base, override)["models"]["utility"] == {
        "model": "gpt-5.6-luna"
    }


def test_deep_merge_mutates_neither_input():
    base = {"retry": {"max_attempts": 2}}
    override = {"retry": {"jitter": True}}
    deep_merge(base, override)
    assert base == {"retry": {"max_attempts": 2}}
    assert override == {"retry": {"jitter": True}}


# ---------------------------------------------------------------------------
# load_mode_config resolution against the real platform defaults
# ---------------------------------------------------------------------------

def _write_config(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


def test_unknown_mode_raises(tmp_path):
    path = _write_config(tmp_path, {"modes": {"ONLY": {}}})
    with pytest.raises(ValueError, match="Unknown mode 'MISSING'"):
        load_mode_config(path, "MISSING")


def test_missing_platform_defaults_section_raises(tmp_path, monkeypatch):
    defaults_less_platform = tmp_path / "platform.yaml"
    defaults_less_platform.write_text(yaml.safe_dump({"modes": {}}))
    monkeypatch.setattr(
        config_module, "PLATFORM_CONFIG_PATH", defaults_less_platform
    )
    path = _write_config(tmp_path, {"modes": {"ANY": {}}})
    with pytest.raises(ValueError, match="no 'defaults' section"):
        load_mode_config(path, "ANY")


def test_mode_without_retry_inherits_request_timeout(tmp_path):
    # The drift-fix itself: a benchmark mode with no retry block at all
    # inherits the platform's request timeout instead of hanging forever
    # on a wedged connection.
    path = _write_config(tmp_path, {"modes": {"BARE": {"budget": {}}}})
    resolved = load_mode_config(path, "BARE")
    assert resolved["retry"]["request_timeout_seconds"] == 600
    assert resolved["models"]["embedding"] == "text-embedding-3-small"


def test_benchmark_override_beats_platform_default(tmp_path):
    path = _write_config(
        tmp_path,
        {"modes": {"TUNED": {"retry": {"request_timeout_seconds": 42}}}},
    )
    resolved = load_mode_config(path, "TUNED")
    assert resolved["retry"]["request_timeout_seconds"] == 42


def test_none_config_path_returns_empty():
    assert load_mode_config(None) == {}


# ---------------------------------------------------------------------------
# Real shipped files: the layering resolves the actual benchmark modes
# ---------------------------------------------------------------------------

def test_relbench_generic_resolves_overrides_and_inherited_defaults():
    resolved = load_mode_config(
        str(REPO_ROOT / "benchmarks" / "relbench" / "config.yaml"),
        "RELBENCH_GENERIC",
    )
    params = resolved["search_strategy"]["params"]
    assert params["implementation_web"] is False
    assert params["ensemble_time_split"] == {
        "member_fraction": 0.7,
        "selector_fraction": 0.3,
    }
    # Mode's own retry pacing survives; the timeout arrives by inheritance.
    assert resolved["retry"]["max_attempts"] == 5
    assert resolved["retry"]["request_timeout_seconds"] == 600


def test_platform_default_modes_carry_no_session_timeouts():
    # Decision from the example E2E run: GENERIC and MINIMAL set no session
    # deadlines — sessions run to completion, bounded only by an explicit
    # time budget. Absent keys resolve to the strategy's None default.
    for mode in ("GENERIC", "MINIMAL"):
        params = load_mode_config(
            str(REPO_ROOT / "src" / "kapso" / "config.yaml"), mode
        )["search_strategy"]["params"]
        assert "ideation_timeout" not in params
        assert "implementation_timeout" not in params


def test_kaggle_resolves_overrides_and_inherited_defaults():
    resolved = load_mode_config(
        str(REPO_ROOT / "benchmarks" / "ioai2026" / "config.yaml"), "KAGGLE"
    )
    assert resolved["search_strategy"]["params"]["implementation_web"] is True
    assert resolved["retry"]["request_timeout_seconds"] == 600
    assert resolved["models"]["embedding"] == "text-embedding-3-small"


def test_every_shipped_mode_constructs_a_model_router():
    # Regression (stale-code audit 2026-08-26): after the CLI-only
    # conversion made ModelRouter embedding-only, every benchmark config
    # still carried utility/reasoning/web_search routes — a ValueError at
    # orchestrator construction that this suite missed because it only
    # asserted on merged dict contents, never the live construction path.
    from kapso.core.config import load_config
    from kapso.core.llm import ModelRouter

    config_files = [
        REPO_ROOT / "src" / "kapso" / "config.yaml",
        *sorted(REPO_ROOT.glob("benchmarks/**/config.yaml")),
        *sorted(REPO_ROOT.glob("examples/**/config.e2e.yaml")),
    ]
    assert len(config_files) >= 6  # packaged + the benchmark/example set
    for config_file in config_files:
        for mode in load_config(str(config_file)).get("modes") or {}:
            resolved = load_mode_config(str(config_file), mode)
            ModelRouter(resolved.get("models"))  # raises on retired roles
