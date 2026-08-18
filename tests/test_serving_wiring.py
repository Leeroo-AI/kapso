# Serving-live wiring tests (P5.2): the launch-side staging, the gate env
# threading, the context-slot replacement, and the judge's citation field
# (Rule 9: the regressions are serving silently off when enabled, the gate
# resolving without its campaign env, static notes surviving a live brief,
# and load-bearing citations dropped on the floor).

import subprocess
from pathlib import Path

import pytest
import yaml

from benchmarks.relbench.context import (
    FEATURE_ENGINEERING_NOTE,
    knowledge_section,
)
from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import (
    FeedbackGenerator,
    FeedbackResult,
)
from kapso.gated_mcp.presets import get_mcp_config
from kapso.learning.serving_launch import prepare_campaign_serving
from kapso.learning.update_frame import init_bank
from tests.test_bank_retriever import card_text
from tests.test_update_frame import seed_bank_home

TASK = {"family": "entity_binary_classification", "dataset": "rel-hm"}


def serving_config(tmp_path, enabled=True):
    return {
        "learning": {
            "serving": {"enabled": enabled},
            "bank": {"local_path": str(tmp_path / "bank-home.git"),
                     "remote": None},
            "retriever": {"k_insights": 2, "k_procedures": 1, "k_pitfalls": 1,
                          "unvisited_discount": 0.5},
        }
    }


def test_prepare_campaign_serving_stages_everything(tmp_path):
    seed_bank_home(tmp_path, {"a-card": card_text("a-card")})
    work_dir = tmp_path / "work"
    serving = prepare_campaign_serving(serving_config(tmp_path), TASK, work_dir)
    # the brief is stamped with the pinned head and carries the card
    assert serving["bank_head"] in serving["brief"]
    assert "[card:a-card]" in serving["brief"]
    # the push record landed inside the harvested tree
    record = yaml.safe_load(Path(serving["record_path"]).read_text())
    assert record["bank_head"] == serving["bank_head"]
    assert record["served"][0]["card"] == "a-card"
    # the pull-tool env mapping is complete and points into the work dir
    env = serving["bank_serving"]
    assert Path(env["KAPSO_BANK_DIR"]).is_dir()
    assert env["KAPSO_BANK_HEAD"] == serving["bank_head"]
    assert env["KAPSO_TASK_FAMILY"] == TASK["family"]
    assert env["KAPSO_TASK_DATASET"] == TASK["dataset"]
    assert str(work_dir) in env["KAPSO_SERVING_PULL_LOG"]
    # the checkout is PINNED: a later bank commit must not change it
    head_now = subprocess.run(
        ["git", "-C", env["KAPSO_BANK_DIR"], "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert head_now == serving["bank_head"]


def test_serving_off_returns_none_and_missing_home_raises(tmp_path):
    assert prepare_campaign_serving(
        serving_config(tmp_path, enabled=False), TASK, tmp_path / "w"
    ) is None
    with pytest.raises(FileNotFoundError, match="bank home"):
        prepare_campaign_serving(serving_config(tmp_path), TASK, tmp_path / "w")


def test_bank_serving_env_threads_into_the_gate_server(tmp_path):
    mapping = {
        "KAPSO_BANK_DIR": str(tmp_path),
        "KAPSO_BANK_HEAD": "abc123",
        "KAPSO_SERVING_PULL_LOG": str(tmp_path / "pull.jsonl"),
        "KAPSO_TASK_FAMILY": "entity_binary_classification",
        "KAPSO_TASK_DATASET": "rel-hm",
    }
    mcp_servers, tools = get_mcp_config(
        ["bank"], gate_failure_policy="error", bank_serving=mapping,
    )
    env = mcp_servers["gated-knowledge"]["env"]
    for key, value in mapping.items():
        assert env[key] == value
    assert "bank" in env["MCP_ENABLED_GATES"]
    # without the mapping the gate must not resolve
    mcp_servers, _ = get_mcp_config(
        ["bank"], gate_failure_policy="skip", bank_serving=None,
    )
    assert "gated-knowledge" not in mcp_servers


def test_knowledge_slot_replaced_by_live_brief():
    live = knowledge_section("SERVED BRIEF BODY")
    assert "Knowledge bank brief" in live and "SERVED BRIEF BODY" in live
    assert FEATURE_ENGINEERING_NOTE not in live
    fallback = knowledge_section(None)
    assert FEATURE_ENGINEERING_NOTE in fallback
    assert "Knowledge bank brief" not in fallback


def test_judge_parses_cards_load_bearing():
    generator = FeedbackGenerator.__new__(FeedbackGenerator)
    parsed = generator._parse_response(
        "<stop>false</stop><evaluation_valid>true</evaluation_valid>"
        "<score>0.7</score><feedback>keep going</feedback>"
        "<cards_load_bearing>[card:group-relative-standing], "
        "cross-branch-ensemble-candidate</cards_load_bearing>"
    )
    assert parsed.cards_load_bearing == [
        "group-relative-standing", "cross-branch-ensemble-candidate",
    ]
    assert parsed.to_dict()["cards_load_bearing"] == parsed.cards_load_bearing
    # none / absent both mean no attribution — and never a parse failure
    for tail in ("<cards_load_bearing>none</cards_load_bearing>", ""):
        parsed = generator._parse_response(
            "<stop>false</stop><feedback>f</feedback>" + tail
        )
        assert parsed.cards_load_bearing == []
    assert FeedbackResult(stop=False, evaluation_valid=True,
                          feedback="x").cards_load_bearing == []
