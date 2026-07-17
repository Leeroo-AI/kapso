"""Hermetic tests for the PostTrainBench runner's runtime-config shaping.

The env_strip contract is the credential-containment boundary: on non-judge
tasks solve.sh passes --strip-agent-env OPENAI_API_KEY and the runner must
thread it into BOTH agent sections, so no agent session inherits kapso's own
LLM key; on judge tasks nothing is passed and the key must flow untouched
(evaluate.py needs it). A silent regression here would only surface as a
rule-violation flag on an official run.
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.posttrain.runner import build_runtime_config  # noqa: E402

SESSION_TIMEOUTS = {"ideation_timeout": 300, "implementation_timeout": 1800}


def load_runtime_mode_config(tmp_path, **kwargs):
    path = build_runtime_config(
        "POSTTRAIN", None, str(tmp_path), dict(SESSION_TIMEOUTS), **kwargs
    )
    with open(path) as f:
        return yaml.safe_load(f)["modes"]["POSTTRAIN"]


def test_agent_env_strip_reaches_strategy_params_and_agent_sections(tmp_path):
    mode_cfg = load_runtime_mode_config(
        tmp_path, agent_env_strip=["OPENAI_API_KEY"]
    )
    assert mode_cfg["search_strategy"]["params"]["env_strip"] == [
        "OPENAI_API_KEY"
    ]
    for section in ("coding_agent", "feedback_generator"):
        assert mode_cfg[section]["agent_specific"]["env_strip"] == [
            "OPENAI_API_KEY"
        ]


def test_judge_tasks_leave_agent_env_untouched(tmp_path):
    mode_cfg = load_runtime_mode_config(tmp_path)
    assert "env_strip" not in mode_cfg["search_strategy"]["params"]
    for section in ("coding_agent", "feedback_generator"):
        assert "env_strip" not in mode_cfg[section]["agent_specific"]


def test_iteration_admission_floor_survives_runtime_config(tmp_path):
    # R10-P2-2: the core reserve gate reads budget.min_iteration_seconds;
    # losing this key in the runtime-config round-trip silently reverts to
    # the 60s default that admitted a doomed iteration at 96.6% budget.
    mode_cfg = load_runtime_mode_config(tmp_path)
    assert mode_cfg["budget"]["min_iteration_seconds"] == 1800
