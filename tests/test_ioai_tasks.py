"""Contract tests for the generic local IOAI-task benchmark (learning harvest).

Hermetic-safe: the handler prompt contract, the runner's metric parse +
fail-loud, and the config's ensemble/selector wiring for K=2. The torch-
dependent evaluator/prepare are validated end-to-end out of band (they need
the task ML stack); a torch-gated check pins the evaluator's fail-loud shape
guard when torch is present.
"""

import os
import time

import pytest
import yaml

from benchmarks.ioai_tasks.handler import LocalTaskHandler
from benchmarks.ioai_tasks.runner import parse_metric

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "ioai_tasks", "config.yaml"
)

SESSION_CAPS = {"ideation_timeout": 1080, "implementation_timeout": 5400}
EVAL_SPEC = {"self_check_command": "python3 dataset/evaluate.py ...",
             "metric_name": "Macro-F1"}


def make_handler(tmp_path, **overrides):
    kwargs = dict(
        task_dir=str(tmp_path / "task"),
        statement="THE TASK STATEMENT",
        deadline_ts=time.time() + 7200,
        session_caps=SESSION_CAPS,
        eval_spec=EVAL_SPEC,
    )
    kwargs.update(overrides)
    return LocalTaskHandler(**kwargs)


def test_handler_context_is_statement_plus_minimal_contract(tmp_path):
    context = make_handler(tmp_path).get_problem_context()
    assert context.startswith("THE TASK STATEMENT")
    assert "submission/solution.py" in context or "solution.py" in context
    assert EVAL_SPEC["self_check_command"] in context
    assert "Macro-F1" in context
    assert "HELD-OUT" in context
    assert "<score>" in context
    # The validation-discipline steer (the recurring cross-task lesson) is present.
    assert "distribution shift" in context and "over-report" in context


def test_handler_rejects_bad_specs(tmp_path):
    with pytest.raises(ValueError, match="session_caps"):
        make_handler(tmp_path, session_caps={})
    with pytest.raises(ValueError, match="eval_spec"):
        make_handler(tmp_path, eval_spec={"metric_name": "x"})


def test_parse_metric_reads_the_evaluator_line():
    assert parse_metric("Macro-F1: 0.5613\nSamples: 700", "Macro-F1") == 0.5613
    with pytest.raises(ValueError, match="Macro-F1"):
        parse_metric("no metric here", "Macro-F1")


def test_config_is_web_off_but_same_models_as_kaggle():
    with open(CONFIG_PATH) as f:
        mode = yaml.safe_load(f)["modes"]["LOCAL"]
    params = mode["search_strategy"]["params"]
    # Same models as the Kaggle runs (codex+fable ideation + Fable lens
    # planner), only web_search muted → leakage-safe on past contests.
    assert params["web_search"] is False
    ens = params["ideation_ensemble"]
    assert [m["cli"] for m in ens] == ["codex", "claude_code"]
    assert params["ideation_lens_planner"]["cli"] == "claude_code"
    assert params["ideation_selector"]["cli"] == "claude_code"
    assert mode["budget"] == {"min_iteration_seconds": 900}
    assert mode["session_budget"]["ideation_fraction"] == 0.2


def test_codex_ideation_search_flag_gated_by_web_search():
    import inspect

    from kapso.execution.search_strategies.generic import codex_ideation

    src = inspect.getsource(codex_ideation.run_codex_ideation)
    # --search is now conditional on the web_search parameter, not hardcoded.
    assert "web_search: bool = True" in src
    assert 'if web_search:' in src and '"--search"' in src


def test_claude_adapter_bans_websearch_via_disallowed_tools():
    # Under --dangerously-skip-permissions, --allowedTools does NOT restrict;
    # web-off must go through --disallowedTools. The adapter reads
    # disallowed_tools and merges it into the banned set.
    from types import SimpleNamespace

    from kapso.execution.coding_agents.adapters.claude_code_agent import (
        ClaudeCodeCodingAgent,
    )

    cfg = SimpleNamespace(agent_specific={
        "auth_mode": "oauth",
        "allowed_tools": ["Read"],
        "disallowed_tools": ["WebSearch", "WebFetch"],
    })
    agent = ClaudeCodeCodingAgent(cfg)
    assert agent._disallowed_tools == ["WebSearch", "WebFetch"]
    assert "WebSearch" in (agent.PRINT_MODE_DEAD_TOOLS + agent._disallowed_tools)


def test_strategy_web_off_sets_disallowed_websearch():
    import inspect

    from kapso.execution.search_strategies.generic import lens_planning, strategy

    src = inspect.getsource(strategy)
    # web-off computes the WebSearch/WebFetch disallow set...
    assert 'self._web_disallowed_tools = (' in src
    assert '["WebSearch", "WebFetch"]' in src
    # ...and threads it into ideation Claude sessions (member/single/lens).
    wired = (
        src.count('"disallowed_tools": self._web_disallowed_tools')
        + inspect.getsource(lens_planning).count(
            '"disallowed_tools": web_disallowed_tools'
        )
    )
    assert wired >= 3
    assert 'web_disallowed_tools=self._web_disallowed_tools' in src


def test_bobai_acquire_manifests_sequester_answers():
    from benchmarks.ioai_tasks.data import acquire_bobai as a

    # contest/ must never carry the reference solution or test labels;
    # gold/ must carry exactly the answer material.
    assert not (set(a.CONTEST) & set(a.GOLD))
    assert any(v.lower().endswith(".ipynb") for v in a.GOLD.values())  # ref soln
    assert any("test_set" in v.lower() and "label" in v.lower()
               for v in a.GOLD.values())                                # test key
    for v in a.CONTEST.values():
        assert not v.lower().endswith(".ipynb")   # no reference solution
        assert "/test_set/" not in v.lower()      # no test answer material


def test_harvest_config_is_fable_max_and_single_sourced():
    from benchmarks.ioai_tasks.harvest.harvest_runner import _harvest_config

    cfg = _harvest_config()
    assert cfg["model"] == "claude-fable-5"
    assert cfg["effort"] == "max"
    assert set(cfg["allowed_tools"]) >= {"Read", "Write", "Bash"}
    assert isinstance(cfg["timeout_seconds"], int)


def test_evaluator_fail_loud_on_bad_shape(tmp_path):
    torch = pytest.importorskip("torch")
    from benchmarks.ioai_tasks.data import bobai_evaluate

    bad = tmp_path / "bad.pt"
    torch.save(torch.zeros(3, 5), bad)  # not [N,1,769]
    sol = tmp_path / "solution.py"
    sol.write_text("class Solution:\n"
                   "    def __init__(self, d): pass\n"
                   "    def predict(self, X): return [0]\n")
    import sys
    argv = ["evaluate.py", "--data", str(bad), "--data-dir", str(tmp_path),
            "--solution", f"{sol}:Solution"]
    old = sys.argv
    sys.argv = argv
    with pytest.raises(ValueError, match="769"):
        bobai_evaluate.main()
    sys.argv = old
