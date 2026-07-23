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


def test_config_has_ensemble_and_selector_for_k2():
    with open(CONFIG_PATH) as f:
        mode = yaml.safe_load(f)["modes"]["LOCAL"]
    params = mode["search_strategy"]["params"]
    assert len(params["ideation_ensemble"]) == 2
    assert params["ideation_selector"]["cli"] == "claude_code"
    assert mode["budget"] == {"min_iteration_seconds": 900}


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
