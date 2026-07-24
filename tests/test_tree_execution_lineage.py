import threading
from types import SimpleNamespace

from kapso.execution.search_strategies.benchmark_tree_search import (
    BenchmarkTreeSearch,
    TreeSearchNode,
)
from kapso.execution.types import ContextData

_BASE_COMMIT = "b" * 40


class _Repository:
    def commit(self, ref):
        if ref != "main":
            raise AssertionError("tree execution resolved the wrong parent ref")
        return SimpleNamespace(hexsha=_BASE_COMMIT)


class _Workspace:
    repo = _Repository()

    @staticmethod
    def get_current_branch():
        return "main"


def test_tree_execution_freezes_one_parent_commit_for_all_lineage() -> None:
    strategy = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
    strategy.experimentation_count = 1
    strategy.workspace = _Workspace()
    strategy.workspace_dir = "/test/workspace"
    strategy.node_history_lock = threading.Lock()
    strategy.node_history = []
    strategy.previous_errors = []
    observed_implementation_bases = []
    strategy._implement = (
        lambda *args, **kwargs: observed_implementation_bases.append(
            kwargs["parent_branch_name"]
        )
        or "implemented"
    )
    observed_diff_bases = []
    strategy._get_code_diff = (
        lambda branch, base: observed_diff_bases.append((branch, base)) or "diff"
    )
    strategy.enforce_evaluation_integrity = lambda node: False

    root = TreeSearchNode(node_id=0, branch_name="main", score=0.0)
    node = TreeSearchNode(
        node_id=1,
        parent_node=root,
        solution="candidate",
    )

    strategy._run_for_node(
        node,
        ContextData(
            problem="test",
            additional_info="",
            kg_results="",
            kg_code_results="",
        ),
        "candidate-1",
    )

    assert node.parent_branch_name == "main"
    assert node.implementation_base_ref == _BASE_COMMIT
    assert node.diff_base_ref == _BASE_COMMIT
    assert node.feedback_base_ref == _BASE_COMMIT
    assert observed_implementation_bases == [_BASE_COMMIT]
    assert observed_diff_bases == [("candidate-1", _BASE_COMMIT)]
    assert strategy.node_history == [node]
