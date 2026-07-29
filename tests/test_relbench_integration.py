"""Unit tests for the RelBench benchmark integration.

Fast, network-free tests: prediction-contract validation, audit scoping,
primary-metric routing, and the candidate-materialization fix in
benchmark_tree_search. Tests that need a populated relbench cache are skipped
unless the cache exists.
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from benchmarks.relbench.task_specs import (
    AUTOCOMPLETE_REGRESSION,
    ENTITY_BINARY,
    PRIMARY_METRIC_OVERRIDES,
    RECOMMENDATION,
    TaskSpec,
)


def make_handler(spec: TaskSpec, n_val=10, n_test=8):
    """Build a RelBenchHandler shell without touching relbench/data."""
    from benchmarks.relbench.handler import RelBenchHandler

    handler = RelBenchHandler.__new__(RelBenchHandler)
    handler.spec = spec
    handler.n_val = n_val
    handler.n_test = n_test
    return handler


def binary_spec():
    return TaskSpec(
        dataset_name="rel-x", task_name="t", family=ENTITY_BINARY,
        primary_metric="roc_auc", maximize=True,
    )


def rec_spec():
    return TaskSpec(
        dataset_name="rel-x", task_name="t", family=RECOMMENDATION,
        primary_metric="link_prediction_map", maximize=True,
        eval_k=5, num_dst_nodes=100,
    )


class TestPredictionValidation:
    def _write(self, tmp_path, split, arr):
        np.save(tmp_path / f"{split}_predictions.npy", arr)
        return tmp_path

    def test_missing_file_rejected(self, tmp_path):
        from benchmarks.relbench.handler import PredictionContractError

        handler = make_handler(binary_spec())
        with pytest.raises(PredictionContractError, match="not written"):
            handler._load_predictions(str(tmp_path), "val")

    def test_binary_range_enforced(self, tmp_path):
        from benchmarks.relbench.handler import PredictionContractError

        handler = make_handler(binary_spec())
        self._write(tmp_path, "val", np.linspace(-3, 3, 10))
        with pytest.raises(PredictionContractError, match="sigmoid"):
            handler._load_predictions(str(tmp_path), "val")

    def test_binary_ok_and_column_squeeze(self, tmp_path):
        handler = make_handler(binary_spec())
        self._write(tmp_path, "val", np.full((10, 1), 0.5))
        arr, warnings = handler._load_predictions(str(tmp_path), "val")
        assert arr.shape == (10,) and not warnings

    def test_nan_rejected(self, tmp_path):
        from benchmarks.relbench.handler import PredictionContractError

        handler = make_handler(binary_spec())
        bad = np.full(10, 0.5)
        bad[3] = np.nan
        self._write(tmp_path, "val", bad)
        with pytest.raises(PredictionContractError, match="NaN"):
            handler._load_predictions(str(tmp_path), "val")

    def test_wrong_shape_rejected(self, tmp_path):
        from benchmarks.relbench.handler import PredictionContractError

        handler = make_handler(binary_spec())
        self._write(tmp_path, "val", np.full(7, 0.5))
        with pytest.raises(PredictionContractError, match="shape"):
            handler._load_predictions(str(tmp_path), "val")

    def test_recommendation_shape_and_warnings(self, tmp_path):
        handler = make_handler(rec_spec())
        pred = np.tile(np.arange(5), (10, 1))
        pred[0] = [1, 1, 2, 3, 4]      # duplicate
        pred[1] = [0, 1, 2, 3, 999]    # out of range
        self._write(tmp_path, "val", pred)
        arr, warnings = handler._load_predictions(str(tmp_path), "val")
        assert arr.shape == (10, 5)
        assert any("duplicate" in w for w in warnings)
        assert any("out of" in w for w in warnings)

    def test_recommendation_float_ids_rejected(self, tmp_path):
        from benchmarks.relbench.handler import PredictionContractError

        handler = make_handler(rec_spec())
        self._write(tmp_path, "val", np.random.rand(10, 5))
        with pytest.raises(PredictionContractError, match="integer"):
            handler._load_predictions(str(tmp_path), "val")


class TestAudit:
    def test_flags_violations_and_skips_vendored(self, tmp_path):
        spec = binary_spec()
        handler = make_handler(spec)
        code = tmp_path / "code"
        code.mkdir()
        (code / "main.py").write_text(
            "t = task.get_table('test', mask_input_cols=False)\n"
        )
        kit_file = (
            Path(__file__).parents[1]
            / "benchmarks" / "relbench" / "data" / "starter_kit" / "common.py"
        )
        (code / "common.py").write_bytes(kit_file.read_bytes())
        report = handler._audit_code(code)
        assert report["clean"] is False
        files = {f["file"] for f in report["findings"]}
        assert "main.py" in files and "common.py" not in files

    def test_autocomplete_allows_full_db(self, tmp_path):
        spec = TaskSpec(
            dataset_name="rel-x", task_name="t", family=AUTOCOMPLETE_REGRESSION,
            primary_metric="r2", maximize=True,
        )
        handler = make_handler(spec)
        code = tmp_path / "code"
        code.mkdir()
        (code / "main.py").write_text("db = ds.get_db(upto_test_timestamp=False)\n")
        assert handler._audit_code(code)["clean"] is True


class TestMetricRouting:
    def test_v2_regression_tasks_optimize_r2(self):
        assert PRIMARY_METRIC_OVERRIDES["rel-trial/studies-enrollment"] == ("r2", True)
        assert PRIMARY_METRIC_OVERRIDES["rel-amazon/review-rating"] == ("r2", True)

    def test_direction_helpers(self):
        handler = make_handler(binary_spec())
        assert handler._is_better(0.8, 0.7)
        spec = TaskSpec(
            dataset_name="d", task_name="t", family="entity_regression",
            primary_metric="mae", maximize=False,
        )
        handler = make_handler(spec)
        assert handler._is_better(0.3, 0.4)


class TestCandidateMaterialization:
    """The benchmark strategy must evaluate the node's committed branch."""

    def test_run_handler_materializes_branch(self):
        from kapso.execution.search_strategies.benchmark_tree_search import (
            BenchmarkTreeSearch,
        )

        strategy = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
        strategy.workspace_dir = "/ws"

        @contextmanager
        def fake_materialize(ref):
            assert ref == "experiment_3"
            yield "/tmp/candidate_xyz"

        strategy.workspace = SimpleNamespace(materialize_ref=fake_materialize)
        handler = MagicMock()
        strategy.problem_handler = handler
        node = SimpleNamespace(branch_name="experiment_3")

        strategy._run_handler_on_candidate(node, "sol", "/ws/kapso_evaluation")
        handler.run.assert_called_once_with(
            file_path="/tmp/candidate_xyz",
            run_data_dir="/ws/kapso_evaluation",
            solution="sol",
        )

    def test_falls_back_to_workspace_on_unknown_ref(self):
        from kapso.execution.search_strategies.benchmark_tree_search import (
            BenchmarkTreeSearch,
        )

        strategy = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
        strategy.workspace_dir = "/ws"

        @contextmanager
        def raise_materialize(ref):
            raise ValueError("Unknown Git ref")
            yield  # pragma: no cover

        strategy.workspace = SimpleNamespace(materialize_ref=raise_materialize)
        handler = MagicMock()
        strategy.problem_handler = handler
        node = SimpleNamespace(branch_name="missing")

        strategy._run_handler_on_candidate(node, "sol", "/rd")
        handler.run.assert_called_once_with(
            file_path="/ws", run_data_dir="/rd", solution="sol"
        )


RELBENCH_CACHE = Path(
    os.environ.get("RELBENCH_PRISTINE_CACHE_DIR", os.path.expanduser("~/.cache/relbench"))
)


@pytest.mark.skipif(
    not (RELBENCH_CACHE / "rel-f1" / "db").exists(),
    reason="requires a populated rel-f1 relbench cache",
)
class TestSandboxOnRealData:
    def test_forecasting_cache_is_leak_free(self, tmp_path):
        import subprocess
        import sys

        dest = tmp_path / "cache"
        repo_root = Path(__file__).parents[1]
        env = os.environ.copy()
        env.pop("RELBENCH_CACHE_DIR", None)
        env["PYTHONPATH"] = str(repo_root)
        subprocess.run(
            [sys.executable, "-m", "benchmarks.relbench.sandbox",
             "--dataset", "rel-f1", "--task", "driver-position", "--dest", str(dest)],
            cwd=repo_root, env=env, check=True, capture_output=True,
        )
        import pandas as pd

        test_df = pd.read_parquet(dest / "rel-f1" / "tasks" / "driver-position" / "test.parquet")
        assert "position" not in test_df.columns
        races = pd.read_parquet(dest / "rel-f1" / "db" / "races.parquet")
        assert str(races["date"].max()) < "2010-01-02"
        assert not os.access(dest / "rel-f1" / "db" / "races.parquet", os.W_OK)


class TestExperimentHistoryDigest:
    """Fix B: node history must reach ideation/selection via additional_info."""

    def _strategy_with_history(self):
        from kapso.execution.search_strategies.benchmark_tree_search import (
            BenchmarkTreeSearch,
        )
        import threading

        s = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
        s.node_history_lock = threading.Lock()
        s.problem_handler = SimpleNamespace(maximize_scoring=False)
        failed = SimpleNamespace(
            branch_name="experiment_0", node_id=13, had_error=True, score=1e18,
            error_message="Debug execution took 900s (exceeded 15 minute limit).",
            evaluation_output="", solution="Stacked LightGBM with cutoff emulation",
        )
        scored = SimpleNamespace(
            branch_name="experiment_1", node_id=18, had_error=False, score=2.71,
            error_message="",
            evaluation_output="stuff\nOFFICIAL VALIDATION METRICS (harness-computed): "
            '{"mae": 2.71}\nmore',
            solution="CatBoost sequence + state-space model",
        )
        s.node_history = [failed, scored]
        return s

    def test_digest_contains_failures_scores_and_direction(self):
        digest = self._strategy_with_history()._experiment_history_digest()
        assert "lower is better" in digest
        assert "experiment_0 FAILED" in digest and "15 minute limit" in digest
        assert "experiment_1 score=2.71" in digest
        assert "OFFICIAL VALIDATION METRICS" in digest
        assert "Stacked LightGBM" in digest

    def test_empty_history_yields_empty_digest(self):
        s = self._strategy_with_history()
        s.node_history = []
        assert s._experiment_history_digest() == ""


class TestSelectionLineageAndReasoning:
    """Findings 5+8: candidates carry lineage; LLM reasoning is logged."""

    def _tree(self):
        from kapso.execution.search_strategies.benchmark_tree_search import (
            BenchmarkTreeSearch,
            TreeSearchNode,
        )

        parent_scored = TreeSearchNode(
            node_id=1, branch_name="experiment_1", solution="scored parent"
        )
        parent_scored.score = 2.684
        parent_unscored = TreeSearchNode(node_id=2, solution="unscored parent")
        child_a = TreeSearchNode(
            node_id=10, parent_node=parent_scored, solution="child of the winner"
        )
        child_b = TreeSearchNode(
            node_id=11, parent_node=parent_unscored, solution="child of unknown"
        )
        parent_scored.children.append(child_a)
        parent_unscored.children.append(child_b)

        strategy = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
        strategy.nodes = [parent_scored, parent_unscored, child_a, child_b]
        strategy.idea_generation_model = "test-model"
        strategy.reasoning_effort = "low"
        strategy.experimentation_count = 3
        return strategy, child_a, child_b

    def test_candidate_line_lineage_cases(self):
        strategy, child_a, child_b = self._tree()
        root_line = strategy._candidate_line(strategy.nodes[0])
        assert "[root]" in root_line
        scored_line = strategy._candidate_line(child_a)
        assert "child of experiment_1, parent score 2.684" in scored_line
        unscored_line = strategy._candidate_line(child_b)
        assert "child of unscored node 2" in unscored_line

    def test_select_prompt_carries_lineage_and_logs_reasoning(self, capsys):
        strategy, child_a, child_b = self._tree()
        captured = {}

        def fake_llm(**kwargs):
            captured.update(kwargs)
            return (
                "Reason for solution id 10: strongest lineage.\n"
                "<output>\n[10]\n</output>"
            )

        strategy.llm = SimpleNamespace(llm_completion_with_system_prompt=fake_llm)
        picked = strategy.select(
            SimpleNamespace(problem="p", additional_info="", kg_results=""),
            top_k=1,
        )
        assert picked == [child_a]
        assert "parent score 2.684" in captured["user_message"]
        assert "child of unscored node 2" in captured["user_message"]
        out = capsys.readouterr().out
        assert "selection (top_k=1) reasoning" in out
        assert "strongest lineage" in out

    def test_prune_logs_reasoning_and_terminates(self, capsys):
        strategy, child_a, child_b = self._tree()

        def fake_llm(**kwargs):
            return "Reason 11: dead end.\n<output>\n[11]\n</output>"

        strategy.llm = SimpleNamespace(llm_completion_with_system_prompt=fake_llm)
        strategy.prune_bad_solutions(
            SimpleNamespace(problem="p", additional_info="", kg_results="")
        )
        assert child_b.is_terminated and not child_a.is_terminated
        assert "pruning reasoning" in capsys.readouterr().out


class TestRepoMemoryMcpMount:
    """Finding 9: sessions mount the repo-memory MCP gate for MCP-capable agents."""

    def _config(self, agent_type="claude_code", agent_specific=None):
        from kapso.execution.coding_agents.base import CodingAgentConfig

        return CodingAgentConfig(
            agent_type=agent_type, model="m", debug_model="m",
            agent_specific=agent_specific if agent_specific is not None else {},
        )

    def test_claude_code_gets_gate_mounted(self, tmp_path):
        from kapso.execution.experiment_workspace.experiment_session import (
            ExperimentSession,
        )

        config = self._config()
        mounted = ExperimentSession._mount_repo_memory_mcp(config, str(tmp_path))
        assert mounted is True
        servers = config.agent_specific["mcp_servers"]
        (server_conf,) = servers.values()
        assert server_conf["env"]["REPO_MEMORY_ROOT"] == str(tmp_path)
        tools = config.agent_specific["allowed_tools"]
        assert "Bash" in tools
        assert any("get_repo_memory_section" in t for t in tools)
        assert any("get_repo_memory_summary" in t for t in tools)

    def test_preset_mcp_servers_passthrough(self, tmp_path):
        from kapso.execution.experiment_workspace.experiment_session import (
            ExperimentSession,
        )

        preset = {"mcp_servers": {"custom": {"command": "x"}}}
        config = self._config(agent_specific=dict(preset))
        mounted = ExperimentSession._mount_repo_memory_mcp(config, str(tmp_path))
        assert mounted is True
        assert config.agent_specific["mcp_servers"] == preset["mcp_servers"]
        assert "allowed_tools" not in config.agent_specific

    def test_non_mcp_agent_untouched(self, tmp_path):
        from kapso.execution.experiment_workspace.experiment_session import (
            ExperimentSession,
        )

        config = self._config(agent_type="aider")
        mounted = ExperimentSession._mount_repo_memory_mcp(config, str(tmp_path))
        assert mounted is False
        assert "mcp_servers" not in config.agent_specific

    def test_implement_instructions_follow_mount_state(self, tmp_path):
        from kapso.execution.search_strategies.benchmark_tree_search import (
            BenchmarkTreeSearch,
        )

        strategy = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
        strategy.previous_errors = []
        strategy.recent_error_count = 10
        context = SimpleNamespace(problem="p", kg_code_results="")
        prompts = {}

        def session(mounted):
            return SimpleNamespace(
                session_folder=str(tmp_path),
                branch_name="experiment_x",
                repo_memory_mcp_mounted=mounted,
                generate_code=lambda prompt: (
                    prompts.__setitem__("last", prompt),
                    SimpleNamespace(output="done"),
                )[1],
            )

        strategy.implement_solution("sol", context, session(True))
        assert "use the MCP tool" in prompts["last"]
        strategy.implement_solution("sol", context, session(False))
        assert "read: `.kapso/repo_memory.json`" in prompts["last"]


RELBENCH_CACHE = Path(
    os.environ.get("RELBENCH_PRISTINE_CACHE_DIR", os.path.expanduser("~/.cache/relbench"))
)


class TestGenericModeConfig:
    def test_mode_parses_with_maintainer_block(self):
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).parents[1] / "benchmarks/relbench/config.yaml").read_text()
        )
        mode = cfg["modes"]["RELBENCH_GENERIC"]
        assert mode["search_strategy"]["type"] == "generic"
        assert mode["search_strategy"]["params"]["parent_policy"] == "best"
        # Codex deployment (2026-07-26): implementation, maintainer, and
        # feedback all run on the codex CLI at the xhigh OpenAI ceiling.
        assert mode["search_strategy"]["params"]["implementation_cli"] == "codex"
        assert mode["evaluation_maintainer"]["type"] == "codex"
        assert mode["feedback_generator"]["type"] == "codex"
        assert mode["search_strategy"]["params"]["ideation_selector"]["cli"] == "codex"
        assert mode["models"]["utility"]["reasoning_effort"] == "xhigh"
        # K=4 expansion (user-directed 2026-07-29, back from the K=8
        # escalation after 8-GPU capacity drought): four GPU-pinned lanes,
        # 10h budget, 5h session caps — the proven record-run shape.
        assert mode["search_strategy"]["params"]["node_expansion_value"] == 4
        lanes = mode["search_strategy"]["params"]["expansion_lane_env"]
        assert len(lanes) == 4
        assert all(lane["OMP_NUM_THREADS"] == "24" for lane in lanes)
        assert [lane["CUDA_VISIBLE_DEVICES"] for lane in lanes] == ["0", "1", "2", "3"]
        assert mode["search_strategy"]["params"]["implementation_timeout"] == 18000
        assert mode["budget"]["time_budget_minutes"] == 600


@pytest.mark.skipif(
    not (RELBENCH_CACHE / "rel-f1" / "db").exists(),
    reason="requires a populated rel-f1 relbench cache",
)
class TestProvidedGrader:
    def _repo(self, tmp_path):
        import subprocess
        import sys as _sys

        root = tmp_path / "candidate"
        (root / "kapso_evaluation").mkdir(parents=True)
        suite = Path(__file__).parents[1] / "benchmarks/relbench/data/generic_eval"
        for f in suite.glob("*"):
            (root / "kapso_evaluation" / f.name).write_bytes(f.read_bytes())
        (root / "main.py").write_text(
            "import os, numpy as np\n"
            "from relbench.tasks import get_task\n"
            "t = get_task(os.environ['RELBENCH_DATASET'], os.environ['RELBENCH_TASK'], download=False)\n"
            "n_val = len(t.get_table('val'))\n"
            "n_test = len(t.get_table('test'))\n"
            "out = os.environ['KAPSO_RUN_DATA_DIR']\n"
            "np.save(f'{out}/val_predictions.npy', np.full(n_val, 13.0))\n"
            "np.save(f'{out}/test_predictions.npy', np.full(n_test, 13.0))\n"
            "print('fixture candidate done, debug=' + str('--debug' in __import__('sys').argv))\n"
        )
        env = os.environ.copy()
        env.update(
            {
                "RELBENCH_CACHE_DIR": str(RELBENCH_CACHE),
                "RELBENCH_DATASET": "rel-f1",
                "RELBENCH_TASK": "driver-position",
                "RELBENCH_PRIMARY_METRIC": "mae",
                "RELBENCH_WORK_DIR": str(tmp_path / "work"),
                "RELBENCH_FULL_TIMEOUT": "300",
                "RELBENCH_DEBUG_TIMEOUT": "300",
            }
        )

        def run(fidelity):
            return subprocess.run(
                [_sys.executable, "kapso_evaluation/grader.py", "--fidelity", fidelity,
                 "--fraction", "1.0", "--seed", "1337"],
                cwd=root, env=env, capture_output=True, text=True, timeout=300,
            )
        return root, tmp_path / "work", run

    def test_fast_scores_without_archiving(self, tmp_path):
        _root, work, run = self._repo(tmp_path)
        proc = run("fast")
        assert proc.returncode == 0, proc.stdout[-2000:]
        manifest = json.loads(proc.stdout.strip().splitlines()[-1].split(" ", 1)[1])
        assert manifest["fidelity"] == "fast" and manifest["score"] > 0
        assert not (work / "runs").exists()

    def test_full_archives_with_code_snapshot(self, tmp_path):
        _root, work, run = self._repo(tmp_path)
        proc = run("full")
        assert proc.returncode == 0, proc.stdout[-2000:]
        manifest = json.loads(proc.stdout.strip().splitlines()[-1].split(" ", 1)[1])
        run_dir = work / "runs" / "run_0001"
        metrics = json.loads((run_dir / "private/metrics.json").read_text())
        assert abs(metrics["val"]["mae"] - manifest["score"]) < 1e-9
        assert metrics["test"] == {}
        assert (run_dir / "test_predictions.npy").exists()
        assert (run_dir / "code" / "main.py").exists()
        assert not (run_dir / "code" / "kapso_evaluation").exists()


@pytest.mark.skipif(
    not (RELBENCH_CACHE / "rel-f1" / "db").exists(),
    reason="requires a populated rel-f1 relbench cache",
)
class TestFinalEvaluateTestFill:
    def test_val_only_archive_gets_test_scored_once(self, tmp_path):
        from relbench.tasks import get_task

        from benchmarks.relbench.handler import RelBenchHandler

        os.environ["RELBENCH_CACHE_DIR"] = str(RELBENCH_CACHE)
        task = get_task("rel-f1", "driver-position", download=False)
        n_test = len(task.get_table("test"))

        handler = RelBenchHandler.__new__(RelBenchHandler)
        handler.task = task
        handler.spec = TaskSpec(
            dataset_name="rel-f1", task_name="driver-position",
            family="entity_regression", primary_metric="mae", maximize=False,
        )
        handler.dataset_name, handler.task_name = "rel-f1", "driver-position"
        handler.runs_dir = tmp_path / "runs"
        handler.work_dir = tmp_path

        run_dir = handler.runs_dir / "run_0001"
        (run_dir / "private").mkdir(parents=True)
        np.save(run_dir / "test_predictions.npy", np.full(n_test, 13.0))
        np.save(run_dir / "val_predictions.npy", np.full(3, 13.0))
        (run_dir / "private/metrics.json").write_text(
            json.dumps({"val": {"mae": 3.0, "r2": 0.1, "rmse": 4.0}, "test": {}})
        )

        report = handler.final_evaluate()
        assert report["test_metrics"]["mae"] > 0
        on_disk = json.loads((run_dir / "private/metrics.json").read_text())
        assert on_disk["test"] == report["test_metrics"]


class TestDataAccessRules:
    def test_external_leverage_fully_relaxed_rules(self):
        """User-directed data policy (final relaxation 2026-07-28):
        pretrained models unconditionally allowed and ENCOURAGED with NO
        carve-outs — the baseline-weights exclusion is fully lifted (no
        method names in the rules); the relbench-API/cache mechanics block
        and the determinism bullet are dropped (cache protection stays
        physical: sanitized read-only cache). What remains: the published-
        solution ban, external-dataset leakage condition with changes.log
        provenance, synthetic-data legality, temporal censoring, the
        two-model val contract, and the forbidden API calls."""
        from types import SimpleNamespace

        from benchmarks.relbench.context import _data_access_rules

        spec = SimpleNamespace(
            is_autocomplete=False,
            dataset_name="rel-f1",
            task_name="driver-position",
            time_col="date",
            is_recommendation=False,
        )
        rules = _data_access_rules(spec)
        assert "do NOT look up this problem's published solution" in rules
        assert "PRETRAINED MODELS (encouraged)" in rules
        assert "EXTERNAL DATASETS (encouraged)" in rules
        assert "ZERO leakage" in rules and "voids the experiment" in rules
        assert "changes.log" in rules
        assert "SYNTHETIC DATA" in rules and "legal" in rules
        assert "Temporal censoring" in rules
        assert "two-model" in rules
        assert "mask_input_cols=False" in rules
        # Removed-by-direction pins: no baseline-weights exclusion, no API/
        # cache mechanics block, no determinism bullet, no model constraints.
        for gone in ("KumoRFM", "Relational Transformer", "PluRel", "Griffin",
                     "Rel-LLM", "WEIGHTS only", "carve-out", "download=False",
                     "OFFICIAL RelBench distribution", "Determinism",
                     "WORLD-KNOWLEDGE", "NEEDS CENSORING", "opaque ids"):
            assert gone not in rules, gone


class TestDesignAxes:
    def test_design_axes_configured_and_fe_standing_direction(self):
        """Anti-freeze integration (user-directed 2026-07-27): the relbench
        mode declares task-specific design axes with feature engineering as
        the first axis, and the problem context carries the standing
        feature-engineering note (matrix never finished; unchanged only on
        measured saturation evidence)."""
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).parents[1] / "benchmarks/relbench/config.yaml").read_text()
        )
        axes = cfg["modes"]["RELBENCH_GENERIC"]["search_strategy"]["params"][
            "design_axes"
        ]
        assert isinstance(axes, list) and len(axes) == 5
        assert "feature engineering" in axes[0]
        assert any("training distribution" in axis for axis in axes)

        from benchmarks.relbench.context import FEATURE_ENGINEERING_NOTE

        assert "standing" in FEATURE_ENGINEERING_NOTE
        assert "never finished" in FEATURE_ENGINEERING_NOTE
        assert "saturation" in FEATURE_ENGINEERING_NOTE
        assert "freezing the feature matrix" in FEATURE_ENGINEERING_NOTE


class TestLivingDocuments:
    def _fake_db_dataset(self):
        import pandas as pd

        class FakeTable:
            def __init__(self, df, pkey, time_col, fkeys):
                self.df = df
                self.pkey_col = pkey
                self.time_col = time_col
                self.fkey_col_to_pkey_table = fkeys

        results = FakeTable(
            pd.DataFrame({"resultId": [1], "raceId": [1], "driverId": [1]}),
            "resultId", "date", {"raceId": "races", "driverId": "drivers"},
        )
        races = FakeTable(
            pd.DataFrame({"raceId": [1], "circuitId": [1], "round": [1]}),
            "raceId", "date", {"circuitId": "circuits"},
        )
        circuits = FakeTable(
            pd.DataFrame({"circuitId": [1], "country": ["x"]}),
            "circuitId", None, {},
        )
        drivers = FakeTable(
            pd.DataFrame({"driverId": [1]}), "driverId", None, {}
        )
        db = SimpleNamespace(
            table_dict={
                "results": results, "races": races,
                "circuits": circuits, "drivers": drivers,
            },
            min_timestamp="1950", max_timestamp="2009",
        )
        dataset = SimpleNamespace(val_timestamp="2005", test_timestamp="2010")
        return db, dataset

    def test_table_information_has_schema_join_graph_and_two_hop(self):
        """The seeded living doc must expose the full join graph including
        the two-hop bridge paths past campaigns never built (results ->
        races -> circuits was unread across 55 registered runs)."""
        from benchmarks.relbench.context import build_table_information

        db, dataset = self._fake_db_dataset()
        doc = build_table_information(db, dataset, "rel-x")
        assert "LIVING DOCUMENT" in doc
        for t in ("results", "races", "circuits", "drivers"):
            assert f"table `{t}`" in doc
        assert "- `results.raceId` -> `races`" in doc
        assert "- `races.circuitId` -> `circuits`" in doc
        assert "`results` -> `races` (via `raceId`) -> `circuits` (via `circuitId`)" in doc
        assert "Semantics and gotchas" in doc

    def test_seeding_is_absent_only(self, tmp_path):
        """Agent edits must survive handler restarts/resumes: an existing
        living doc is never overwritten."""
        from benchmarks.relbench.handler import RelBenchHandler

        handler = RelBenchHandler.__new__(RelBenchHandler)
        handler.shared_cache_dir = tmp_path
        handler.dataset_name = "rel-x"
        handler.problem_id = "rel-x--t"
        db, dataset = self._fake_db_dataset()
        handler.dataset = SimpleNamespace(get_db=lambda: db, **vars(dataset))

        handler._seed_living_documents()
        info = tmp_path / "table_information.md"
        hist = tmp_path / "features_history.md"
        assert info.exists() and hist.exists()
        assert "rel-x--t" in hist.read_text()

        info.write_text("AGENT EDITED\n")
        hist.write_text("AGENT MEMORY\n")
        handler._seed_living_documents()
        assert info.read_text() == "AGENT EDITED\n"
        assert hist.read_text() == "AGENT MEMORY\n"

    def test_context_carries_living_docs_and_hard_rules(self):
        """User-directed (2026-07-28): all-tables hard rule, features-over-
        architecture preference, and the living-documents contract must be
        in the problem context constants."""
        from benchmarks.relbench.context import (
            FEATURE_ENGINEERING_NOTE,
            FEATURES_HISTORY_TEMPLATE,
            LIVING_DOCUMENTS_NOTE,
        )

        assert "ALL TABLES (hard rule)" in FEATURE_ENGINEERING_NOTE
        assert "EVERY table" in FEATURE_ENGINEERING_NOTE
        assert "MEASURED reason" in FEATURE_ENGINEERING_NOTE
        assert "99%" in FEATURE_ENGINEERING_NOTE
        assert "FEATURES OVER ARCHITECTURE" in FEATURE_ENGINEERING_NOTE
        from benchmarks.relbench.context import BEST_PRACTICES_NOTE

        assert "INFORMATION" in BEST_PRACTICES_NOTE
        assert "REGIME-SHIFTED" in BEST_PRACTICES_NOTE
        assert "frozen-origin" in BEST_PRACTICES_NOTE
        assert "follow the measurement" in BEST_PRACTICES_NOTE
        assert "table_information.md" in LIVING_DOCUMENTS_NOTE
        assert "features_history.md" in LIVING_DOCUMENTS_NOTE
        assert "EDIT" in LIVING_DOCUMENTS_NOTE
        assert "Append-only" in LIVING_DOCUMENTS_NOTE
        assert "$KAPSO_SHARED_CACHE_DIR" in LIVING_DOCUMENTS_NOTE
        assert "TESTED-REJECTED" in FEATURES_HISTORY_TEMPLATE
