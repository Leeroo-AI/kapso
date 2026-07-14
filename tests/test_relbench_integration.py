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
