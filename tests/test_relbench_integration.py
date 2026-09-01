"""Unit tests for the RelBench benchmark integration.

Fast, network-free tests: prediction-contract validation, audit scoping,
primary-metric routing, and the candidate-materialization fix in
the generic strategy. Tests that need a populated relbench cache are skipped
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

# The relbench package is installed separately (pip install relbench); it is
# not a declared extra. Without it these skip rather than erroring at import.
pytest.importorskip("relbench")

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
        # No mode-level model routes: the CLI-only conversion made
        # ModelRouter embedding-only (stale-code audit 2026-08-26) —
        # a utility/reasoning block here would ValueError at
        # orchestrator construction.
        assert "models" not in mode
        # K=4 expansion (user-directed 2026-07-29): four GPU-pinned lanes,
        # 10h budget, 5h session caps; threads sized for the on-demand
        # a2-highgpu-4g fallback (48 vCPU -> 11/lane).
        assert mode["search_strategy"]["params"]["node_expansion_value"] == 4
        lanes = mode["search_strategy"]["params"]["expansion_lane_env"]
        assert len(lanes) == 4
        assert all(lane["OMP_NUM_THREADS"] == "11" for lane in lanes)
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
        # Governance contract: the archive is stamped with the evaluator that
        # produced it, and that evaluator's tree is snapshotted under the id.
        label = json.loads((run_dir / "private/selection.json").read_text())
        assert label["evaluator_id"] == manifest["evaluator_id"]
        snapshot = work / "evaluators" / manifest["evaluator_id"]
        assert (snapshot / "grader.py").exists()
        assert (snapshot / "kapso_eval_archive.py").exists()

    def test_rescore_reproduces_the_archived_score(self, tmp_path):
        """run <-> --rescore agreement on the real grader and real data: the
        recomputation from stored predictions must equal the archive-time
        score bit-for-bit, because final selection treats any disagreement
        as tampering and refuses to ship."""
        import subprocess
        import sys as _sys

        root, work, run = self._repo(tmp_path)
        proc = run("full")
        assert proc.returncode == 0, proc.stdout[-2000:]
        archived = json.loads(
            proc.stdout.strip().splitlines()[-1].split(" ", 1)[1]
        )
        env = os.environ.copy()
        env.update(
            {
                "RELBENCH_CACHE_DIR": str(RELBENCH_CACHE),
                "RELBENCH_DATASET": "rel-f1",
                "RELBENCH_TASK": "driver-position",
                "RELBENCH_PRIMARY_METRIC": "mae",
            }
        )
        rescore = subprocess.run(
            [_sys.executable, "kapso_evaluation/grader.py",
             "--rescore", str(work / "runs" / "run_0001")],
            cwd=root, env=env, capture_output=True, text=True, timeout=300,
        )
        assert rescore.returncode == 0, rescore.stdout[-2000:]
        payload = json.loads(
            rescore.stdout.strip().splitlines()[-1].split(" ", 1)[1]
        )
        assert payload["mode"] == "rescore"
        assert payload["run"] == "run_0001"
        assert payload["score"] == archived["score"]
        assert payload["evaluator_id"] == archived["evaluator_id"]
        assert payload["metrics"]["mae"] == archived["score"]


def _bare_driver_position_handler(tmp_path):
    from relbench.tasks import get_task

    from benchmarks.relbench.handler import RelBenchHandler

    os.environ["RELBENCH_CACHE_DIR"] = str(RELBENCH_CACHE)
    task = get_task("rel-f1", "driver-position", download=False)
    handler = RelBenchHandler.__new__(RelBenchHandler)
    handler.task = task
    handler.spec = TaskSpec(
        dataset_name="rel-f1", task_name="driver-position",
        family="entity_regression", primary_metric="mae", maximize=False,
    )
    handler.dataset_name, handler.task_name = "rel-f1", "driver-position"
    handler.runs_dir = tmp_path / "runs"
    handler.work_dir = tmp_path
    return handler, task


GENERIC_EVAL_DIR = Path(__file__).parents[1] / "benchmarks/relbench/data/generic_eval"

# The maintainer-built entrypoint is a thin forwarder onto the provided
# grader — this is the wrapper shape the registered contract expects, and
# using it here proves forwarding covers --rescore.
FORWARDING_WRAPPER = (
    "import subprocess\n"
    "import sys\n"
    "from pathlib import Path\n"
    "raise SystemExit(subprocess.call(\n"
    "    [sys.executable, str(Path(__file__).resolve().parent / 'grader.py'),\n"
    "     *sys.argv[1:]]))\n"
)


def _governed_head(work_dir):
    """Snapshot a real evaluator tree (provided suite + wrapper) and return
    its fingerprint — the head every archived run must be stamped with."""
    from kapso.execution import evaluation_archive_sandbox as sandbox

    source = work_dir / "eval_source"
    source.mkdir()
    for f in sorted(GENERIC_EVAL_DIR.glob("*")):
        if f.is_file():
            (source / f.name).write_bytes(f.read_bytes())
    (source / "kapso_eval.py").write_text(FORWARDING_WRAPPER)
    head = sandbox.fingerprint_tree(source)
    sandbox.snapshot_evaluator_tree(work_dir, source, head)
    return head


def _archive_run(runs_dir, name, val_pred, test_pred, status, session="exp_A",
                 forged_val=None, head="", score=None):
    """One archived run as the governed grader would record it: predictions,
    label with evaluator stamp, metrics.json, and the archive-time manifest
    line the rescore tripwire cross-checks."""
    run_dir = runs_dir / name
    (run_dir / "private").mkdir(parents=True)
    np.save(run_dir / "val_predictions.npy", val_pred)
    np.save(run_dir / "test_predictions.npy", test_pred)
    (run_dir / "private/selection.json").write_text(
        json.dumps({"status": status, "session": session, "by": "test",
                    "evaluator_id": head})
    )
    (run_dir / "private/metrics.json").write_text(
        json.dumps({"val": forged_val or {"mae": 999.0}, "test": {}})
    )
    if score is not None:
        line = {"fidelity": "full", "fraction": 1.0, "seed": 1337,
                "items": int(len(val_pred)), "total_items": int(len(val_pred)),
                "score": score, "run": name, "session": session}
        (run_dir / "manifest.txt").write_text(
            "KAPSO_EVAL_MANIFEST " + json.dumps(line) + "\n"
        )
    return run_dir


@pytest.mark.skipif(
    not (RELBENCH_CACHE / "rel-f1" / "db").exists(),
    reason="requires a populated rel-f1 relbench cache",
)
class TestRunSelectionLabels:
    """Selection-eligibility labels (user-directed 2026-07-31, after the
    user-ignore incident: final_evaluate shipped a self-disqualified leaky
    intermediate). Pool = registered finals stamped with the HEAD evaluator;
    ranking values are recomputed from stored predictions by the head
    evaluator's own --rescore, never read from metrics.json."""

    def test_final_evaluate_ignores_pending_and_voided_runs(self, tmp_path):
        handler, task = _bare_driver_position_handler(tmp_path)
        head = _governed_head(handler.work_dir)
        n_val = len(task.get_table("val"))
        n_test = len(task.get_table("test"))
        val_y = task.get_table("val", mask_input_cols=False).df[
            task.target_col].to_numpy(dtype=float)
        expected = float(np.abs(val_y - 11.0).mean())

        # Superseded intermediate with a FORGED perfect metrics.json val and
        # genuinely best predictions — must not be selectable.
        _archive_run(handler.runs_dir, "run_0001", val_y.copy(),
                     np.full(n_test, 11.0), "superseded",
                     forged_val={"mae": 0.0001}, head=head, score=0.0)
        # Registered final: mediocre but real.
        _archive_run(handler.runs_dir, "run_0002", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "final", head=head, score=expected)
        # Self-voided run with perfect predictions — excluded.
        _archive_run(handler.runs_dir, "run_0003", val_y.copy(),
                     np.full(n_test, 11.0), "self-voided", head=head, score=0.0)

        report = handler.final_evaluate()
        assert report["run"] == "run_0002"
        # Val in the report is the head evaluator's recomputation from the
        # stored predictions, not the (forged-able) metrics.json.
        assert abs(report["val_metrics"]["mae"] - expected) < 1e-9
        assert report["test_metrics"]["mae"] > 0
        assert report["head_evaluator_id"] == head
        assert set(report["scored"]) == {"run_0002"}

    def test_session_finals_are_inferred_when_the_hook_never_fired(self, tmp_path):
        """The manifest of record is printed by the maintainer-owned wrapper and
        may carry no run/session identity, so the hook can silently never fire
        (observed live: every run left 'pending' -> final_evaluate errored and a
        completed task recorded as failed). selection.json always records the
        session, so the archive derives each session's last run as its final."""
        handler, task = _bare_driver_position_handler(tmp_path)
        head = _governed_head(handler.work_dir)
        n_val = len(task.get_table("val")); n_test = len(task.get_table("test"))
        val_y = task.get_table("val", mask_input_cols=False).df[
            task.target_col].to_numpy(dtype=float)
        mae_of = lambda c: float(np.abs(val_y - c).mean())
        # session A: two runs, the later one is the final; session B: one run
        _archive_run(handler.runs_dir, "run_0001", np.full(n_val, 20.0),
                     np.full(n_test, 20.0), "pending", session="exp_A",
                     head=head, score=mae_of(20.0))
        _archive_run(handler.runs_dir, "run_0002", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "pending", session="exp_A",
                     head=head, score=mae_of(11.0))
        _archive_run(handler.runs_dir, "run_0003", np.full(n_val, 30.0),
                     np.full(n_test, 30.0), "pending", session="exp_B",
                     head=head, score=mae_of(30.0))
        report = handler.final_evaluate()
        status = lambda n: json.loads(
            (handler.runs_dir / n / "private/selection.json").read_text())["status"]
        assert status("run_0001") == "superseded"
        assert status("run_0002") == "final"
        assert status("run_0003") == "final"
        # of the two finals, argmax(val) on a minimize metric picks run_0002
        assert report["run"] == "run_0002"
        assert abs(report["val_metrics"]["mae"] - mae_of(11.0)) < 1e-9

    def test_final_under_a_superseded_evaluator_is_excluded(self, tmp_path):
        """The archive-level mirror of the in-loop None-projection rule: a
        final measured under a superseded evaluator never re-ranks against
        head runs, even with the best raw score on the board."""
        handler, task = _bare_driver_position_handler(tmp_path)
        head = _governed_head(handler.work_dir)
        n_val = len(task.get_table("val")); n_test = len(task.get_table("test"))
        val_y = task.get_table("val", mask_input_cols=False).df[
            task.target_col].to_numpy(dtype=float)
        mae_of = lambda c: float(np.abs(val_y - c).mean())
        _archive_run(handler.runs_dir, "run_0001", val_y.copy(),
                     np.full(n_test, 11.0), "final", session="exp_A",
                     head="superseded-evaluator", score=0.0)
        _archive_run(handler.runs_dir, "run_0002", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "final", session="exp_B",
                     head=head, score=mae_of(11.0))
        report = handler.final_evaluate()
        assert report["run"] == "run_0002"
        assert "run_0001" in report["excluded"]
        assert "never re-ranked across rulers" in report["excluded"]["run_0001"]

    def test_edited_archive_score_fails_loud(self, tmp_path):
        """A forged manifest score disagrees with its own recomputation —
        final selection refuses to ship rather than resolving silently."""
        handler, task = _bare_driver_position_handler(tmp_path)
        head = _governed_head(handler.work_dir)
        n_val = len(task.get_table("val")); n_test = len(task.get_table("test"))
        _archive_run(handler.runs_dir, "run_0001", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "final", head=head, score=0.0001)
        with pytest.raises(ValueError, match="does not match its recomputation"):
            handler.final_evaluate()

    def test_inference_never_overrides_a_decided_label(self, tmp_path):
        handler, task = _bare_driver_position_handler(tmp_path)
        head = _governed_head(handler.work_dir)
        n_val = len(task.get_table("val")); n_test = len(task.get_table("test"))
        _archive_run(handler.runs_dir, "run_0001", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "self-voided", session="exp_A",
                     head=head)
        _archive_run(handler.runs_dir, "run_0002", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "invalid", session="exp_A",
                     head=head)
        assert "error" in handler.final_evaluate()   # nothing eligible
        status = lambda n: json.loads(
            (handler.runs_dir / n / "private/selection.json").read_text())["status"]
        assert status("run_0001") == "self-voided" and status("run_0002") == "invalid"

    def test_zero_finals_is_the_documented_error(self, tmp_path):
        handler, task = _bare_driver_position_handler(tmp_path)
        head = _governed_head(handler.work_dir)
        n_val = len(task.get_table("val"))
        n_test = len(task.get_table("test"))
        _archive_run(handler.runs_dir, "run_0001", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "self-voided", head=head)
        assert "error" in handler.final_evaluate()

    def test_missing_label_raises(self, tmp_path):
        handler, task = _bare_driver_position_handler(tmp_path)
        run_dir = handler.runs_dir / "run_0001"
        (run_dir / "private").mkdir(parents=True)
        (run_dir / "private/metrics.json").write_text(
            json.dumps({"val": {"mae": 1.0}, "test": {}}))
        with pytest.raises(FileNotFoundError):
            handler.final_evaluate()

    def test_finalize_stamps_final_and_supersedes_session_siblings(self, tmp_path):
        handler, task = _bare_driver_position_handler(tmp_path)
        n_val = len(task.get_table("val"))
        n_test = len(task.get_table("test"))
        for name, session in (("run_0001", "exp_A"), ("run_0002", "exp_A"),
                              ("run_0003", "exp_B")):
            _archive_run(handler.runs_dir, name, np.full(n_val, 11.0),
                         np.full(n_test, 11.0), "pending", session=session)
        manifest = {"fidelity": "full", "run": "run_0002", "session": "exp_A"}
        handler.finalize_run_selection(manifest, valid=True)
        status = lambda n: json.loads(
            (handler.runs_dir / n / "private/selection.json").read_text())["status"]
        assert status("run_0002") == "final"
        assert status("run_0001") == "superseded"
        assert status("run_0003") == "pending"  # other session untouched

    def test_finalize_respects_self_voided_and_judge_veto(self, tmp_path):
        handler, task = _bare_driver_position_handler(tmp_path)
        n_val = len(task.get_table("val"))
        n_test = len(task.get_table("test"))
        _archive_run(handler.runs_dir, "run_0001", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "self-voided")
        handler.finalize_run_selection(
            {"fidelity": "full", "run": "run_0001", "session": "exp_A"}, True)
        rec = json.loads(
            (handler.runs_dir / "run_0001/private/selection.json").read_text())
        assert rec["status"] == "self-voided"  # retraction outranks promotion

        _archive_run(handler.runs_dir, "run_0002", np.full(n_val, 11.0),
                     np.full(n_test, 11.0), "pending")
        handler.finalize_run_selection(
            {"fidelity": "full", "run": "run_0002", "session": "exp_A"}, False)
        rec2 = json.loads(
            (handler.runs_dir / "run_0002/private/selection.json").read_text())
        assert rec2["status"] == "invalid"

    def test_fast_manifest_is_a_noop(self, tmp_path):
        handler, _ = _bare_driver_position_handler(tmp_path)
        handler.finalize_run_selection(
            {"fidelity": "fast", "run": "", "session": "exp_A"}, True)


class TestBooleanTargetCoercion:
    """Some relbench task tables store a boolean target as text ('t'/'f') —
    rel-trial studies-has_dmc, eligibilities-adult, eligibilities-child.
    relbench's own metrics then raise `pos_label=1 is not a valid label`, which
    killed the calibration run and failed the whole task live. Every consumer
    must see 0/1."""

    def test_text_boolean_target_becomes_int(self):
        import pandas as pd
        from types import SimpleNamespace

        from benchmarks.relbench.task_specs import coerce_boolean_target

        table = SimpleNamespace(df=pd.DataFrame({"target": ["t", "f", "T", "F"]}))
        coerce_boolean_target(table, "target")
        assert table.df["target"].tolist() == [1, 0, 1, 0]
        assert str(table.df["target"].dtype) == "int64"

    def test_numeric_target_untouched(self):
        import pandas as pd
        from types import SimpleNamespace

        from benchmarks.relbench.task_specs import coerce_boolean_target

        table = SimpleNamespace(df=pd.DataFrame({"target": [0.0, 1.0]}))
        coerce_boolean_target(table, "target")
        assert table.df["target"].tolist() == [0.0, 1.0]

    def test_unexpected_strings_raise(self):
        import pandas as pd
        from types import SimpleNamespace

        from benchmarks.relbench.task_specs import coerce_boolean_target

        table = SimpleNamespace(df=pd.DataFrame({"target": ["t", "maybe"]}))
        with pytest.raises(ValueError, match="non-boolean strings"):
            coerce_boolean_target(table, "target")

    def test_flat_cache_skips_coercion_for_recommendation_tasks(self):
        """A RecommendationTask has no scalar `target_col` — link prediction
        carries src/dst entity columns instead. Reaching for `task.target_col`
        while writing the flat cache raised AttributeError and failed every
        recommendation task at sandbox-build time (observed live on
        rel-ratebeer/user-place-liked). Coercion must be gated on the same
        `src_entity_col` check the rest of the builder dispatches on."""
        import inspect

        from benchmarks.relbench import sandbox

        src = inspect.getsource(sandbox.build_sanitized_cache)
        coerce_at = src.index("coerce_boolean_target(table")
        guard_at = src.index('if not hasattr(task, "src_entity_col")')
        assert guard_at < coerce_at, (
            "coerce_boolean_target must sit behind the RecommendationTask guard"
        )


class TestGraderSelectionLabel:
    def test_void_run_stamps_and_rejects_cross_session(self, tmp_path, monkeypatch):
        import importlib.util
        import sys as _sys
        spec = importlib.util.spec_from_file_location(
            "relbench_grader",
            Path("benchmarks/relbench/data/generic_eval/grader.py"),
        )
        grader = importlib.util.module_from_spec(spec)
        # Never drop a __pycache__ into the eval suite dir: the grader
        # snapshot/copy tests iterate that directory's entries.
        monkeypatch.setattr(_sys, "dont_write_bytecode", True)
        spec.loader.exec_module(grader)

        monkeypatch.setenv("RELBENCH_WORK_DIR", str(tmp_path))
        own_session = grader._session_id()
        sel = tmp_path / "runs" / "run_0001" / "private"
        sel.mkdir(parents=True)
        (sel / "selection.json").write_text(json.dumps(
            {"status": "pending", "session": own_session, "by": "grader"}))
        grader.void_run("run_0001", "seed-day leakage")
        rec = json.loads((sel / "selection.json").read_text())
        assert rec["status"] == "self-voided"
        assert rec["reason"] == "seed-day leakage"

        (sel / "selection.json").write_text(json.dumps(
            {"status": "pending", "session": "someone-else", "by": "grader"}))
        # Refusals come from the vendored archive contract and raise loudly
        # (no exit-code translation layer): cross-session retraction and an
        # empty reason both fail the CLI with the cause in the traceback.
        with pytest.raises(PermissionError, match="belongs to session"):
            grader.void_run("run_0001", "not mine")
        with pytest.raises(ValueError, match="non-empty"):
            grader.void_run("run_0001", "   ")

    def test_strategy_hook_is_wired(self):
        import inspect

        from kapso.execution.search_strategies.generic import strategy as strat
        src = inspect.getsource(strat.GenericSearch)
        assert "_manifest_of_record" in src
        assert "finalize_run_selection" in src

        from kapso.execution.evaluation_maintainer.maintainer import parse_manifest_line
        line = ('KAPSO_EVAL_MANIFEST {"fidelity": "full", "fraction": 1.0, '
                '"seed": 1, "items": 5, "total_items": 5, "score": 0.5, '
                '"run": "run_0007", "session": "generic_exp_3"}')
        parsed = parse_manifest_line(line)
        assert parsed["run"] == "run_0007"
        assert parsed["session"] == "generic_exp_3"


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

        assert "highest-value direction" in FEATURE_ENGINEERING_NOTE
        assert "never finished" in FEATURE_ENGINEERING_NOTE
        assert "freezing the feature matrix" in FEATURE_ENGINEERING_NOTE
        assert "guidance, not constraints" in FEATURE_ENGINEERING_NOTE


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

    def test_table_information_seed_is_agent_built_template(self):
        """User-directed (2026-07-30): the living doc starts near-empty — no
        deterministic schema/join dump (the schema already lives in the
        problem context); the seed instructs agents to build the knowledge
        base themselves."""
        from benchmarks.relbench.context import build_table_information

        doc = build_table_information("rel-x")
        assert "LIVING DOCUMENT" in doc
        assert "empty by design" in doc
        assert "multi-hop join paths" in doc
        assert "measured reasons" in doc
        assert "table `" not in doc  # no pre-generated schema rows

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

    def test_context_carries_living_docs_and_suggestions(self):
        """User-directed (2026-07-30): FE guidance is suggestion-strength
        (all-tables + features-over-architecture as leans, not hard rules),
        no best-practices or published-SOTA sections, and the
        living-documents contract stays in the problem context constants."""
        import benchmarks.relbench.context as ctx
        from benchmarks.relbench.context import (
            FEATURE_ENGINEERING_NOTE,
            FEATURES_HISTORY_TEMPLATE,
            LIVING_DOCUMENTS_NOTE,
        )

        # OOS-val contract (user-directed 2026-07-30): generic rule with the
        # explicit train+validation allowance for the test chain.
        from benchmarks.relbench.context import _prediction_contract
        import inspect
        src = inspect.getsource(_prediction_contract)
        for phrase in ("OUT-OF-SAMPLE", "stacking meta-learners",
                       "IS allowed", "pre-refit model"):
            assert phrase in src, phrase

        # Validation-realism note (user-directed 2026-07-30, after the
        # user-attendance/user-ignore selection failures): generic
        # predictive-modelling rule — no task-specific wording.
        for phrase in ("ONE finite sample", "never use the official validation score",
                       "mirror how test differs from train", "near-identical variants"):
            assert phrase in src, phrase
        for banned in ("Thanksgiving", "tick", "holiday"):
            assert banned not in src, f"task-specific wording leaked: {banned}"

        assert "Consider ALL tables" in FEATURE_ENGINEERING_NOTE

class TestOOSValAuditHooks:
    def test_advisory_verify_flags_do_not_dirty_the_run(self, tmp_path):
        """The OOS review hooks must surface train+val mixing (the exact shape
        of the observed in-sample-val incident) WITHOUT marking the run dirty —
        mixing is legal for the test chain; and a direct fit on val labels is
        also surfaced."""
        from types import SimpleNamespace

        from benchmarks.relbench.handler import RelBenchHandler

        (tmp_path / "model.py").write_text(
            "train_val_idx = np.concatenate([self.train_idx, self.val_idx])\n"
            "clf.fit(X[train_val_idx], y[train_val_idx])\n"
            "calib.fit(scores, val_labels)\n"
        )
        stub = SimpleNamespace(spec=SimpleNamespace(is_autocomplete=False))
        stub._AUDIT_PATTERNS = RelBenchHandler._AUDIT_PATTERNS
        audit = RelBenchHandler._audit_code(
            SimpleNamespace(_audit_patterns=RelBenchHandler._AUDIT_PATTERNS), tmp_path
        )
        concerns = [f["concern"] for f in audit["findings"]]
        assert any(c.startswith("verify: train+val mixing") for c in concerns)
        assert any(c.startswith("verify: possible fit on validation labels") for c in concerns)
        assert audit["clean"] is True  # advisory only — no violation claimed
