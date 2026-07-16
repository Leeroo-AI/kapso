"""RelBench problem handler for Kapso.

Implements the benchmark handler contract consumed by `benchmark_tree_search`:
- `get_problem_context()` — static, information-dense task briefing.
- `run(file_path, run_data_dir, solution=...)` — executes the candidate's
  `main.py` (debug then full) against a sanitized read-only RelBench cache,
  validates the prediction files, scores the VALIDATION split (this drives the
  search), computes TEST metrics privately, and archives everything.
- `final_evaluate()` — picks the best-by-validation archived run, replays the
  anti-leakage audit, and emits a leaderboard-ready report.
- `stop_condition()` — optional early stop on a validation target.

Protocol stance (matches the RelBench evaluation protocol):
- model selection happens exclusively on validation metrics;
- test labels never enter the search loop, the candidate's process, or the
  problem context;
- the candidate's process reads data only from a sanitized cache with test
  information physically removed (see sandbox.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from kapso.environment.handlers.base import ProblemHandler, ProblemRunResult

from benchmarks.relbench.context import build_problem_context
from benchmarks.relbench.task_specs import TaskSpec, resolve_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_DEVICE = os.getenv("CUDA_DEVICE", "0")

MAX_OUTPUT_LINES = 400
MAX_STREAM_LINES = 25000


class PredictionContractError(Exception):
    """Candidate produced predictions that violate the contract."""


class RelBenchHandler(ProblemHandler):
    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        work_root: str = "tmp/relbench",
        planned_iterations: int = 20,
        target_val_score: Optional[float] = None,
        sota_file: Optional[str] = None,
        extra_knowledge_file: Optional[str] = None,
        rebuild_sanitized_cache: bool = False,
    ):
        super().__init__(additional_context="")
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.problem_id = f"{dataset_name}--{task_name}"
        self.planned_iterations = planned_iterations
        self.target_val_score = target_val_score

        self.work_dir = Path(work_root).absolute() / self.problem_id
        self.runs_dir = self.work_dir / "runs"
        self.shared_cache_dir = self.work_dir / "shared_cache"
        self.sanitized_cache_dir = self.work_dir / "sanitized_cache"
        for d in (self.runs_dir, self.shared_cache_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Load task through the pristine cache (handler-private).
        # ------------------------------------------------------------------
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task

        print(f"[RelBenchHandler] loading {dataset_name}/{task_name} (downloads on first use)...")
        self.dataset = get_dataset(dataset_name, download=True)
        self.task = get_task(dataset_name, task_name, download=True)
        self.spec: TaskSpec = resolve_spec(self.task, dataset_name, task_name)
        self.maximize_scoring = self.spec.maximize

        self._train_table = self.task.get_table("train", mask_input_cols=False)
        self._val_table = self.task.get_table("val", mask_input_cols=False)
        self._test_table = self.task.get_table("test", mask_input_cols=False)
        self.n_val = len(self._val_table)
        self.n_test = len(self._test_table)

        # Timeouts (env overrides for operators).
        self.full_timeout = int(os.getenv("RELBENCH_FULL_TIMEOUT", self.spec.full_timeout))
        self.debug_timeout = int(os.getenv("RELBENCH_DEBUG_TIMEOUT", self.spec.debug_timeout))
        self.spec.full_timeout = self.full_timeout
        self.spec.debug_timeout = self.debug_timeout

        # ------------------------------------------------------------------
        # Sanitized cache for the candidate's process.
        # ------------------------------------------------------------------
        self._ensure_sanitized_cache(rebuild=rebuild_sanitized_cache)

        # ------------------------------------------------------------------
        # Problem context.
        # ------------------------------------------------------------------
        sota_note = self._load_sota_note(sota_file)
        extra_knowledge = ""
        if extra_knowledge_file and os.path.exists(extra_knowledge_file):
            extra_knowledge = Path(extra_knowledge_file).read_text()

        self.problem_context = build_problem_context(
            task=self.task,
            dataset=self.dataset,
            spec=self.spec,
            db=self.dataset.get_db(),
            train_df=self._train_table.df,
            val_df=self._val_table.df,
            n_test=self.n_test,
            has_gpu=self._detect_gpu(),
            num_cpus=os.cpu_count() or 8,
            mem_gb=self._detect_mem_gb(),
            sota_note=sota_note,
            extra_knowledge=extra_knowledge,
        )

        # Harden the whole process tree: coding agents (e.g. claude_code) can
        # execute code during development sessions, and those subprocesses
        # inherit this environment — point them at the sanitized cache too.
        # The handler's own evaluation is unaffected: its task tables and db
        # were loaded from the pristine cache above and are lru-cached in
        # memory.
        os.environ["RELBENCH_CACHE_DIR"] = str(self.sanitized_cache_dir)
        # The generic-search provided grader (data/generic_eval/grader.py)
        # runs inside coding-agent sessions and reads its configuration from
        # the environment — export the full contract at startup.
        os.environ.update(
            {
                "RELBENCH_DATASET": self.dataset_name,
                "RELBENCH_TASK": self.task_name,
                "RELBENCH_PRIMARY_METRIC": self.spec.primary_metric,
                "RELBENCH_WORK_DIR": str(self.work_dir),
                "RELBENCH_FULL_TIMEOUT": str(self.full_timeout),
                "RELBENCH_DEBUG_TIMEOUT": str(self.debug_timeout),
                "KAPSO_SHARED_CACHE_DIR": str(self.shared_cache_dir),
            }
        )
        # Coding-agent sessions run `python` from PATH; make sure that
        # resolves to this interpreter's env (which has relbench + the
        # modeling stack) rather than whatever the login profile puts first.
        interpreter_bin = str(Path(sys.executable).parent)
        if not os.environ.get("PATH", "").startswith(interpreter_bin):
            os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")

        # Run bookkeeping.
        self._exec_lock = threading.Lock()
        self._run_counter = self._existing_run_count()
        self._best_val: Optional[float] = self._best_archived_val()
        self._target_reached = False

        print(f"[RelBenchHandler] ready: {self.spec.family}, primary={self.spec.primary_metric} "
              f"({'max' if self.spec.maximize else 'min'}), n_val={self.n_val}, n_test={self.n_test}")

    # ======================================================================
    # Handler contract
    # ======================================================================

    def get_problem_context(self, budget_progress: float = 0, **kwargs) -> str:
        return self.problem_context

    def run(self, file_path, run_data_dir, solution="", debug=False, *args, **kwargs) -> ProblemRunResult:
        with self._exec_lock:
            return self._run_locked(file_path, run_data_dir, solution)

    def stop_condition(self) -> bool:
        return self._target_reached

    def final_evaluate(self, file_path: str = "", **kwargs) -> Dict:
        """Best-by-validation archived run -> leaderboard-ready report."""
        best = None
        for run_dir in sorted(self.runs_dir.glob("run_*")):
            mfile = run_dir / "private" / "metrics.json"
            if not mfile.exists():
                continue
            metrics = json.loads(mfile.read_text())
            val_primary = metrics.get("val", {}).get(self.spec.primary_metric)
            if val_primary is None:
                continue
            if best is None or self._is_better(val_primary, best[0]):
                best = (val_primary, run_dir, metrics)

        if best is None:
            return {"error": "no successful runs archived"}

        val_primary, run_dir, metrics = best
        if not metrics.get("test"):
            # Generic-search archives carry val only (the in-loop grader runs
            # against the sanitized cache and cannot score test). Compute the
            # selected run's test metrics here, exactly once, from its
            # archived predictions.
            test_pred = np.load(run_dir / "test_predictions.npy", allow_pickle=False)
            metrics["test"] = {
                k: float(v) for k, v in self.task.evaluate(test_pred).items()
            }
            (run_dir / "private" / "metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )
        audit = self._audit_code(run_dir / "code")
        report = {
            "dataset": self.dataset_name,
            "task": self.task_name,
            "family": self.spec.family,
            "primary_metric": self.spec.primary_metric,
            "selected_by": f"best validation {self.spec.primary_metric}",
            "run": run_dir.name,
            "val_metrics": metrics.get("val", {}),
            "test_metrics": metrics.get("test", {}),
            "audit": audit,
            "code_path": str(run_dir / "code"),
            "predictions": {
                "val": str(run_dir / "val_predictions.npy"),
                "test": str(run_dir / "test_predictions.npy"),
            },
        }
        out = self.work_dir / "final_report.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[RelBenchHandler] final report -> {out}")
        return report

    # ======================================================================
    # Execution
    # ======================================================================

    def _run_locked(self, file_path, run_data_dir, solution: str) -> ProblemRunResult:
        self._run_counter += 1
        run_index = self._run_counter
        # The strategy may pass a workspace-relative path while the candidate
        # process runs from its own directory — absolutize before it goes into
        # the child env.
        run_data_dir = os.path.abspath(str(run_data_dir))
        file_path = os.path.abspath(str(file_path))
        os.makedirs(run_data_dir, exist_ok=True)
        self._wipe_predictions(run_data_dir)

        env = self._child_env(run_data_dir)
        main_py = Path(file_path) / "main.py"
        if not main_py.exists():
            return self._error_result(
                f"main.py not found in candidate directory {file_path}. The solution must "
                "be implemented as main.py (plus helper modules) at the repository root.",
                run_index,
            )

        # ---- debug run -----------------------------------------------------
        print("#" * 100 + f"\n[RelBenchHandler] run {run_index}: debug mode")
        had_error, out_dbg, elapsed = self._run_command(
            file_path, [sys.executable, "main.py", "--debug"], self.debug_timeout, env
        )
        if elapsed + 1 >= self.debug_timeout:
            return self._error_result(
                f"Debug mode exceeded {self.debug_timeout // 60} minutes and was killed. Debug "
                "mode must aggressively subsample so the whole pipeline finishes fast.\n"
                + tail(out_dbg),
                run_index,
            )
        if had_error:
            return self._error_result("Debug run failed.\n" + tail(out_dbg), run_index)
        try:
            self._load_predictions(run_data_dir, "val")
            self._load_predictions(run_data_dir, "test")
        except PredictionContractError as e:
            return self._error_result(f"Debug run prediction contract violation: {e}", run_index)

        # ---- full run ------------------------------------------------------
        print("#" * 100 + f"\n[RelBenchHandler] run {run_index}: full mode")
        self._wipe_predictions(run_data_dir)
        had_error, out_full, elapsed = self._run_command(
            file_path, [sys.executable, "main.py"], self.full_timeout, env
        )
        if had_error:
            msg = "Full run failed"
            if elapsed >= self.full_timeout:
                msg = (
                    f"Full run exceeded the {self.full_timeout / 3600:.1f}h limit and was "
                    "killed. Make the solution time-aware: measure and print phase timings, "
                    "reduce epochs/fanouts, cache expensive artifacts in KAPSO_SHARED_CACHE_DIR"
                )
            return self._error_result(msg + ".\n" + tail(out_full), run_index)

        try:
            val_pred, val_warn = self._load_predictions(run_data_dir, "val")
            test_pred, test_warn = self._load_predictions(run_data_dir, "test")
        except PredictionContractError as e:
            return self._error_result(f"Prediction contract violation: {e}", run_index)

        # ---- score ----------------------------------------------------------
        try:
            val_metrics = {k: float(v) for k, v in self.task.evaluate(val_pred, self._val_table).items()}
        except Exception as e:
            return self._error_result(f"Official validation scoring failed: {e}", run_index)
        test_metrics: Dict[str, float] = {}
        try:
            test_metrics = {k: float(v) for k, v in self.task.evaluate(test_pred).items()}
        except Exception as e:
            print(f"[RelBenchHandler] PRIVATE test scoring failed (not shown to agent): {e}")

        score = val_metrics[self.spec.primary_metric]
        if self._val_score_suspicious(score, val_metrics):
            val_warn = list(val_warn) + [
                "validation score is near-perfect — if val_predictions came from a model "
                "that saw val labels in training, the selection signal is void and the "
                "solution will collapse at final test; use the two-model pattern "
                "(val preds from a train-only model, test preds may use train+val)"
            ]
        self._archive_run(run_index, file_path, run_data_dir, solution, val_metrics, test_metrics)

        improved = self._best_val is None or self._is_better(score, self._best_val)
        if improved:
            self._best_val = score
        if self.target_val_score is not None and self._is_better(score, self.target_val_score, or_equal=True):
            self._target_reached = True

        output = self._compose_output(run_index, out_full, val_metrics, val_pred, val_warn + test_warn, improved)
        print(f"[RelBenchHandler] run {run_index}: val {self.spec.primary_metric}={score:.6f} "
              f"(best={self._best_val:.6f})")
        # `feedbacks` of the best node is appended to the next iteration's
        # ideation context by the orchestrator — keep it short and directive.
        budget_pct = min(100, round(100 * run_index / max(1, self.planned_iterations)))
        feedback = (
            f"Best run so far: validation {json.dumps(val_metrics)} "
            f"(primary {self.spec.primary_metric}={score:.6f}, "
            f"{'higher' if self.spec.maximize else 'lower'} is better). "
            f"~{budget_pct}% of the search budget is used"
            + (
                " — switch to exploiting/ensembling the best experiments."
                if budget_pct >= 75
                else " — keep exploring structurally different approaches while refining the leader."
            )
            + (" Contract warnings: " + " | ".join(val_warn + test_warn) if (val_warn + test_warn) else "")
        )
        return ProblemRunResult(
            score=score,
            output=output,
            run_had_error=False,
            error_message="",
            error_details="",
            detailed_output="",
            feedbacks=feedback,
        )

    def _compose_output(
        self,
        run_index: int,
        run_log: str,
        val_metrics: Dict[str, float],
        val_pred: np.ndarray,
        warnings: List[str],
        improved: bool,
    ) -> str:
        budget_pct = min(100, round(100 * run_index / max(1, self.planned_iterations)))
        phase = "EXPLORATION" if budget_pct < 75 else "EXPLOITATION/ENSEMBLING (>75% budget: combine best models)"
        parts = [
            clean_log(run_log),
            "\n" + "=" * 60,
            f"OFFICIAL VALIDATION METRICS (harness-computed): {json.dumps(val_metrics)}",
            f"Primary metric {self.spec.primary_metric} = {val_metrics[self.spec.primary_metric]:.6f} "
            f"({'higher' if self.spec.maximize else 'lower'} is better)"
            + (" — NEW BEST" if improved else f" — best so far {self._best_val:.6f}"),
            f"Prediction sanity: {self._pred_stats(val_pred)}",
        ]
        if warnings:
            parts.append("Contract warnings: " + " | ".join(warnings))
        parts.append(
            f"Search progress: handler run {run_index} of ~{self.planned_iterations} planned "
            f"(~{budget_pct}% budget) — phase: {phase}"
        )
        return "\n".join(parts)

    def _pred_stats(self, pred: np.ndarray) -> str:
        if self.spec.is_recommendation:
            uniq = len(np.unique(pred))
            return f"shape={pred.shape}, distinct dst ids used={uniq}"
        p = pred.astype(float)
        return (
            f"shape={pred.shape}, min={np.min(p):.4g}, mean={np.mean(p):.4g}, "
            f"max={np.max(p):.4g}, std={np.std(p):.4g}"
        )

    def _error_result(self, message: str, run_index: int) -> ProblemRunResult:
        budget_pct = min(100, round(100 * run_index / max(1, self.planned_iterations)))
        message += f"\n(Search progress: handler run {run_index}, ~{budget_pct}% budget.)"
        return ProblemRunResult(
            score=-1e18 if self.spec.maximize else 1e18,
            output="",
            run_had_error=True,
            error_message=message[:2000],
            error_details=message[:6000],
        )

    def _run_command(self, cwd, command, timeout, env) -> Tuple[bool, str, float]:
        start = time.time()

        def _child_guard():
            # Shared box: keep evaluation children polite and bounded.
            # Low CPU priority + a hard address-space ceiling well under
            # system RAM so a runaway candidate OOMs itself, not the host.
            os.nice(10)
            try:
                import resource

                limit = int(os.getenv("RELBENCH_CHILD_MEM_BYTES", 20 * 1024**3))
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except Exception:
                pass

        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
            start_new_session=True,
            preexec_fn=_child_guard,
        )
        lines: List[str] = []
        killed = False

        def _watchdog():
            try:
                process.wait(timeout)
            except subprocess.TimeoutExpired:
                nonlocal killed
                killed = True
                try:
                    os.killpg(os.getpgid(process.pid), 9)
                except Exception:
                    process.kill()

        watchdog = threading.Thread(target=_watchdog, daemon=True)
        watchdog.start()
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if len(lines) > MAX_STREAM_LINES:
                killed = True
                try:
                    os.killpg(os.getpgid(process.pid), 9)
                except Exception:
                    process.kill()
                lines.append("\n[handler] output flood (>25k lines) — process killed. "
                             "Disable progress bars/verbose logging.\n")
                break
        process.wait()
        watchdog.join(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        elapsed = time.time() - start
        if len(lines) > MAX_OUTPUT_LINES:
            lines = lines[: MAX_OUTPUT_LINES // 2] + ["\n ... [truncated] ...\n"] + lines[-MAX_OUTPUT_LINES // 2 :]
        output = "".join(lines)
        had_error = killed or process.returncode != 0
        return had_error, output, elapsed

    def _child_env(self, run_data_dir: str) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "RELBENCH_CACHE_DIR": str(self.sanitized_cache_dir),
                "KAPSO_RUN_DATA_DIR": run_data_dir,
                "KAPSO_SHARED_CACHE_DIR": str(self.shared_cache_dir),
                "RELBENCH_DATASET": self.dataset_name,
                "RELBENCH_TASK": self.task_name,
                "CUDA_DEVICE": CUDA_DEVICE,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HOME": str(self.shared_cache_dir / "hf"),
            }
        )
        return env

    # ======================================================================
    # Prediction loading / validation
    # ======================================================================

    def _wipe_predictions(self, run_data_dir: str) -> None:
        for split in ("val", "test"):
            p = Path(run_data_dir) / f"{split}_predictions.npy"
            if p.exists():
                p.unlink()

    def _load_predictions(self, run_data_dir: str, split: str) -> Tuple[np.ndarray, List[str]]:
        n_rows = self.n_val if split == "val" else self.n_test
        path = Path(run_data_dir) / f"{split}_predictions.npy"
        if not path.exists():
            raise PredictionContractError(
                f"{path.name} was not written to KAPSO_RUN_DATA_DIR. Every run (debug and "
                "full) must write BOTH val_predictions.npy and test_predictions.npy."
            )
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            raise PredictionContractError(f"could not np.load {path.name}: {e}")

        expected = self.spec.expected_pred_shape(n_rows)
        warnings: List[str] = []

        if self.spec.is_recommendation:
            if arr.ndim != 2 or arr.shape != expected:
                raise PredictionContractError(
                    f"{path.name} shape {arr.shape} != required {expected} "
                    f"(rows aligned with task.get_table('{split}'), K={self.spec.eval_k})."
                )
            if not np.issubdtype(arr.dtype, np.integer):
                if np.issubdtype(arr.dtype, np.floating) and np.all(np.isfinite(arr)) and np.all(arr == np.round(arr)):
                    arr = arr.astype(np.int64)
                else:
                    raise PredictionContractError(
                        f"{path.name} must contain integer destination ids (got dtype {arr.dtype})."
                    )
            oob = int(((arr < 0) | (arr >= self.spec.num_dst_nodes)).sum())
            if oob:
                warnings.append(
                    f"{split}: {oob} destination ids out of [0,{self.spec.num_dst_nodes}) — they can never match"
                )
            dup_rows = int(sum(len(np.unique(row)) < len(row) for row in arr))
            if dup_rows:
                warnings.append(
                    f"{split}: {dup_rows} rows contain duplicate ids — duplicates waste ranking slots"
                )
            return arr, warnings

        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1 and not self.spec.is_multiclass:
            arr = arr[:, 0]
        if arr.shape != expected:
            raise PredictionContractError(
                f"{path.name} shape {arr.shape} != required {expected} "
                f"(rows aligned with task.get_table('{split}'))."
            )
        arr = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(arr)):
            raise PredictionContractError(f"{path.name} contains NaN/inf values.")
        if self.spec.family.endswith("binary_classification"):
            if arr.min() < -1e-6 or arr.max() > 1 + 1e-6:
                raise PredictionContractError(
                    f"{path.name}: binary predictions must be probabilities in [0,1] "
                    f"(got range [{arr.min():.4g}, {arr.max():.4g}]); apply a sigmoid."
                )
            arr = np.clip(arr, 0.0, 1.0)
        return arr, warnings

    # ======================================================================
    # Archiving / reporting
    # ======================================================================

    def _archive_run(
        self,
        run_index: int,
        file_path,
        run_data_dir: str,
        solution: str,
        val_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ) -> None:
        run_dir = self.runs_dir / f"run_{run_index:04d}"
        private = run_dir / "private"
        code_dir = run_dir / "code"
        private.mkdir(parents=True, exist_ok=True)
        code_dir.mkdir(parents=True, exist_ok=True)

        for split in ("val", "test"):
            src = Path(run_data_dir) / f"{split}_predictions.npy"
            if src.exists():
                shutil.copy2(src, run_dir / src.name)

        (run_dir / "solution.md").write_text(solution or "")
        (private / "metrics.json").write_text(
            json.dumps({"val": val_metrics, "test": test_metrics,
                        "solution_sha": hashlib.sha256((solution or "").encode()).hexdigest()[:16]},
                       indent=2)
        )
        # Snapshot candidate code (small text files only).
        src_root = Path(file_path)
        for f in src_root.rglob("*"):
            if not f.is_file() or ".git" in f.parts or "kapso_evaluation" in f.parts:
                continue
            if f.stat().st_size > 2_000_000:
                continue
            rel = f.relative_to(src_root)
            dest = code_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    def _existing_run_count(self) -> int:
        runs = sorted(self.runs_dir.glob("run_*"))
        return int(runs[-1].name.split("_")[1]) if runs else 0

    def _best_archived_val(self) -> Optional[float]:
        best = None
        for mfile in self.runs_dir.glob("run_*/private/metrics.json"):
            try:
                v = json.loads(mfile.read_text())["val"].get(self.spec.primary_metric)
            except Exception:
                continue
            if v is not None and (best is None or self._is_better(v, best)):
                best = v
        return best

    def _is_better(self, a: float, b: float, or_equal: bool = False) -> bool:
        if or_equal:
            return a >= b if self.spec.maximize else a <= b
        return a > b if self.spec.maximize else a < b

    def _val_score_suspicious(self, score: float, val_metrics: Dict[str, float]) -> bool:
        """Near-perfect validation usually means the model trained on val."""
        m = self.spec.primary_metric
        if m in ("roc_auc", "accuracy", "link_prediction_map") and score >= 0.9995:
            return True
        if m == "r2" and score >= 0.9999:
            return True
        if m == "mae" and score <= 1e-9:
            return True
        return False

    # ======================================================================
    # Anti-leak audit
    # ======================================================================

    @property
    def _audit_patterns(self):
        patterns = list(self._AUDIT_PATTERNS)
        if not self.spec.is_autocomplete:
            # Legal for autocomplete (test rows exist only after the cutoff and the
            # sanitized db is blanked); for forecasting the sanitized db is truncated
            # anyway, but flag the intent.
            patterns.append(
                (r"upto_test_timestamp\s*=\s*False", "requests untruncated database")
            )
        return patterns

    _AUDIT_PATTERNS = [
        (r"mask_input_cols\s*=\s*False", "unmasks task tables (test labels)"),
        (r"removed_cols", "reads AutoComplete removed target columns"),
        (r"RELBENCH_CACHE_DIR", "tampers with the data cache location"),
        (r"\.cache[/\\]relbench", "touches the pristine relbench cache path"),
        (r"download\s*=\s*True", "attempts dataset re-download"),
        (r"relbench\.stanford\.edu|pooch", "fetches benchmark files directly"),
        (r"_get_table\s*\(", "calls private table builders"),
        (r"tmp[/\\]relbench", "hardcodes the handler work directory (private archives)"),
        (r"private[/\\]metrics\.json", "probes quarantined test metrics"),
    ]

    def _audit_code(self, code_dir: Path) -> Dict:
        # Hash the shipped starter kit so unmodified vendored files (which
        # legitimately contain e.g. `download=True` as reference material) are
        # not flagged; anything the candidate added or edited is audited.
        shipped = {}
        kit_dir = Path(__file__).parent / "data" / "starter_kit"
        for f in kit_dir.rglob("*.py"):
            shipped[hashlib.sha256(f.read_bytes()).hexdigest()] = f.name
        findings = []
        for f in sorted(code_dir.rglob("*.py")):
            try:
                if hashlib.sha256(f.read_bytes()).hexdigest() in shipped:
                    continue
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for pattern, why in self._audit_patterns:
                for m in re.finditer(pattern, text):
                    line_no = text[: m.start()].count("\n") + 1
                    findings.append({
                        "file": str(f.relative_to(code_dir)),
                        "line": line_no,
                        "pattern": pattern,
                        "concern": why or "manual review",
                        "snippet": text.splitlines()[line_no - 1][:200].strip(),
                    })
        return {
            "clean": not any(f["concern"] != "manual review" for f in findings),
            "findings": findings,
            "note": "static scan; manually review flagged lines before publishing results",
        }

    # ======================================================================
    # Setup helpers
    # ======================================================================

    def _ensure_sanitized_cache(self, rebuild: bool) -> None:
        marker = self.work_dir / "sanitized_cache.meta.json"
        want = {"dataset": self.dataset_name, "task": self.task_name, "version": 3}
        if not rebuild and marker.exists() and self.sanitized_cache_dir.exists():
            try:
                if json.loads(marker.read_text()) == want:
                    print(f"[RelBenchHandler] sanitized cache OK at {self.sanitized_cache_dir}")
                    return
            except Exception:
                pass
        print("[RelBenchHandler] building sanitized cache (fresh subprocess)...")
        env = os.environ.copy()
        env.pop("RELBENCH_CACHE_DIR", None)  # sanitizer reads the pristine cache
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.relbench.sandbox",
                "--dataset", self.dataset_name,
                "--task", self.task_name,
                "--dest", str(self.sanitized_cache_dir),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )
        marker.write_text(json.dumps(want))

    def _load_sota_note(self, sota_file: Optional[str]) -> str:
        path = Path(sota_file) if sota_file else Path(__file__).parent / "data" / "sota.json"
        if not path.exists():
            return ""
        try:
            table = json.loads(path.read_text())
        except Exception:
            return ""
        entry = table.get(f"{self.dataset_name}/{self.task_name}")
        if not entry:
            return ""
        lines = [
            f"Best published TEST {entry.get('metric', self.spec.primary_metric)}: "
            f"{entry.get('value')} ({entry.get('method', 'unknown method')})."
        ]
        if entry.get("runner_up"):
            lines.append(f"Runner-up: {entry['runner_up']}.")
        if entry.get("note"):
            lines.append(entry["note"])
        lines.append(
            "Validation and test are correlated but not identical; use the number as a bar "
            "for ambition, not as a fitting target."
        )
        return "\n".join(lines)

    @staticmethod
    def _detect_gpu() -> bool:
        return shutil.which("nvidia-smi") is not None and subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True
        ).returncode == 0

    @staticmethod
    def _detect_mem_gb() -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) // (1024 * 1024)
        except OSError:
            pass
        return 32


# ---------------------------------------------------------------------------
# Log utilities
# ---------------------------------------------------------------------------

def tail(text: str, n_chars: int = 4000) -> str:
    return text if len(text) <= n_chars else "...[truncated]...\n" + text[-n_chars:]


_NOISE = re.compile(
    r"(warn|warning|deprecat|futurewarning|userwarning|\[info\]|it/s\]|%\|)", re.IGNORECASE
)


def clean_log(output: str) -> str:
    lines = [l for l in output.split("\n") if not _NOISE.search(l)]
    return "\n".join(lines)
