"""EvaluationMaintainer: sole writer and timekeeper of evaluation.

A deterministic Python frame around scoped coding-agent calls, in the same
pattern as the FeedbackGenerator. Every trust boundary is a mechanical
post-condition in this frame — never an instruction to the agent:

- provided evaluator logic is immutable: after any agent call, the provided
  files must re-hash byte-identical or the transaction fails;
- registration invariants (manifest == tree, runnable at fast fidelity,
  manifest line printed) are executed by the frame, not asserted by prompts;
- calibration timing is measured by the frame's own subprocess run — the
  timing model cannot be hallucinated. This small executor is deliberately
  the embryo of the future Kapso-owned evaluation runner.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import git

from kapso.core.prompt_loader import load_prompt, render_prompt
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.evaluation_integrity import (
    build_data_manifest,
    build_evaluation_manifest,
    manifest_fingerprint,
)
from kapso.execution.evaluation_maintainer.registry import (
    EvaluationRegistry,
    EvaluatorVersion,
    TimingModel,
)


EVALUATION_DIR_NAME = "kapso_evaluation"
ENTRYPOINT_NAME = "kapso_eval.py"
# One greppable JSON line the entrypoint must print, e.g.:
# KAPSO_EVAL_MANIFEST {"fidelity": "fast", "fraction": 0.03, "seed": 1337,
#                      "items": 12, "total_items": 400, "score": 0.5}
MANIFEST_MARKER = "KAPSO_EVAL_MANIFEST"


class EvaluationMaintainerError(RuntimeError):
    """Raised when a maintainer transaction violates its post-conditions."""


def evaluation_command(*, fidelity: str, fraction: float, seed: int) -> str:
    """The registered invocation contract (CLI args, never env vars)."""
    return (
        f"python {EVALUATION_DIR_NAME}/{ENTRYPOINT_NAME} "
        f"--fidelity {fidelity} --fraction {fraction} --seed {seed}"
    )


def parse_manifest_line(stdout: str) -> Dict[str, Any]:
    """Parse the single KAPSO_EVAL_MANIFEST JSON line an evaluation prints."""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(MANIFEST_MARKER):
            payload = json.loads(stripped[len(MANIFEST_MARKER):].strip())
            for key in ("fidelity", "fraction", "seed", "items",
                        "total_items"):
                if key not in payload:
                    raise EvaluationMaintainerError(
                        f"Evaluation manifest line is missing {key!r}"
                    )
            if payload["items"] <= 0 or payload["total_items"] <= 0:
                raise EvaluationMaintainerError(
                    "Evaluation manifest item counts must be positive"
                )
            return payload
    raise EvaluationMaintainerError(
        f"Evaluation output contains no {MANIFEST_MARKER} line"
    )


@dataclass(frozen=True)
class EvaluationChangeRequest:
    """A request to change the evaluation, filed by another component."""

    iteration: int
    requested_by: str  # "implementation" | "feedback" | "user"
    summary: str
    evidence: str  # logs / output excerpt, passed in full


@dataclass(frozen=True)
class ChangeOutcome:
    accepted: bool
    reason: str
    new_version: Optional[EvaluatorVersion]
    requires_reanchor: bool


@dataclass(frozen=True)
class TimingEstimate:
    expected_seconds: float
    upper_seconds: float
    basis: str  # "calibration" | "measured(n=...)"


@dataclass(frozen=True)
class MaintainerTransactionTelemetry:
    """Budget telemetry for the last maintainer transaction."""

    cost_usd: float
    duration_seconds: float


class EvaluationMaintainer:
    """Owns kapso_evaluation/: setup, change governance, and timing."""

    def __init__(
        self,
        *,
        coding_agent_config: CodingAgentConfig,
        workspace_dir: str,
        fast_fraction: float,
        subsample_seed: int,
        calibration_fraction: float,
        calibration_timeout_seconds: float,
        fast_variant_threshold_seconds: float,
        overhead_factor: float,
        protected_data_paths: Optional[List[str]] = None,
        command_runner: Optional[Callable[..., Any]] = None,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.evaluation_dir = self.workspace_dir / EVALUATION_DIR_NAME
        self.registry = EvaluationRegistry(str(self.workspace_dir))
        self.coding_agent_config = coding_agent_config
        self.fast_fraction = float(fast_fraction)
        self.subsample_seed = int(subsample_seed)
        self.calibration_fraction = float(calibration_fraction)
        self.calibration_timeout_seconds = float(calibration_timeout_seconds)
        self.fast_variant_threshold_seconds = float(
            fast_variant_threshold_seconds
        )
        self.overhead_factor = float(overhead_factor)
        self.protected_data_paths = list(protected_data_paths or [])
        # Injectable for hermetic tests, like core.llm's sleep_fn seam.
        self._run_command = command_runner or subprocess.run
        self._provided_baseline: Dict[str, str] = {}
        self.last_transaction_telemetry: Optional[
            MaintainerTransactionTelemetry
        ] = None

    # =====================================================================
    # Setup transaction
    # =====================================================================

    def setup(
        self,
        *,
        goal: str,
        eval_dir: Optional[str],
        data_dir: Optional[str],
    ) -> EvaluatorVersion:
        """Verify-or-build the evaluator, time it, register v1, commit."""
        transaction_started = time.monotonic()

        provided = eval_dir is not None
        if provided:
            # The provided files were copied into kapso_evaluation/ by
            # workspace setup; their manifest is the immutability baseline.
            self._provided_baseline = build_evaluation_manifest(
                self.evaluation_dir
            )
            prompt = render_prompt(
                load_prompt(
                    "execution/evaluation_maintainer/prompts/"
                    "setup_provided.md"
                ),
                {
                    "goal": goal,
                    "entrypoint_name": ENTRYPOINT_NAME,
                    "manifest_marker": MANIFEST_MARKER,
                    "fast_fraction": str(self.fast_fraction),
                    "subsample_seed": str(self.subsample_seed),
                },
            )
        else:
            self.evaluation_dir.mkdir(parents=True, exist_ok=True)
            prompt = render_prompt(
                load_prompt(
                    "execution/evaluation_maintainer/prompts/setup_build.md"
                ),
                {
                    "goal": goal,
                    "data_dir": data_dir or "(no data directory provided)",
                    "entrypoint_name": ENTRYPOINT_NAME,
                    "manifest_marker": MANIFEST_MARKER,
                    "fast_fraction": str(self.fast_fraction),
                    "subsample_seed": str(self.subsample_seed),
                },
            )

        agent_cost = self._run_agent(prompt)
        self._enforce_provided_immutable()

        timing, calibration_manifest = self._run_calibration()
        full_estimate = timing.expected_seconds(1.0)
        fast_enabled = (
            full_estimate > self.fast_variant_threshold_seconds
        )

        version = EvaluatorVersion(
            evaluator_id=manifest_fingerprint(
                build_evaluation_manifest(self.evaluation_dir)
            ),
            version=1,
            provenance="provided" if provided else "maintainer_built",
            parent_evaluator=None,
            fidelity_support={
                "fast_enabled": fast_enabled,
                "fast_fraction": self.fast_fraction,
                "subsample_seed": self.subsample_seed,
                "total_items": calibration_manifest["total_items"],
            },
            timing=timing,
            created_at_iteration=0,
            reason="setup",
            data_manifest=build_data_manifest(
                self.workspace_dir, self.protected_data_paths
            ),
        )
        self.registry.register(version)
        self._commit_evaluation("chore(kapso): register evaluator v1")

        self.last_transaction_telemetry = MaintainerTransactionTelemetry(
            cost_usd=agent_cost,
            duration_seconds=time.monotonic() - transaction_started,
        )
        return version

    # =====================================================================
    # Change-request governance
    # =====================================================================

    def handle_change_request(
        self, request: EvaluationChangeRequest
    ) -> ChangeOutcome:
        """Triage and, if accepted, implement + register a new version."""
        transaction_started = time.monotonic()
        head = self.registry.head()
        if head is None:
            raise EvaluationMaintainerError(
                "Cannot handle a change request before setup registered v1"
            )

        prompt = render_prompt(
            load_prompt(
                "execution/evaluation_maintainer/prompts/change_request.md"
            ),
            {
                "requested_by": request.requested_by,
                "summary": request.summary,
                "evidence": request.evidence,
                "provided_logic_immutable": (
                    "true" if self._provided_baseline else "false"
                ),
                "entrypoint_name": ENTRYPOINT_NAME,
                "manifest_marker": MANIFEST_MARKER,
            },
        )
        agent_output, agent_cost = self._run_agent_with_output(prompt)
        self._enforce_provided_immutable()

        verdict_match = re.search(
            r"<change_verdict>\s*(accept|reject)\s*</change_verdict>",
            agent_output,
        )
        reason_match = re.search(
            r"<reason>(.*?)</reason>", agent_output, re.DOTALL
        )
        if verdict_match is None or reason_match is None:
            raise EvaluationMaintainerError(
                "Change-request agent did not return the required "
                "<change_verdict> and <reason> tags"
            )
        reason = reason_match.group(1).strip()

        if verdict_match.group(1) == "reject":
            self.last_transaction_telemetry = MaintainerTransactionTelemetry(
                cost_usd=agent_cost,
                duration_seconds=time.monotonic() - transaction_started,
            )
            return ChangeOutcome(
                accepted=False,
                reason=reason,
                new_version=None,
                requires_reanchor=False,
            )

        new_manifest = build_evaluation_manifest(self.evaluation_dir)
        new_id = manifest_fingerprint(new_manifest)
        if new_id == head.evaluator_id:
            raise EvaluationMaintainerError(
                "Change request was accepted but the evaluation tree is "
                "byte-identical to the registered head"
            )

        timing, calibration_manifest = self._run_calibration()
        version = EvaluatorVersion(
            evaluator_id=new_id,
            version=head.version + 1,
            provenance=head.provenance,
            parent_evaluator=head.evaluator_id,
            fidelity_support={
                **head.fidelity_support,
                "total_items": calibration_manifest["total_items"],
            },
            timing=timing,
            created_at_iteration=request.iteration,
            reason=f"CR@iteration {request.iteration}: {reason}",
            # The inputs half never changes through a CR: an accepted
            # fix edits evaluation logic, not the protected data.
            data_manifest=head.data_manifest,
        )
        self.registry.register(version)
        self._commit_evaluation(
            f"chore(kapso): register evaluator v{version.version}"
        )

        self.last_transaction_telemetry = MaintainerTransactionTelemetry(
            cost_usd=agent_cost,
            duration_seconds=time.monotonic() - transaction_started,
        )
        return ChangeOutcome(
            accepted=True,
            reason=reason,
            new_version=version,
            requires_reanchor=True,
        )

    # =====================================================================
    # Timing
    # =====================================================================

    def timing(self, fraction: float) -> TimingEstimate:
        head = self.registry.head()
        if head is None:
            raise EvaluationMaintainerError(
                "No registered evaluator; run setup first"
            )
        samples = len(head.timing.measured_samples)
        return TimingEstimate(
            expected_seconds=head.timing.expected_seconds(fraction),
            upper_seconds=head.timing.upper_seconds(
                fraction, self.overhead_factor
            ),
            basis=(
                f"measured(n={samples})" if samples else "calibration"
            ),
        )

    def record_run(self, *, fraction: float, duration_seconds: float) -> None:
        head = self.registry.head()
        if head is None:
            raise EvaluationMaintainerError(
                "No registered evaluator; run setup first"
            )
        self.registry.update_head_timing(
            head.timing.with_sample(fraction, duration_seconds)
        )

    def evaluation_command(self, *, fidelity: str, fraction: float) -> str:
        """The registered invocation contract (CLI args, never env vars)."""
        return evaluation_command(
            fidelity=fidelity, fraction=fraction, seed=self.subsample_seed
        )

    # =====================================================================
    # Frame mechanics (mechanical post-conditions)
    # =====================================================================

    def _run_agent(self, prompt: str) -> float:
        _, cost = self._run_agent_with_output(prompt)
        return cost

    def _run_agent_with_output(self, prompt: str) -> tuple:
        agent = CodingAgentFactory.create(self.coding_agent_config)
        agent.initialize(str(self.workspace_dir))
        result = agent.generate_code(prompt)
        cost = agent.get_cumulative_cost()
        agent.cleanup()
        if not result.success:
            raise EvaluationMaintainerError(
                f"Maintainer agent call failed: {result.error}"
            )
        return result.output or "", cost

    def _enforce_provided_immutable(self) -> None:
        """Post-condition, not a prompt: provided bytes never change."""
        if not self._provided_baseline:
            return
        current = build_evaluation_manifest(self.evaluation_dir)
        violations = sorted(
            path
            for path, digest in self._provided_baseline.items()
            if current.get(path) != digest
        )
        if violations:
            raise EvaluationMaintainerError(
                "Provided evaluator files were modified by a maintainer "
                "transaction: " + ", ".join(violations)
            )

    def _run_calibration(self) -> tuple:
        """Deterministic timing measurement — module-run, deadline-bounded."""
        command = shlex.split(
            self.evaluation_command(
                fidelity="fast", fraction=self.calibration_fraction
            )
        )
        started = time.monotonic()
        completed = self._run_command(
            command,
            cwd=str(self.workspace_dir),
            capture_output=True,
            text=True,
            timeout=self.calibration_timeout_seconds,
        )
        duration = time.monotonic() - started
        if completed.returncode != 0:
            raise EvaluationMaintainerError(
                "Calibration run failed "
                f"(exit {completed.returncode}): {completed.stderr}"
            )
        manifest = parse_manifest_line(completed.stdout)
        items = manifest["items"]
        return (
            TimingModel(
                per_item_seconds=duration / items,
                startup_seconds=0.0,
                total_items=manifest["total_items"],
            ),
            manifest,
        )

    def _commit_evaluation(self, message: str) -> None:
        repo = git.Repo(str(self.workspace_dir))
        repo.git.add("-f", [EVALUATION_DIR_NAME])
        repo.git.add([str(EvaluationRegistry.RELATIVE_PATH)])
        if repo.is_dirty(untracked_files=True):
            repo.git.commit("-m", message)
