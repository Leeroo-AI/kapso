# The codify run — evolve minus ideation (CD§2/§3).
#
# The card IS the idea: one spec, one lane, implement -> evaluate -> judge ->
# feedback -> iterate, bounded by config. The frame stages the workspace
# (card as spec, cited implementations as materials, fixture inputs, the
# reproduction gates), an implementor session adapts — never authors — the
# code, the registered evaluation runs on the placement target, and the
# feedback judge evaluates THE CLAIMS (reproduction, faithfulness,
# preconditions honesty, ledger consistency). PASS = mechanical gates green
# AND judge endorsement. The representation flip is NOT done here — the
# driver writes verdict.yaml and the update transaction holding a green
# verdict commits the flip (the validator enforces it).

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.codify_gates import (
    actually_invoked_findings,
    reproduction_findings,
    weak_assertion_findings,
)
from kapso.learning.trajectory_store import TrajectoryStore

IMPLEMENTOR_PROMPT = """You implement ONE knowledge card's procedure as
runnable code. The card is the spec — implement exactly the method it
states, by ADAPTING the cited implementations in materials/; parameterize
bindings (dataset names, paths, thresholds — observed values as defaults);
do not invent an alternative method even if you believe it better —
fidelity is the acceptance criterion.

Workspace layout (you are in it):
- card.md        the spec: fact, method body, declared preconditions
- materials/     the cited archived implementations — adapt, don't author
- inputs/        the fixture run's inputs (read-only)
- gates.yaml     the reproduction gates your evaluation must assert
- replay/eval.py YOU write this: the registered evaluation — it runs your
  code on inputs/, asserts the gates' recorded values (decision outcomes
  exactly; numeric within the stated band; artifacts produced), and writes
  outcome.yaml at the workspace root: {{decisions: ..., metrics: ...}}.

Write code/ + an entrypoint, write replay/eval.py, run it
(`python replay/eval.py`), and iterate until it is green. Your final
message: one line — green or what failed.
{feedback_section}"""

JUDGE_PROMPT = """You judge one codify run's CLAIMS — you fix nothing.
The card (the spec): {card_path}
The workspace after the run: {workspace}
The reproduction gates: {gates_path}
The evaluation outcome: {outcome_path}
Mechanical findings (already computed by the frame): {mechanical}

Answer the four claims questions, each with a finding:
1. REPRODUCTION — does the outcome actually reproduce the recorded results
   (read outcome.yaml against gates.yaml yourself)?
2. FAITHFULNESS — does code/ implement the card's stated method, or does it
   reach the numbers by a shortcut that is not the method?
3. PRECONDITIONS HONESTY — is what the code actually needs (data shapes,
   hardware, environment) covered by the card's declared preconditions?
4. LEDGER CONSISTENCY — is the effect consistent with what the card's
   evidence ledger claims?

Write EXACTLY ONE file — {verdict_path} — YAML:

    endorse: true | false
    findings:
      reproduction: <finding>
      faithfulness: <finding>
      preconditions: <finding>
      ledger: <finding>
    feedback: >-
      <actionable feedback for the next iteration when not endorsing>

Your final message: one line — endorse true/false and why.
"""


class CodifyRunDriver:
    """One codify run: stage -> [implement -> evaluate -> judge]* -> verdict."""

    def __init__(
        self,
        store: TrajectoryStore,
        config: Dict[str, Any],
        agent_factory=CodingAgentFactory,
        executor=None,
    ):
        self.store = store
        self.config = config
        self.codify_config = config["learning"]["codify"]
        self.agent_factory = agent_factory
        # The evaluation executor: run the registered evaluation in the
        # staged workspace on the placement target. Injected for tests;
        # resolved from config target otherwise (local | gcp_ephemeral).
        self.executor = executor or self._resolve_executor()

    def _resolve_executor(self):
        target = self.codify_config["target"]
        if target == "local":
            return LocalExecutor(self.codify_config)
        if target == "gcp_ephemeral":
            from kapso.learning.gcp_ephemeral import GcpEphemeralExecutor
            return GcpEphemeralExecutor(self.config)
        raise ValueError(f"unknown codify target {target!r}")

    # ------------------------------------------------------------------ run

    def run(
        self, request: Dict[str, Any], card_text: str, run_dir: str
    ) -> Dict[str, Any]:
        """Execute one codify request. request: {card, fixture: {trajectory,
        inputs: [refs]}, materials: [refs into the fixture trajectory],
        gates: {decisions, metrics, artifacts}}. Returns the verdict dict
        (also written to <run_dir>/verdict.yaml)."""
        run_root = Path(run_dir).expanduser()
        workspace = run_root / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "card.md").write_text(card_text)
        gates = request["gates"]
        with open(workspace / "gates.yaml", "w") as handle:
            yaml.safe_dump(gates, handle, sort_keys=False)

        staged = self._stage(request, workspace)
        leak_findings = actually_invoked_findings(staged, gates)
        if leak_findings:
            raise ValueError(
                "codify staging leaked fixture outputs: " + "; ".join(leak_findings)
            )
        (run_root / "staged-inventory.txt").write_text("\n".join(staged) + "\n")

        # Machine class from the card's declared preconditions (CD§3): a
        # gpu-bearing precondition selects the campaign-class machine.
        front = yaml.safe_load(card_text.split("---")[1]) or {}
        gpu = "gpu" in str(front.get("preconditions") or "").lower()

        max_iterations = self.codify_config["max_iterations"]
        feedback = ""
        verdict: Dict[str, Any] = {"status": "failed", "iterations": 0}
        for iteration in range(1, max_iterations + 1):
            self._implementor_session(workspace, feedback)
            outcome_path = self.executor.run_evaluation(workspace, gpu=gpu)
            outcome = yaml.safe_load(Path(outcome_path).read_text()) or {}
            mechanical = reproduction_findings(
                gates, outcome, str(workspace),
                self.codify_config["tolerance_z"],
            )
            replay_eval = workspace / "replay" / "eval.py"
            if replay_eval.is_file():
                mechanical += weak_assertion_findings(
                    replay_eval.read_text(), gates
                )
            else:
                mechanical.append("replay/eval.py was never written")
            judge = self._judge_session(
                run_root, workspace, mechanical, iteration
            )
            endorsed = bool(judge.get("endorse")) and not mechanical
            verdict = {
                "status": "green" if endorsed else "failed",
                "iterations": iteration,
                "mechanical_findings": mechanical,
                "judge": judge,
            }
            if endorsed:
                break
            feedback = str(judge.get("feedback") or "") + (
                ("\nMechanical findings:\n" + "\n".join(mechanical))
                if mechanical else ""
            )
        with open(run_root / "verdict.yaml", "w") as handle:
            yaml.safe_dump(verdict, handle, sort_keys=False)
        return verdict

    # -------------------------------------------------------------- staging

    def _stage(self, request: Dict[str, Any], workspace: Path) -> List[str]:
        """Materials + fixture INPUTS from the store into the workspace;
        returns the staged inventory (workspace-relative paths)."""
        trajectory = request["fixture"]["trajectory"]
        bundle = self.store.resolve(trajectory)
        staged: List[str] = []
        materials_dir = workspace / "materials"
        materials_dir.mkdir()
        for ref in request.get("materials", []):
            source = bundle / str(ref).partition("#")[0]
            if not source.is_file():
                raise FileNotFoundError(
                    f"material {ref} does not resolve in {trajectory}"
                )
            target = materials_dir / source.name
            shutil.copy2(source, target)
            staged.append(str(target.relative_to(workspace)))
        inputs_dir = workspace / "inputs"
        inputs_dir.mkdir()
        for ref in request["fixture"].get("inputs", []):
            source = bundle / str(ref).partition("#")[0]
            if not source.is_file():
                raise FileNotFoundError(
                    f"fixture input {ref} does not resolve in {trajectory}"
                )
            target = inputs_dir / source.name
            shutil.copy2(source, target)
            staged.append(str(target.relative_to(workspace)))
        return staged

    # ------------------------------------------------------------- sessions

    def _implementor_session(self, workspace: Path, feedback: str) -> None:
        spec = self.codify_config["implementor"]
        agent_specific: Dict[str, Any] = {"effort": spec["effort"]}
        if spec["cli"] == "codex":
            agent_specific["sandbox"] = "workspace-write"
        else:
            agent_specific["auth_mode"] = spec["auth_mode"]
        agent = self.agent_factory.create(CodingAgentConfig(
            agent_type=spec["cli"], model=spec["model"],
            debug_model=spec["model"], agent_specific=agent_specific,
        ))
        agent.initialize(str(workspace))
        feedback_section = (
            "\nPrevious iteration's judge feedback — address every point:\n"
            + feedback if feedback else ""
        )
        agent.generate_code(
            IMPLEMENTOR_PROMPT.format(feedback_section=feedback_section),
            timeout_seconds=self.codify_config["iteration_timeout_minutes"] * 60,
        )

    def _judge_session(
        self, run_root: Path, workspace: Path, mechanical: List[str],
        iteration: int,
    ) -> Dict[str, Any]:
        spec = self.codify_config["judge"]
        agent_specific: Dict[str, Any] = {"effort": spec["effort"]}
        if spec["cli"] == "codex":
            agent_specific["sandbox"] = "workspace-write"
        else:
            agent_specific["auth_mode"] = spec["auth_mode"]
        agent = self.agent_factory.create(CodingAgentConfig(
            agent_type=spec["cli"], model=spec["model"],
            debug_model=spec["model"], agent_specific=agent_specific,
        ))
        agent.initialize(str(run_root))
        verdict_path = run_root / f"judge-{iteration}.yaml"
        agent.generate_code(
            JUDGE_PROMPT.format(
                card_path=workspace / "card.md",
                workspace=workspace,
                gates_path=workspace / "gates.yaml",
                outcome_path=workspace / "outcome.yaml",
                mechanical="; ".join(mechanical) or "none — mechanically green",
                verdict_path=verdict_path,
            ),
            timeout_seconds=self.codify_config["iteration_timeout_minutes"] * 60,
        )
        if not verdict_path.is_file():
            raise FileNotFoundError(
                f"codify judge produced no verdict at {verdict_path}"
            )
        judge = yaml.safe_load(verdict_path.read_text())
        if not isinstance(judge, dict) or "endorse" not in judge:
            raise ValueError("codify judge verdict must carry `endorse`")
        return judge


class LocalExecutor:
    """Run the registered evaluation in the sandboxed workspace on this box
    — harness tests and CPU-cheap gates only (`target: local`)."""

    def __init__(self, codify_config: Dict[str, Any]):
        self.timeout = codify_config["iteration_timeout_minutes"] * 60

    def run_evaluation(self, workspace: Path, gpu: bool = False) -> Path:
        eval_path = workspace / "replay" / "eval.py"
        if not eval_path.is_file():
            # The implementor never wrote the evaluation; an empty outcome
            # lets the gates name every miss instead of crashing the loop.
            (workspace / "outcome.yaml").write_text("{}\n")
            return workspace / "outcome.yaml"
        result = subprocess.run(
            ["python", str(eval_path)], cwd=str(workspace),
            capture_output=True, text=True, timeout=self.timeout,
        )
        (workspace / "replay-stdout.txt").write_text(
            result.stdout + result.stderr
        )
        outcome_path = workspace / "outcome.yaml"
        if not outcome_path.is_file():
            outcome_path.write_text("{}\n")
        return outcome_path
