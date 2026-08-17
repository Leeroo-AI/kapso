# Behavior suite — semantic production tests with agentic review.
#
# Design: docs/plans/learning/behavior-tests.md. Mechanical tests prove the
# machinery holds its invariants; this suite proves the behavior is
# semantically right, judged by a cross-model reviewer against fixtures whose
# correct outcome we authored (truth.md, human-signed at creation). Verdicts
# are gates: {verdict: PASS|FAIL, rationale} — a semantic FAIL blocks learner
# promotion exactly as a gauntlet FAIL does.
#
# A scenario directory:
#   scenario.yaml   {id, machinery: serve|grade-exam|update, description, ...}
#   truth.md        ground truth + source refs + expected outcome
#   rubric.md       the reviewer's questions
#   fixture/        scenario material (bank/, reports, task coords, ...)
#
# The runner stages a scratch copy, invokes the REAL machinery (never a mock
# of the thing under test), then launches the reviewer session over truth +
# rubric + the run's artifacts. v1 machineries: `serve` (compile_brief) and
# `grade-exam` / `update` (the real frames — real sessions; the scenarios
# that need them land with their phases' fixtures).

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.bank import Bank
from kapso.learning.graders.frame import GradingFrame
from kapso.learning.retriever import compile_brief
from kapso.learning.trajectory_store import TrajectoryStore
from kapso.learning.update_frame import UpdateFrame

MACHINERIES = ("serve", "grade-exam", "update")

REVIEWER_PROMPT = """You review one behavior scenario of a learning system —
semantic correctness, not mechanics. The fixture's truth.md states the known
correct outcome (with source refs); the rubric lists the questions; the run's
artifacts are what the real machinery actually did. You fix nothing and you
never see other scenarios.

Scenario: {scenario_id} — {description}
Truth (the known correct outcome): {truth_path}
Rubric: {rubric_path}
The run's artifacts: {artifacts_dir}

Answer every rubric question with a per-question finding, then write EXACTLY
ONE file — the verdict file is {verdict_path} — in this YAML shape:

    verdict: PASS | FAIL
    rationale: >-
      <per-question findings compressed to the decision; a FAIL names the
      offending artifact — it is a bug report for the next crew version>

Your final message: one line — the verdict.
"""


class BehaviorRunner:
    """Run behavior scenarios: real machinery + agentic review."""

    def __init__(
        self,
        store: TrajectoryStore,
        config: Dict[str, Any],
        agent_factory=CodingAgentFactory,
    ):
        self.store = store
        self.config = config
        self.behavior_config = config["learning"]["behavior"]
        self.agent_factory = agent_factory

    def run_scenario(self, scenario_dir: str, run_root: str) -> Dict[str, Any]:
        scenario_path = Path(scenario_dir).expanduser()
        scenario = yaml.safe_load((scenario_path / "scenario.yaml").read_text())
        if scenario.get("machinery") not in MACHINERIES:
            raise ValueError(
                f"scenario {scenario.get('id')!r} machinery "
                f"{scenario.get('machinery')!r} is not one of {MACHINERIES}"
            )
        for name in ("truth.md", "rubric.md"):
            if not (scenario_path / name).is_file():
                raise FileNotFoundError(f"scenario {scenario_dir} missing {name}")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = Path(run_root).expanduser() / f"{scenario['id']}-{stamp}"
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        machinery = scenario["machinery"]
        if machinery == "serve":
            self._run_serve(scenario, scenario_path, artifacts_dir)
        elif machinery == "grade-exam":
            self._run_grade_exam(scenario, scenario_path, artifacts_dir)
        else:
            self._run_update(scenario, scenario_path, artifacts_dir)

        verdict_path = run_dir / "verdict.yaml"
        prompt = REVIEWER_PROMPT.format(
            scenario_id=scenario["id"],
            description=scenario.get("description", ""),
            truth_path=scenario_path / "truth.md",
            rubric_path=scenario_path / "rubric.md",
            artifacts_dir=artifacts_dir,
            verdict_path=verdict_path,
        )
        reviewer = self.behavior_config["reviewer"]
        agent = self.agent_factory.create(CodingAgentConfig(
            agent_type=reviewer["cli"],
            model=reviewer["model"],
            debug_model=reviewer["model"],
            agent_specific={
                "effort": reviewer["effort"],
                **({"auth_mode": reviewer["auth_mode"]}
                   if reviewer["cli"] == "claude_code" else
                   {"sandbox": "workspace-write"}),
            },
        ))
        agent.initialize(str(run_dir))
        agent.generate_code(
            prompt,
            timeout_seconds=self.behavior_config["timeout_minutes"] * 60,
        )
        if not verdict_path.is_file():
            raise FileNotFoundError(
                f"reviewer produced no verdict for {scenario['id']}"
            )
        verdict = yaml.safe_load(verdict_path.read_text())
        if not isinstance(verdict, dict) or verdict.get("verdict") not in (
            "PASS", "FAIL"
        ):
            raise ValueError(
                f"scenario {scenario['id']}: verdict must be PASS or FAIL"
            )
        if not str(verdict.get("rationale", "")).strip():
            raise ValueError(
                f"scenario {scenario['id']}: verdict has no rationale — no "
                f"naked tags"
            )
        return {"scenario": scenario["id"], **verdict, "run_dir": str(run_dir)}

    # --------------------------------------------------------- machineries

    def _run_serve(
        self, scenario: Dict[str, Any], scenario_path: Path, artifacts_dir: Path
    ) -> None:
        """B8-class: the real push core over the fixture bank."""
        bank = Bank(str(scenario_path / "fixture" / "bank"))
        result = compile_brief(
            bank,
            scenario["task"],
            scenario.get("bank_head", "fixture"),
            self.config["learning"]["retriever"],
        )
        (artifacts_dir / "brief.md").write_text(result["brief"])
        with open(artifacts_dir / "serving-record.yaml", "w") as handle:
            yaml.safe_dump(result["record"], handle, sort_keys=False)

    def _run_grade_exam(
        self, scenario: Dict[str, Any], scenario_path: Path, artifacts_dir: Path
    ) -> None:
        """B7-class: the real exam over a fixture bank + a real trajectory."""
        grading = GradingFrame(self.store, self.config,
                               agent_factory=self.agent_factory)
        report_path = grading.grade_exam(
            scenario["trajectory"],
            str(scenario_path / "fixture" / "bank"),
            scenario.get("bank_head", "fixture"),
            str(artifacts_dir),
        )
        shutil.copy2(report_path, artifacts_dir / "report.md")

    def _run_update(
        self, scenario: Dict[str, Any], scenario_path: Path, artifacts_dir: Path
    ) -> None:
        """B1–B6-class: a real update run over a fixture bank home + batch."""
        fixture_config = {
            **self.config,
            "learning": {
                **self.config["learning"],
                "bank": {
                    "local_path": str(scenario_path / "fixture" / "bank-home.git"),
                    "remote": None,
                },
            },
        }
        frame = UpdateFrame(self.store, fixture_config,
                            agent_factory=self.agent_factory)
        batch = [
            {
                "trajectory": item["trajectory"],
                "hindcast_report": str(scenario_path / "fixture" / item["report"]),
            }
            for item in scenario.get("batch", [])
        ]
        run_dir = frame.run_update(
            batch, str(artifacts_dir), scenario.get("learner_version", "behave")
        )
        shutil.copy2(run_dir / "report.md", artifacts_dir / "learner-report.md")

    # ---------------------------------------------------------------- suite

    def run_all(self, run_root: str) -> Dict[str, Any]:
        """Every scenario under the configured dir; the roll-up is a gate —
        any FAIL fails the suite (gates dominate scores)."""
        scenarios_dir = Path(self.behavior_config["scenarios_dir"]).expanduser()
        results: List[Dict[str, Any]] = []
        for scenario_path in sorted(scenarios_dir.iterdir()):
            if (scenario_path / "scenario.yaml").is_file():
                results.append(self.run_scenario(str(scenario_path), run_root))
        rollup = {
            "verdict": "FAIL" if any(r["verdict"] == "FAIL" for r in results)
            else "PASS",
            "scenarios": results,
        }
        rollup_path = Path(run_root).expanduser() / "behavior-rollup.yaml"
        with open(rollup_path, "w") as handle:
            yaml.safe_dump(rollup, handle, sort_keys=False)
        return rollup
