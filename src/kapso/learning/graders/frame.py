# Grading frame — the deterministic machinery around the grader roles.
#
# Design: learn-from-trajectories-grader-scoring.md §6, with one Rule-10
# simplification made at implementation and flagged in the build report: the
# "lead" session's duties there (coverage counting, delegation, mode
# switching) are all mechanical, so the FRAME does them — the crew's
# information barriers survive as separate sessions, which were the point:
# each report-writer sees ONE trajectory (never other reports, scorecards, or
# trends), the verifier is a different model attacking each report, and the
# scorecard-assessor is the only role that sees the whole set. All arithmetic
# is frame math; agent numbers are never trusted for aggregation.
#
# Modes: `grade_exam` (operating regime — one arriving trajectory, writer +
# verifier only) and `grade_full` (development regime — every held-out
# trajectory + aggregation + assessor verdict; gauntlet verdicts merge in
# when the P4 trap runners exist).

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.bank import Bank
from kapso.learning.graders.hindcast import HindcastReport, HindcastValidator
from kapso.learning.graders.scorecard import (
    aggregate,
    calibration_table,
    paired_deltas,
    validate_verdict,
)
from kapso.learning.graders.split import held_out_ids
from kapso.learning.retriever import compile_brief
from kapso.learning.trajectory_store import TrajectoryStore

CREW_DIR = Path(__file__).parent.parent / "crews" / "grading"
_BLOCK_FINDING_PATTERN = re.compile(r"^- \*\*F-\d+\*\* \[block\]", re.MULTILINE)


def _fill(template: str, values: Dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    if "{{" in template:
        raise ValueError(f"unfilled placeholder in prompt: {template[template.index('{{'):][:60]}")
    return template


class GradingFrame:
    """Stage -> sessions -> validate -> aggregate for hindcast grading."""

    def __init__(
        self,
        store: TrajectoryStore,
        config: Dict[str, Any],
        agent_factory=CodingAgentFactory,
    ):
        self.store = store
        self.learning_config = config["learning"]
        self.graders_config = self.learning_config["graders"]
        self.crew_config = self.graders_config["crew"]
        self.retriever_config = self.learning_config["retriever"]
        self.agent_factory = agent_factory

    # ------------------------------------------------------------- sessions

    def _run_role(self, role: str, prompt: str, cwd: Path) -> str:
        spec = self.crew_config[role]
        agent_specific: Dict[str, Any] = {"effort": spec["effort"]}
        if spec["cli"] == "codex":
            agent_specific["sandbox"] = "workspace-write"
        else:
            agent_specific["auth_mode"] = spec["auth_mode"]
        agent = self.agent_factory.create(CodingAgentConfig(
            agent_type=spec["cli"],
            model=spec["model"],
            debug_model=spec["model"],
            agent_specific=agent_specific,
        ))
        agent.initialize(str(cwd))
        timeout = self.crew_config["timeout_minutes"] * 60
        result = agent.generate_code(prompt, timeout_seconds=timeout)
        return result.output or ""

    def _run_role_for_artifact(
        self, role: str, prompt: str, cwd: Path, artifact: Path
    ) -> None:
        """Run a role session that must produce `artifact`. A session can
        die without writing it for infrastructure reasons (rate-limit
        events, provider disconnects) — retry ONCE with a fresh session,
        then let the caller's missing-artifact raise fire (fail loud; the
        retry covers transport, never content)."""
        self._run_role(role, prompt, cwd)
        if not artifact.is_file():
            self._run_role(role, prompt, cwd)

    # ------------------------------------------------------------ one exam

    def grade_trajectory(
        self,
        trajectory_id: str,
        bank_dir: str,
        bank_head: str,
        run_dir: Path,
        learn_set_dir: str,
    ) -> Dict[str, Any]:
        """Writer -> frame validation -> verifier -> (one repair) -> admitted
        report. Returns the parsed report frontmatter + paths."""
        manifest = self.store.manifest(trajectory_id)
        if not manifest.get("derived", {}).get("mined"):
            raise FileNotFoundError(
                f"{trajectory_id} has no mined view — mine before grading"
            )
        bundle_dir = self.store.resolve(trajectory_id)
        bank = Bank(bank_dir)
        task_coords = {
            "family": manifest.get("family"), "dataset": manifest.get("dataset")
        }

        slot = run_dir / "hindcast" / trajectory_id.replace("/", "--")
        slot.mkdir(parents=True, exist_ok=True)
        push = compile_brief(bank, task_coords, bank_head, self.retriever_config)
        (slot / "brief.md").write_text(push["brief"])
        with open(slot / "serving-record.yaml", "w") as handle:
            yaml.safe_dump(push["record"], handle, sort_keys=False)

        values = {
            "trajectory_id": trajectory_id,
            "bank_head": bank_head,
            "mined_dir": bundle_dir / "mined",
            "bundle_dir": bundle_dir,
            "bank_dir": Path(bank_dir).resolve(),
            "brief_path": slot / "brief.md",
            "record_path": slot / "serving-record.yaml",
            "learn_set_dir": learn_set_dir,
            "report_path": slot / "report.md",
            "findings_path": slot / "verifier-findings.md",
        }
        writer_prompt = _fill((CREW_DIR / "report_writer_prompt.md").read_text(), values)
        verifier_prompt = _fill((CREW_DIR / "verifier_prompt.md").read_text(), values)

        inventory = manifest["inventory"]["sha256"]
        # The listing is the allowed source surface (the bank's past). A ref
        # outside the trajectory's own bundle must land inside a listed mined
        # view — the split/chronology contract is enforced here, not trusted
        # to the writers.
        allowed_roots = [
            Path(line).resolve()
            for line in Path(learn_set_dir).read_text().splitlines()
            if line.strip()
        ]
        bundle_root = bundle_dir.resolve()

        def ref_exists(path: str) -> bool:
            if path in inventory or path == "brief.md":
                return True
            target = (bundle_dir / path).resolve()
            if not target.exists() and "trajectories/" in path:
                # Store-rooted refs (data/trajectories/<id>/...) are
                # unambiguous and auditable — writers copy them straight
                # from the learn-set listing. Map the store tail and hold
                # them to the same allowed-roots law.
                tail = path.split("trajectories/", 1)[1]
                target = (self.store.local / tail).resolve()
            if not target.exists():
                return False
            if target.is_relative_to(bundle_root):
                return True
            return any(target.is_relative_to(root) for root in allowed_roots)

        validator = HindcastValidator(
            self.graders_config, ref_exists, set(bank.cards)
        )

        repair_rounds = self.crew_config["repair_rounds"]
        rounds_used = 0
        self._run_role_for_artifact(
            "report_writer", writer_prompt, run_dir, slot / "report.md"
        )
        while True:
            report_path = slot / "report.md"
            if not report_path.is_file():
                raise FileNotFoundError(
                    f"report-writer produced no report for {trajectory_id}"
                )
            report = HindcastReport.parse(report_path.read_text())
            findings = validator.validate(report)
            if not findings:
                self._run_role_for_artifact(
                    "verifier", verifier_prompt, run_dir,
                    slot / "verifier-findings.md",
                )
                findings = self._block_findings(slot / "verifier-findings.md")
            if not findings:
                return {
                    "trajectory": trajectory_id,
                    "frontmatter": report.frontmatter,
                    "report": report,
                    "slot": slot,
                }
            if rounds_used >= repair_rounds:
                raise RuntimeError(
                    f"hindcast report for {trajectory_id} failed after "
                    f"{rounds_used} repair round(s): " + "; ".join(findings)
                )
            rounds_used += 1
            numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(findings))
            self._run_role_for_artifact(
                "report_writer",
                writer_prompt
                + "\n\nYour previous report was rejected. Named findings — fix "
                  "each and rewrite the report file:\n" + numbered,
                run_dir,
                slot / "report.md",
            )

    def _block_findings(self, findings_path: Path) -> List[str]:
        if not findings_path.is_file():
            raise FileNotFoundError(f"verifier produced no findings file at {findings_path}")
        text = findings_path.read_text()
        blocks = _BLOCK_FINDING_PATTERN.findall(text)
        if not blocks:
            return []
        entries = [
            line for line in text.splitlines() if "[block]" in line
        ]
        return [f"verifier block finding: {line.strip()}" for line in entries]

    # ---------------------------------------------------------- full grading

    def grade_full(
        self,
        split: Dict[str, Any],
        bank_dir: str,
        bank_head: str,
        learner_version: str,
        run_root: str,
        incumbent_reports: Optional[List[Dict[str, Any]]] = None,
        gauntlet_rollup: Optional[str] = None,
    ) -> Path:
        """The development-regime exam: every held-out trajectory graded, the
        scorecard assembled with frame math, the assessor writing only the
        decision + rationale (validated). Full mode grades exactly
        split.held_out — nothing else is the exam."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = Path(run_root).expanduser() / stamp
        run_dir.mkdir(parents=True)
        learn_set_dir = self._learn_set_listing(split, run_dir)

        graded = []
        for trajectory_id in held_out_ids(split):
            graded.append(self.grade_trajectory(
                trajectory_id, bank_dir, bank_head, run_dir, learn_set_dir
            ))

        report_rows = [
            {
                "trajectory": g["trajectory"],
                "split_version": split["version"],
                "hindcast": g["frontmatter"].get("hindcast", {}),
            }
            for g in graded
        ]
        aggregation = aggregate(report_rows)
        deltas = (
            paired_deltas(report_rows, incumbent_reports)
            if incumbent_reports else None
        )
        settlements = self._settlements(graded)
        calibration = calibration_table(settlements, self.graders_config)

        for name, payload in (
            ("aggregation.yaml", aggregation),
            ("deltas.yaml", deltas),
            ("calibration.yaml", {"pooled": len(settlements), "table": calibration}),
        ):
            with open(run_dir / name, "w") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)

        assessor_values = {
            "aggregation_path": run_dir / "aggregation.yaml",
            "deltas_path": run_dir / "deltas.yaml",
            "calibration_path": run_dir / "calibration.yaml",
            "gauntlet_path": run_dir / "gauntlet.md",
            "reports_dir": run_dir / "hindcast",
            "verdict_path": run_dir / "verdict.yaml",
        }
        self._run_role(
            "assessor",
            _fill((CREW_DIR / "assessor_prompt.md").read_text(), assessor_values),
            run_dir,
        )
        verdict_path = run_dir / "verdict.yaml"
        if not verdict_path.is_file():
            raise FileNotFoundError("assessor produced no verdict.yaml")
        verdict = yaml.safe_load(verdict_path.read_text())
        if not isinstance(verdict, dict):
            raise ValueError("verdict.yaml is not a mapping")
        verdict_findings = validate_verdict(
            verdict, deltas or {}, gauntlet_rollup or "PASS"
        )
        if verdict_findings:
            raise RuntimeError(
                "scorecard verdict rejected: " + "; ".join(verdict_findings)
            )

        scorecard = {
            "learner_version": learner_version,
            "split_version": split["version"],
            "bank_head": bank_head,
            "n_reports": len(graded),
            "aggregation": aggregation,
            "deltas": deltas,
            "calibration": calibration,
            "calibration_pooled": len(settlements),
            "gauntlet": gauntlet_rollup or "not-run",
            "verdict": verdict,
        }
        with open(run_dir / "scorecard.yaml", "w") as handle:
            yaml.safe_dump(scorecard, handle, sort_keys=False)
        return run_dir

    # ------------------------------------------------------------ exam mode

    def grade_exam(
        self,
        trajectory_id: str,
        bank_dir: str,
        bank_head: str,
        run_root: str,
        learn_set_ids: List[str],
    ) -> Path:
        """Exam-before-lesson: one arriving trajectory, writer + verifier only
        — no gauntlet, no scorecard. The CALLER owns the allowed source
        surface (the bank's past): the development driver passes the ids
        already ingested (batch 0 passes none — an empty past is a valid
        past, every miss is then MISS-NOVEL and foresight has no
        denominator); the operating regime passes the whole local store minus
        the arrival, because there local IS the past. Deriving the surface
        from the store here would leak held-out and future material into a
        development replay."""
        if trajectory_id in learn_set_ids:
            raise ValueError(
                f"{trajectory_id} cannot be in its own learn-set surface"
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = Path(run_root).expanduser() / f"exam-{stamp}"
        run_dir.mkdir(parents=True)
        lines = []
        for learn_id in learn_set_ids:
            mined = self.store.local / learn_id / "mined"
            if not mined.is_dir():
                raise FileNotFoundError(
                    f"learn-set id {learn_id} has no mined view at {mined}"
                )
            lines.append(str(mined))
        listing = run_dir / "learn-set-mined-views.txt"
        listing.write_text("\n".join(lines) + "\n" if lines else "")
        graded = self.grade_trajectory(
            trajectory_id, bank_dir, bank_head, run_dir, str(listing)
        )
        return graded["slot"] / "report.md"

    # ------------------------------------------------------------- helpers

    def _learn_set_listing(self, split: Dict[str, Any], run_dir: Path) -> str:
        """A listing file of learn-set mined-view paths — the writers' and
        verifier's source-search surface (never the held-out views)."""
        lines = []
        for entry in split["learn"]:
            bundle_dir = self.store.local / entry["id"]
            mined = bundle_dir / "mined"
            if mined.is_dir():
                lines.append(str(mined))
        listing = run_dir / "learn-set-mined-views.txt"
        listing.write_text("\n".join(lines) + "\n")
        return str(listing)

    def _settlements(self, graded: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calibration rows: settled claims whose card was in the serving
        record (claimed = the served reliability score). Cards settled
        without being served have no claimed score to calibrate — skipped by
        construction, not by judgment."""
        settlements = []
        for g in graded:
            record = yaml.safe_load((g["slot"] / "serving-record.yaml").read_text())
            served_scores = {
                row["card"]: row.get("score") for row in record.get("served", [])
            }
            for entry in g["report"].sections["claims"]:
                if entry["marker"] not in ("AGREED", "CONTRADICTED"):
                    continue
                for _, name in entry["card_refs"]:
                    if name in served_scores and served_scores[name] is not None:
                        settlements.append({
                            "claimed": served_scores[name],
                            "agreed": entry["marker"] == "AGREED",
                        })
        return settlements
