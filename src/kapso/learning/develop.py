# Development driver — the learner-as-candidate loop (design §4.4).
#
# One development run of a learner version: a FRESH disposable bank, the
# learn-set replayed chronologically with exam-before-lesson (each batch is
# hindcast against the evolving bank BEFORE ingestion — the training curve),
# then the real exam (grade_full on held-out) and the keep-best ledger row.
# Banks are disposable here; the artifact under development is the crew.
#
# The driver is deliberately dumb: batching is chronology + a size knob, the
# split is the exam's authority (batch disjointness asserted), and every
# decision that matters comes from the graders, not from here.

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.graders.frame import GradingFrame
from kapso.learning.graders.split import assert_batch_disjoint
from kapso.learning.trajectory_store import TrajectoryStore
from kapso.learning.update_frame import UpdateFrame, init_bank


def _chronological_batches(
    split: Dict[str, Any], batch_size: int
) -> List[List[str]]:
    entries = sorted(split["learn"], key=lambda e: (str(e["date"]), e["id"]))
    ids = [entry["id"] for entry in entries]
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


class DevelopmentDriver:
    """Run one learner version through the development regime."""

    def __init__(
        self,
        store: TrajectoryStore,
        config: Dict[str, Any],
        agent_factory=CodingAgentFactory,
    ):
        self.store = store
        self.config = config
        self.learning_config = config["learning"]
        self.develop_config = self.learning_config["develop"]
        self.agent_factory = agent_factory

    def run(self, split: Dict[str, Any], learner_version: str) -> Path:
        """Fresh bank -> chronological exam-then-ingest over the learn set ->
        full exam on held-out -> versions ledger row. Returns the scorecard
        run dir."""
        root = Path(self.develop_config["run_root"]).expanduser() / learner_version
        if root.exists():
            raise FileExistsError(
                f"development run for {learner_version} already exists at {root} "
                f"— learner versions are immutable; bump the version"
            )
        root.mkdir(parents=True)
        bank_home = root / "bank-home.git"
        init_bank(str(bank_home))

        # Thread the disposable bank home through a run-scoped config copy —
        # the production bank config is never touched by development runs.
        run_config = {
            **self.config,
            "learning": {
                **self.learning_config,
                "bank": {"local_path": str(bank_home), "remote": None},
            },
        }
        update_frame = UpdateFrame(
            self.store, run_config, agent_factory=self.agent_factory
        )
        grading = GradingFrame(
            self.store, run_config, agent_factory=self.agent_factory
        )

        batches = _chronological_batches(split, self.develop_config["batch_size"])
        curve: List[Dict[str, Any]] = []
        # The exam's allowed source surface is the bank's past: exactly the
        # learn trajectories already ingested. The current batch is not past
        # yet (exam-before-lesson), and held-out material never enters.
        ingested: List[str] = []
        for index, batch_ids in enumerate(batches):
            assert_batch_disjoint(split, batch_ids)
            bank_head = self._bank_head(bank_home)
            checkout = self._serving_checkout(root, bank_home, index)
            batch = []
            for trajectory_id in batch_ids:
                report_path = grading.grade_exam(
                    trajectory_id, str(checkout), bank_head,
                    str(root / "exams" / f"batch-{index:02d}"),
                    list(ingested),
                )
                report = yaml.safe_load(
                    report_path.read_text().split("---")[1]
                )
                curve.append({
                    "batch": index,
                    "trajectory": trajectory_id,
                    "hindcast": report.get("hindcast", {}),
                })
                batch.append({
                    "trajectory": trajectory_id,
                    "hindcast_report": str(report_path),
                })
            update_frame.run_update(
                batch, str(root / "updates"), learner_version
            )
            ingested.extend(batch_ids)
        with open(root / "training-curve.yaml", "w") as handle:
            yaml.safe_dump(curve, handle, sort_keys=False)

        bank_head = self._bank_head(bank_home)
        exam_checkout = self._serving_checkout(root, bank_home, len(batches))
        scorecard_dir = grading.grade_full(
            split, str(exam_checkout), bank_head, learner_version,
            str(root / "graders"),
        )
        self._ledger_row(learner_version, scorecard_dir)
        return scorecard_dir

    # ------------------------------------------------------------- helpers

    def _bank_head(self, bank_home: Path) -> str:
        return subprocess.run(
            ["git", "--git-dir", str(bank_home), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def _serving_checkout(self, root: Path, bank_home: Path, index: int) -> Path:
        """A pinned read-only checkout of the current head (what the exam's
        push replay reads)."""
        checkout = root / "serving" / f"head-{index:02d}"
        if checkout.exists():
            shutil.rmtree(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", str(bank_home), str(checkout)],
            check=True, capture_output=True,
        )
        return checkout

    def _ledger_row(self, learner_version: str, scorecard_dir: Path) -> None:
        """The keep-best ledger (§4.4): a learner version replaces the
        incumbent only on an `accept` decision; the ledger records every
        version's exam either way."""
        ledger_path = Path(self.develop_config["versions_ledger"]).expanduser()
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = (
            yaml.safe_load(ledger_path.read_text()) if ledger_path.is_file() else None
        ) or {"incumbent": None, "versions": []}
        scorecard = yaml.safe_load((scorecard_dir / "scorecard.yaml").read_text())
        decision = scorecard["verdict"]["decision"]
        ledger["versions"].append({
            "version": learner_version,
            "scorecard": str(scorecard_dir / "scorecard.yaml"),
            "split_version": scorecard["split_version"],
            "decision": decision,
            "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if decision == "accept":
            ledger["incumbent"] = learner_version
        with open(ledger_path, "w") as handle:
            yaml.safe_dump(ledger, handle, sort_keys=False)
