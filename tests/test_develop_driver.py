# Development-driver integration test (P4): the whole development regime on
# fakes — fresh disposable bank, exam-before-lesson replay, full exam,
# keep-best ledger. One composite fake serves all roles across both frames
# (Rule 9: this is the regression net for the regime's plumbing end to end).

from pathlib import Path

import pytest
import yaml

from kapso.learning.develop import DevelopmentDriver, _chronological_batches
from kapso.learning.mining import MiningFrame
from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from tests.test_grading_frame import CLEAN_FINDINGS, GOOD_REPORT, VERDICT_YAML, _extract_path
from tests.test_mining_frame import MINING_CONFIG, FakeFactory, FakeLead, write_valid_mined
from tests.test_trajectory_store import build_work_dir
from tests.test_update_frame import good_lead

LEARN_ID = "rel-amazon--user-churn/20260101T000000_lane-t1"
HELD_OUT_ID = "rel-event--user-repeat/20260102T000000_lane-t2"

SPLIT = {
    "version": 1, "rule": "family+time", "rationale": "driver test",
    "learn": [{"id": LEARN_ID, "family": "rel-amazon", "date": "2026-01-01"}],
    "held_out": [{"id": HELD_OUT_ID, "family": "rel-event", "date": "2026-01-02"}],
}


class CompositeFake:
    """Every session in the regime: update leads write a good run; grading
    writers/verifiers/assessors write their artifacts."""

    def __init__(self):
        self.cwd = None

    def initialize(self, workspace):
        self.cwd = workspace

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        class Result:
            success = True
            output = "done"

        if "knowledge-update crew" in prompt:
            good_lead(self.cwd)
            return Result()
        if "verdict_path" in prompt or "verdict.yaml" in prompt:
            _extract_path(prompt, "verdict.yaml").write_text(VERDICT_YAML)
            return Result()
        if "Check classes" in prompt:
            _extract_path(prompt, "verifier-findings.md").write_text(CLEAN_FINDINGS)
            return Result()
        trajectory = prompt.split("Trajectory: ")[1].splitlines()[0].strip()
        _extract_path(prompt, "report.md").write_text(
            GOOD_REPORT.format(trajectory=trajectory)
        )
        return Result()


class CompositeFactory:
    def create(self, config):
        return CompositeFake()


def develop_config(tmp_path):
    return {
        "learning": {
            "trajectory_store": {"local": str(tmp_path / "store"), "remote": None},
            "graders": {
                "score_band": 0.20, "min_settlements": 2,
                "calibration_min": 20, "calibration_buckets": [0.4, 0.7],
                "gauntlet": {"stability_tolerance": 0.10},
                "crew": {
                    "report_writer": {"cli": "codex", "model": "m", "effort": "xhigh"},
                    "verifier": {"cli": "claude_code", "model": "m",
                                 "effort": "xhigh", "auth_mode": "oauth"},
                    "assessor": {"cli": "codex", "model": "m", "effort": "xhigh"},
                    "repair_rounds": 1, "timeout_minutes": 1,
                },
            },
            "retriever": {"k_insights": 2, "k_procedures": 1, "k_pitfalls": 1,
                          "unvisited_discount": 0.5},
            "bank": {"local_path": str(tmp_path / "unused-prod-bank.git"),
                     "remote": None},
            "update_crew": {
                "lead": {"cli": "claude_code", "model": "m", "effort": "xhigh",
                         "auth_mode": "oauth"},
                "worker": {"cli": "codex", "model": "m", "effort": "xhigh"},
                "critic": {"cli": "claude_code", "model": "m"},
                "repair_rounds": 1, "timeout_minutes": 1,
                "dup_nominate_jaccard": 0.5, "sightings_expiry_batches": 6,
                "run_root": str(tmp_path / "prod-runs"),
            },
            "develop": {
                "batch_size": 3,
                "run_root": str(tmp_path / "develop"),
                "versions_ledger": str(tmp_path / "versions.yaml"),
            },
        }
    }


def seed_mined_trajectory(store, tmp_path, trajectory_id):
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(store, trajectory_id, work_dir=str(work_dir),
                    campaign_log=str(log))
    MiningFrame(store, MINING_CONFIG,
                agent_factory=FakeFactory(FakeLead([write_valid_mined]))
                ).mine(trajectory_id)


def test_development_regime_end_to_end(tmp_path):
    # Regression: fresh bank per version, exam-before-lesson curve recorded,
    # learn-set ingested, held-out exam produces the scorecard, ledger row
    # written with incumbent promotion on accept.
    config = develop_config(tmp_path)
    store = TrajectoryStore.from_config(config)
    seed_mined_trajectory(store, tmp_path / "a", LEARN_ID)
    seed_mined_trajectory(store, tmp_path / "b", HELD_OUT_ID)

    driver = DevelopmentDriver(store, config, agent_factory=CompositeFactory())
    scorecard_dir = driver.run(SPLIT, "crew_v1")

    scorecard = yaml.safe_load((scorecard_dir / "scorecard.yaml").read_text())
    assert scorecard["learner_version"] == "crew_v1"
    assert scorecard["n_reports"] == 1  # exactly the held-out set
    assert scorecard["gauntlet"] == "PASS"  # traps ran before the exam

    root = Path(config["learning"]["develop"]["run_root"]) / "crew_v1"
    curve = yaml.safe_load((root / "training-curve.yaml").read_text())
    assert [c["trajectory"] for c in curve] == [LEARN_ID]  # prequential order
    # Split-leak regression: batch 0's exam surface is the bank's past —
    # empty — never "everything else in the store" (which holds held-out).
    listings = list((root / "exams").glob("*/*/learn-set-mined-views.txt"))
    assert listings and all(l.read_text() == "" for l in listings)
    # The traps ran the crew on sandbox homes and left their proof.
    gauntlet = yaml.safe_load((root / "gauntlet.md").read_text().split("---")[1])
    assert gauntlet["verdict"] == "PASS"
    assert set(gauntlet["gauntlet"]) == {"duplicate", "stability"}
    assert (root / "bank-home.git").is_dir()  # the disposable bank
    # the learn-set lesson landed in the disposable bank, not the prod bank
    assert list((root / "updates").glob("lr_*/report.md"))
    assert not Path(config["learning"]["bank"]["local_path"]).exists()

    ledger = yaml.safe_load(
        Path(config["learning"]["develop"]["versions_ledger"]).read_text()
    )
    assert ledger["incumbent"] == "crew_v1"
    assert ledger["versions"][0]["decision"] == "accept"

    # learner versions are immutable — a re-run must refuse
    with pytest.raises(FileExistsError, match="immutable"):
        driver.run(SPLIT, "crew_v1")


def test_chronological_batching():
    # Regression: replay order is time, not listing order.
    split = {"learn": [
        {"id": "b", "date": "2026-02-01"},
        {"id": "a", "date": "2026-01-01"},
        {"id": "c", "date": "2026-03-01"},
    ]}
    assert _chronological_batches(split, 2) == [["a", "b"], ["c"]]
