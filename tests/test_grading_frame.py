# Grading frame tests (P3): the writer -> validate -> verify -> repair loop,
# exam mode, and full-mode scorecard assembly — fake sessions at the provider
# boundary, real frame machinery everywhere else (Rule 9).

from pathlib import Path

import pytest
import yaml

from kapso.learning.graders.frame import GradingFrame
from kapso.learning.mining import MiningFrame
from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from tests.test_bank_retriever import build_bank, card_text
from tests.test_mining_frame import (
    MINING_CONFIG,
    FakeFactory,
    FakeLead,
    write_valid_mined,
)
from tests.test_trajectory_store import TRAJECTORY_ID, build_work_dir

GOOD_REPORT = """---
trajectory: {trajectory}
bank_head: lr_test
brief: brief.md
hindcast:
  foresight: 0.50
  accuracy: null
  serving: null
  score: 0.50
  rationale: >-
    One hit, one learnable miss; nothing settled (thin), nothing served that
    could land. Foresight binds.
---

## Extraction
- **HIT-UNSERVED** — the campaign's winning move was banked but left out of
  the brief [mined/it-1/flow-1.md].
- **MISS-UNCARDED** — a learnable lesson never carded
  [mined/it-1/flow-1.md].

## Claims settlement

## Serving
"""

CLEAN_FINDINGS = """No blocking defects found.

## Verified clean
- NOVEL attestations: none claimed.
- Settlements: none written, none to attack.
"""

VERDICT_YAML = """decision: accept
rationale: >-
  First generation, no incumbent: the absolute picture is coherent and the
  reports' rationales agree with their markers.
"""


class FakeRoleAgent:
    """One fake for all three roles: decides what to write by reading its
    prompt (report path / findings path / verdict path)."""

    def __init__(self, journal, report_text=None, findings_text=CLEAN_FINDINGS):
        self.journal = journal
        self.report_text = report_text
        self.findings_text = findings_text
        self.cwd = None

    def initialize(self, workspace):
        self.cwd = workspace

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        self.journal.append(prompt)
        if "verdict_path" in prompt or "verdict.yaml" in prompt:
            target = _extract_path(prompt, "verdict.yaml")
            target.write_text(VERDICT_YAML)
        elif "findings" in prompt and "Check classes" in prompt:
            target = _extract_path(prompt, "verifier-findings.md")
            target.write_text(self.findings_text)
        else:
            target = _extract_path(prompt, "report.md")
            trajectory = prompt.split("Trajectory: ")[1].splitlines()[0].strip()
            text = self.report_text or GOOD_REPORT.format(trajectory=trajectory)
            target.write_text(text)

        class Result:
            success = True
            output = "done"

        return Result()


def _extract_path(prompt, suffix):
    for token in prompt.replace(",", " ").split():
        if token.endswith(suffix):
            return Path(token)
    raise AssertionError(f"no {suffix} path in prompt")


class RoleFactory:
    def __init__(self, agent):
        self.agent = agent

    def create(self, config):
        return self.agent


def learning_config(tmp_path):
    return {
        "learning": {
            "trajectory_store": {"local": str(tmp_path / "store"), "remote": None},
            "graders": {
                "score_band": 0.20,
                "min_settlements": 2,
                "calibration_min": 20,
                "calibration_buckets": [0.4, 0.7],
                "crew": {
                    "report_writer": {"cli": "codex", "model": "m", "effort": "xhigh"},
                    "verifier": {"cli": "claude_code", "model": "m",
                                 "effort": "xhigh", "auth_mode": "oauth"},
                    "assessor": {"cli": "codex", "model": "m", "effort": "xhigh"},
                    "repair_rounds": 1,
                    "timeout_minutes": 1,
                },
            },
            "retriever": {"k_insights": 2, "k_procedures": 1, "k_pitfalls": 1,
                          "unvisited_discount": 0.5},
        }
    }


def make_graded_store(tmp_path):
    """A store with one mined trajectory (the mining fake writes the view)."""
    config = learning_config(tmp_path)
    store = TrajectoryStore.from_config(config)
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(store, TRAJECTORY_ID, work_dir=str(work_dir), campaign_log=str(log))
    miner = MiningFrame(store, MINING_CONFIG,
                        agent_factory=FakeFactory(FakeLead([write_valid_mined])))
    miner.mine(TRAJECTORY_ID)
    return config, store


def test_exam_mode_end_to_end(tmp_path):
    # Regression: exam-before-lesson — brief compiled by the real push core,
    # writer report admitted through validator + verifier, report path
    # returned.
    config, store = make_graded_store(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    journal = []
    frame = GradingFrame(store, config, agent_factory=RoleFactory(FakeRoleAgent(journal)))
    report_path = frame.grade_exam(
        TRAJECTORY_ID, str(bank_root), "lr_test", str(tmp_path / "runs")
    )
    assert report_path.is_file()
    slot = report_path.parent
    assert (slot / "brief.md").is_file()
    record = yaml.safe_load((slot / "serving-record.yaml").read_text())
    assert record["bank_head"] == "lr_test"
    # writer prompt carried the brief and never other reports
    writer_prompt = journal[0]
    assert "brief.md" in writer_prompt and TRAJECTORY_ID in writer_prompt


def test_bad_report_gets_one_repair_then_fails(tmp_path):
    # Regression: a corridor-violating report bounces once with named
    # findings; still bad -> loud failure, never an admitted lie.
    config, store = make_graded_store(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    journal = []
    bad = GOOD_REPORT.replace("foresight: 0.50", "foresight: 0.95")
    frame = GradingFrame(
        store, config,
        agent_factory=RoleFactory(FakeRoleAgent(journal, report_text=bad)),
    )
    with pytest.raises(RuntimeError, match="outside its corridor"):
        frame.grade_exam(TRAJECTORY_ID, str(bank_root), "lr_test",
                         str(tmp_path / "runs"))
    assert any("previous report was rejected" in p for p in journal)


def test_verifier_block_finding_forces_repair(tmp_path):
    # Regression: the adversarial second opinion — a block finding rejects an
    # otherwise-valid report.
    config, store = make_graded_store(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    blocking = ("- **F-01** [block] [class: novel] the NOVEL claim has a "
                "learn-set source. Required: reclassify as MISS-UNCARDED.\n")
    journal = []
    frame = GradingFrame(
        store, config,
        agent_factory=RoleFactory(FakeRoleAgent(journal, findings_text=blocking)),
    )
    with pytest.raises(RuntimeError, match="verifier block finding"):
        frame.grade_exam(TRAJECTORY_ID, str(bank_root), "lr_test",
                         str(tmp_path / "runs"))


def test_full_mode_assembles_scorecard(tmp_path):
    # Regression: full mode grades exactly the held-out list, frame-computes
    # aggregation, and validates the assessor's verdict block.
    config, store = make_graded_store(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    split = {
        "version": 1, "rule": "family+time", "rationale": "test",
        "learn": [],
        "held_out": [{"id": TRAJECTORY_ID, "family": "rel-amazon",
                      "date": "2026-01-01"}],
    }
    frame = GradingFrame(store, config, agent_factory=RoleFactory(FakeRoleAgent([])))
    run_dir = frame.grade_full(
        split, str(bank_root), "lr_test", "crew_v1", str(tmp_path / "runs")
    )
    scorecard = yaml.safe_load((run_dir / "scorecard.yaml").read_text())
    assert scorecard["learner_version"] == "crew_v1"
    assert scorecard["n_reports"] == 1
    assert scorecard["aggregation"]["foresight"]["n"] == 1
    assert scorecard["verdict"]["decision"] == "accept"
    assert scorecard["gauntlet"] == "not-run"


def test_unmined_trajectory_refuses_grading(tmp_path):
    # Regression: grading reads mined views only — an unmined trajectory is
    # not gradable.
    config, store = make_graded_store(tmp_path)
    other_id = "rel-hm--user-churn/20260105T000000_lane-z1"
    work_dir, log = build_work_dir(tmp_path / "other")
    save_trajectory(store, other_id, work_dir=str(work_dir), campaign_log=str(log))
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    frame = GradingFrame(store, config, agent_factory=RoleFactory(FakeRoleAgent([])))
    with pytest.raises(FileNotFoundError, match="no mined view"):
        frame.grade_exam(other_id, str(bank_root), "lr_test", str(tmp_path / "runs"))
