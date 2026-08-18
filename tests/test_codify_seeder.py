# Codify seeder tests (P7.1, CD§1): the frame's layer-1 arithmetic — executed
# verdicts only (a mention never counts), closure through founding references
# with set-semantics dedup, the failed-attempt guard, and the worksheet row
# (Rule 9: each is a wrong-machine-spins-up regression).

from pathlib import Path

from kapso.learning.bank import Bank
from kapso.learning.trajectory_store import TrajectoryStore
from kapso.learning.update_frame import UpdateFrame
from tests.test_update_frame import make_config

PROC_TMPL = """---
type: procedure
title: {title}
description: >-
  A fixture procedure.
tags: []
timestamp: 2026-08-18T09:00:00Z
scope: domain
scope_conditions: "any"
representation: text
entrypoint: null
evidence:
{evidence}reliability:
  validity: 0.7
  boundary: 0.5
  coverage: 0.4
  score: 0.65
  rationale: >-
    Fixture ledger.
  state: {state}
provenance: {{version: {version}}}
log:
{log}supersedes: null
contradicts: {contradicts}
probe: >-
  Re-run the gate on one fold.
---

Run the gated acceptance procedure end to end [E1].
"""


def entry(trajectory, verdict="confirm", ref="runs/run_0001/metrics.json",
          learner_run="lr_20260810T000000"):
    return f"""  - source:
      learner_run: {learner_run}
      trajectory: {trajectory}
      ref: {ref}
      card_version: null
    verdict: {verdict}
    usage: >-
      Never served; the campaign ran the method independently.
    effect: >-
      Gate cleared as recorded.
"""


def log_row(change="Founded.", commit="lr_20260810T000000"):
    return f"""  - version: 1
    date: 2026-08-18
    commit: {commit}
    change: >-
      {change}
"""


def write_proc(root, name, evidence_rows, state="active", contradicts="[]",
               log_rows=None, version=1):
    directory = root / "procedures" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "card.md").write_text(PROC_TMPL.format(
        title=name.replace("-", " ").title(),
        evidence="".join(evidence_rows),
        state=state, contradicts=contradicts,
        log="".join(log_rows or [log_row()]),
        version=version,
    ))


def bare_bank(tmp_path):
    root = tmp_path / "bank"
    (root / "insights").mkdir(parents=True)
    (root / "insights" / "index.md").write_text("# Insights\n")
    (root / "procedures").mkdir()
    (root / "procedures" / "index.md").write_text("# Procedures\n")
    return root


def test_executed_filter_and_set_semantics(tmp_path):
    root = bare_bank(tmp_path)
    write_proc(root, "two-campaigns", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-a--t/20260101T000000_lane-a", verdict="exercise"),  # same campaign
        entry("rel-b--t/20260102T000000_lane-b", verdict="weaken"),
        entry("rel-c--t/20260103T000000_lane-c", verdict="spawn"),  # mention-only
    ])
    bank = Bank(str(root))
    card = bank.cards["two-campaigns"]
    assert bank.codify_recurrence(card) == 2  # spawn never counts; dedup holds


def test_closure_expands_founding_refs_and_dedups_shared_sources(tmp_path):
    root = bare_bank(tmp_path)
    (root / "retired" / "procedures" / "old-proc").mkdir(parents=True)
    parent = PROC_TMPL.format(
        title="Old Proc",
        evidence=entry("rel-a--t/20260101T000000_lane-a")
        + entry("rel-d--t/20260104T000000_lane-d"),
        state="superseded", contradicts="[]", log=log_row(), version=1,
    )
    (root / "retired" / "procedures" / "old-proc" / "card.md").write_text(parent)
    write_proc(root, "successor", [
        entry("rel-a--t/20260101T000000_lane-a"),  # shared with the parent
        """  - source:
      learner_run: lr_20260815T000000
      trajectory: null
      ref: retired/procedures/old-proc/card.md#evidence
      card_version: null
    verdict: confirm
    usage: >-
      Merge founding — stands for the parent's full ledger.
    effect: >-
      Parent ledger net outcome.
""",
    ])
    bank = Bank(str(root))
    # closure = successor's own rel-a + parent's {rel-a, rel-d}; union dedups
    assert bank.codify_recurrence(bank.cards["successor"]) == 2


def test_failed_attempt_blocks_until_new_executed_evidence(tmp_path):
    root = bare_bank(tmp_path)
    write_proc(root, "burned-once", [
        entry("rel-a--t/20260101T000000_lane-a", learner_run="lr_20260810T000000"),
        entry("rel-b--t/20260102T000000_lane-b", learner_run="lr_20260811T000000"),
    ], log_rows=[
        log_row(),
        log_row(change="codify attempt failed — trace at attempts/1/",
                commit="lr_20260812T000000"),
    ], version=2)
    write_proc(root, "burned-then-confirmed", [
        entry("rel-a--t/20260101T000000_lane-a", learner_run="lr_20260810T000000"),
        entry("rel-b--t/20260102T000000_lane-b", learner_run="lr_20260813T000000"),
    ], log_rows=[
        log_row(),
        log_row(change="codify attempt failed — trace at attempts/1/",
                commit="lr_20260812T000000"),
    ], version=2)
    bank = Bank(str(root))
    assert bank.codify_blocked_by_failed_attempt(bank.cards["burned-once"])
    assert not bank.codify_blocked_by_failed_attempt(
        bank.cards["burned-then-confirmed"]
    )


def test_seeder_rows_fire_only_for_eligible_text_procedures(tmp_path):
    root = bare_bank(tmp_path)
    write_proc(root, "ready-proc", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-b--t/20260102T000000_lane-b"),
    ])
    write_proc(root, "single-campaign", [
        entry("rel-a--t/20260101T000000_lane-a"),
    ])
    write_proc(root, "contested-proc", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-b--t/20260102T000000_lane-b"),
    ], contradicts="[other-card]")
    config = make_config(tmp_path)
    frame = UpdateFrame(TrajectoryStore.from_config(config), config)
    run_dir = tmp_path / "run"
    (run_dir / "work").mkdir(parents=True)
    frame._seed_worksheet(run_dir, [], Bank(str(root)))
    worksheet = (run_dir / "work" / "observations.md").read_text()
    codify_rows = [l for l in worksheet.splitlines() if "[seed: codify]" in l
                   or "codify" in l]
    assert any("ready-proc" in l and "2 executed source campaigns" in l
               for l in codify_rows)
    assert not any("single-campaign" in l for l in codify_rows)
    assert not any("contested-proc" in l for l in codify_rows)
