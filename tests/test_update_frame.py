# Update-run frame tests (P4): worksheet seeding, coverage arithmetic, the
# transaction pipeline (validate -> repair -> commit -> report), against a
# fake lead at the provider boundary (Rule 9).

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from kapso.learning.update_frame import UpdateFrame, init_bank
from tests.test_bank_retriever import card_text
from tests.test_mining_frame import FakeFactory, FakeLead
from tests.test_trajectory_store import TRAJECTORY_ID, build_work_dir

HINDCAST_FIXTURE = """---
trajectory: {trajectory}
bank_head: lr_seed
brief: brief.md
hindcast:
  foresight: 0.50
  accuracy: null
  serving: null
  score: 0.50
  rationale: >-
    Fixture report.
---

## Extraction
- **MISS-UNCARDED** — a lesson with a learn-set source
  [mined/it-1/flow-1.md].

## Claims settlement
- **AGREED** — [insight: a-card]: predicted and measured +0.7136 ± 0.001
  [mined/it-1/flow-1.md].

## Serving
- **UPTAKE-FAIL** — [insight: a-card]: served, re-derived anyway
  [mined/it-1/flow-1.md].
"""

# Serving section extended with a payoff-graded used entry and a noise
# entry — used by the seeding test only (scripted-fake tests keep the
# minimal report so their turn counts stay stable).
SERVING_OUTCOME_LINES = """- **SERVED-USED** — [insight: a-card]: followed, decision paid +0.001
  [mined/it-1/flow-1.md].
- **SERVE-NOISE** — [insight: b-card]: served, irrelevant here
  [mined/it-1/flow-1.md].
"""

EVIDENCE_APPEND = """  - source:
      learner_run: {lr_id}
      trajectory: {trajectory}
      ref: runs/run_0001/metrics.json
      card_version: null
    verdict: confirm
    usage: >-
      Independent evidence — the campaign never saw the card.
    effect: >-
      KEPT at +0.7136 ~ 3.6 clustered SE on the validation split.
"""


def make_config(tmp_path):
    return {
        "learning": {
            "trajectory_store": {"local": str(tmp_path / "store"), "remote": None},
            "graders": {"score_band": 0.20, "min_settlements": 2,
                        "calibration_min": 20, "calibration_buckets": [0.4, 0.7]},
            "bank": {"local_path": str(tmp_path / "bank-home.git"), "remote": None},
            "codify": {"min_recurrence": 2, "replay_max_age_days": 60},
            "update_crew": {
                "lead": {"cli": "claude_code", "model": "m", "effort": "xhigh",
                         "auth_mode": "oauth"},
                "worker": {"cli": "codex", "model": "m", "effort": "xhigh"},
                "critic": {"cli": "claude_code", "model": "m"},
                "repair_rounds": 1,
                "timeout_minutes": 1,
                "dup_nominate_jaccard": 0.5,
                "sightings_expiry_batches": 6,
                "rewrite_rows_per_run": 2,
                "body_floors": {"rule": 35, "section": 25, "confidence": 8},
                "run_root": str(tmp_path / "runs"),
            },
        }
    }


def seed_bank_home(tmp_path, cards):
    """init_bank + push the fixture cards as the founding state."""
    home = tmp_path / "bank-home.git"
    init_bank(str(home))
    seed = tmp_path / "seed-clone"
    subprocess.run(["git", "clone", str(home), str(seed)],
                   check=True, capture_output=True)
    for name, text in cards.items():
        (seed / "insights" / f"{name}.md").write_text(text)
    listing = "\n".join(f"- [{n}]({n}.md) — hero" for n in cards)
    (seed / "insights" / "index.md").write_text("# Insights\n" + listing + "\n")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "founding cards"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"],
                   check=True, capture_output=True)
    return home


def make_batch(tmp_path, store):
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(store, TRAJECTORY_ID, work_dir=str(work_dir),
                    campaign_log=str(log))
    report_path = tmp_path / "hindcast-report.md"
    report_path.write_text(HINDCAST_FIXTURE.format(trajectory=TRAJECTORY_ID))
    return [{"trajectory": TRAJECTORY_ID, "hindcast_report": str(report_path)}]


def good_lead(workspace):
    """A fake lead that does a minimal honest run: one verdict per seeded
    row, one admissible evidence append, bookends written."""
    run_dir = Path(workspace)
    worksheet = (run_dir / "work" / "observations.md").read_text()
    inputs = yaml.safe_load((run_dir / "inputs.yaml").read_text())
    journal = ["# Routing journal", ""]
    for line in worksheet.splitlines():
        if not line.startswith("- **"):
            continue
        row_id = line.split("**")[1]
        if "[seed: lift" in line:
            journal.append(
                f"- **{row_id} → ATTACH** (fast-path) — lifted the AGREED "
                f"settlement onto a-card; delta copied, never recomputed. "
                f"[mined/it-1/flow-1.md]"
            )
            card_path = run_dir / "bank" / "insights" / "a-card.md"
            text = card_path.read_text()
            append = EVIDENCE_APPEND.format(
                lr_id=inputs["lr_id"],
                trajectory=inputs["batch"][0]["trajectory"],
            )
            text = text.replace("reliability:", append + "reliability:", 1)
            # Step D: an outcome verdict demands reassessment — rationale
            # rewritten citing the new entry, version bump + log entry (the
            # frame enforces it; reliability is claim-layer).
            text = text.replace(
                "Validity from two confirmations; boundary untested; "
                "coverage thin.",
                "Reassessed after the lifted settlement: in-scope confirm "
                "holds validity; boundary still untested — a boundary probe "
                "is the next mover.",
            )
            text = text.replace("provenance: {version: 1}",
                                "provenance: {version: 2}")
            text = text.replace(
                "supersedes: null",
                "  - version: 2\n    date: 2026-08-18\n"
                f"    commit: {inputs['lr_id']}\n"
                "    change: Reassessed after the lifted settlement.\n"
                "supersedes: null",
                1,
            )
            card_path.write_text(text)
        elif "[seed: card-candidate]" in line:
            journal.append(
                f"- **{row_id} → SIGHTING** — single observation, no endorsed "
                f"mechanism; awaiting recurrence. [mined/it-1/flow-1.md]"
            )
            sightings = run_dir / "bank" / "sightings.md"
            sightings.write_text(
                sightings.read_text()
                + f"- 2026-08-17 · {inputs['batch'][0]['trajectory']} · a "
                  f"lesson awaiting recurrence\n"
            )
        elif "[seed: serving-feedback]" in line:
            journal.append(
                f"- **{row_id} → NOTE** — uptake failure acknowledged for the "
                f"closing assessment; serving-side, no bank edit."
            )
    (run_dir / "work" / "journal.md").write_text("\n".join(journal) + "\n")
    (run_dir / "work" / "headline.md").write_text(
        "One settlement lifted onto a-card; one sighting recorded.\n"
    )
    (run_dir / "work" / "closing.md").write_text(
        "Serving uptake failed once on a-card — watch the rendering.\n"
    )
    (run_dir / "work" / "critic-findings.md").write_text(
        "- **F-01** [warn] [class: routing] none — clean pass. Required: n/a\n"
    )
    return "run complete"


def make_frame(tmp_path, writers, cards=None):
    config = make_config(tmp_path)
    store = TrajectoryStore.from_config(config)
    seed_bank_home(tmp_path, cards if cards is not None
                   else {"a-card": card_text("a-card")})
    batch = make_batch(tmp_path, store)
    lead = FakeLead(writers)
    frame = UpdateFrame(store, config, agent_factory=FakeFactory(lead))
    return frame, config, batch, lead


def test_clean_run_commits_tags_and_reports(tmp_path):
    # Regression: the full pipeline — seeded worksheet, honest fake run,
    # validation green, one commit tagged lr_<id> pushed to the home, report
    # assembled with the health block.
    frame, config, batch, _ = make_frame(tmp_path, [good_lead])
    run_dir = frame.run_update(
        batch, config["learning"]["update_crew"]["run_root"], "crew_v1"
    )
    report = (run_dir / "report.md").read_text()
    assert report.startswith("---\n")
    frontmatter = yaml.safe_load(report.split("---")[1])
    assert frontmatter["batch"] == [TRAJECTORY_ID]
    assert frontmatter["health"]["cards"]["active"] == 1
    assert frontmatter["bank"]["before"] != frontmatter["bank"]["after"]
    tags = subprocess.run(
        ["git", "-C", config["learning"]["bank"]["local_path"], "tag"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert frontmatter["run"] in tags
    # the evidence append survived the push
    show = subprocess.run(
        ["git", "-C", config["learning"]["bank"]["local_path"], "show",
         "main:insights/a-card.md"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert TRAJECTORY_ID in show
    # derived edges rebuilt
    assert (run_dir / "bank" / "index" / "edges.yaml").is_file()


def test_rewrite_rows_capped_and_priority_ordered(tmp_path):
    # Migration seeding: non-conforming bodies get rewrite rows, highest
    # reliability first, capped by config; conforming cards seed nothing.
    cards = {
        "a-card": card_text("a-card"),  # conforming; good_lead edits it
        "legacy-low": card_text("legacy-low", score=0.3,
                                body="Old prose without sections."),
        "legacy-high": card_text("legacy-high", score=0.9,
                                 body="Old prose without sections."),
        "legacy-mid": card_text("legacy-mid", score=0.6,
                                body="Old prose without sections."),
        "conforming": card_text("conforming", score=0.99),
    }
    captured = {}

    def snoop(workspace):
        captured["worksheet"] = (
            Path(workspace) / "work" / "observations.md"
        ).read_text()
        return good_lead(workspace)

    frame, config, batch, _ = make_frame(tmp_path, [snoop, snoop], cards=cards)
    with pytest.raises(RuntimeError):
        frame.run_update(
            batch, config["learning"]["update_crew"]["run_root"], "crew_v1"
        )
    worksheet = captured["worksheet"]
    rewrite_rows = [l for l in worksheet.splitlines() if "[rewrite]" in l]
    assert len(rewrite_rows) == 2  # capped by rewrite_rows_per_run
    assert "legacy-high" in rewrite_rows[0]
    assert "legacy-mid" in rewrite_rows[1]
    assert not any("conforming" in row for row in rewrite_rows)
    assert not any("legacy-low" in row for row in rewrite_rows)


def test_worksheet_seeding_classes(tmp_path):
    # Regression: the seeded worksheet carries lift/card-candidate/
    # serving-feedback rows from the hindcast report and tension + dup-merge
    # docket rows from the pre-run bank.
    cards = {
        "a-card": card_text("a-card", contradicts=("b-card",)),
        "b-card": card_text("b-card", contradicts=("a-card",)),
        "twin-one": card_text(
            "twin-one",
            body="Group-relative ranking beats absolute values because "
                 "competition happens within the pool [E1]."),
        "twin-two": card_text(
            "twin-two",
            body="Group-relative ranking beats absolute values because "
                 "competition happens within the pool, twice [E1]."),
    }
    captured = {}

    def snoop(workspace):
        captured["worksheet"] = (
            Path(workspace) / "work" / "observations.md"
        ).read_text()
        return good_lead(workspace)

    frame, config, batch, _ = make_frame(tmp_path, [snoop, snoop], cards=cards)
    # extend the report's serving section for this test only
    report_path = Path(batch[0]["hindcast_report"])
    report_path.write_text(report_path.read_text() + SERVING_OUTCOME_LINES)
    # the run will fail validation later (docket rows unhandled by good_lead)
    # — seeding is what this test asserts.
    with pytest.raises(RuntimeError):
        frame.run_update(
            batch, config["learning"]["update_crew"]["run_root"], "crew_v1"
        )
    worksheet = captured["worksheet"]
    assert "[seed: lift → a-card]" in worksheet
    assert "[seed: card-candidate]" in worksheet
    # Both the uptake failure AND the payoff-graded used entry seed rows —
    # serving outcomes are claim evidence now; SERVE-NOISE stays out.
    assert worksheet.count("[seed: serving-feedback]") == 2
    assert "decision paid" in worksheet
    assert "irrelevant here" not in worksheet
    assert "[tension]" in worksheet and "a-card and b-card" in worksheet
    assert "[dup-merge]" in worksheet and "similarity nominates" in worksheet


def test_missing_verdict_fails_after_repair(tmp_path):
    # Regression: coverage arithmetic — a worksheet row without a journal
    # verdict bounces once with the named finding, then fails loud.
    def lazy_lead(workspace):
        good_lead(workspace)
        journal = Path(workspace) / "work" / "journal.md"
        text = journal.read_text()
        journal.write_text(
            "\n".join(l for l in text.splitlines() if "NOTE" not in l) + "\n"
        )
        return "done"

    frame, config, batch, lead = make_frame(tmp_path, [lazy_lead, lazy_lead])
    with pytest.raises(RuntimeError, match="has no journal verdict"):
        frame.run_update(
            batch, config["learning"]["update_crew"]["run_root"], "crew_v1"
        )
    assert any("has no journal verdict" in p for p in lead.prompts[1:])


def test_pass_without_rebuttal_fails(tmp_path):
    # Regression: a PASS without its surviving rebuttal is not a valid pass.
    def passing_lead(workspace):
        good_lead(workspace)
        journal = Path(workspace) / "work" / "journal.md"
        text = journal.read_text().replace(
            "→ NOTE** — uptake failure acknowledged for the closing "
            "assessment; serving-side, no bank edit.",
            "→ PASS** — nothing to do here.",
        )
        journal.write_text(text)
        return "done"

    frame, config, batch, _ = make_frame(tmp_path, [passing_lead, passing_lead])
    with pytest.raises(RuntimeError, match="rebuttal"):
        frame.run_update(
            batch, config["learning"]["update_crew"]["run_root"], "crew_v1"
        )


def test_inadmissible_evidence_fails(tmp_path):
    # Regression: the §5.2 admission runs inside the transaction — an
    # unresolvable ref in an appended entry rejects the run.
    def bad_evidence_lead(workspace):
        good_lead(workspace)
        card_path = Path(workspace) / "bank" / "insights" / "a-card.md"
        card_path.write_text(card_path.read_text().replace(
            "ref: runs/run_0001/metrics.json", "ref: runs/run_0099/ghost.json"
        ))
        return "done"

    frame, config, batch, _ = make_frame(
        tmp_path, [bad_evidence_lead, bad_evidence_lead]
    )
    with pytest.raises(RuntimeError, match="resolves in neither"):
        frame.run_update(
            batch, config["learning"]["update_crew"]["run_root"], "crew_v1"
        )


def test_init_bank_refuses_second_run(tmp_path):
    # Regression: the bank home is created once; re-init must not clobber.
    init_bank(str(tmp_path / "home.git"))
    with pytest.raises(FileExistsError):
        init_bank(str(tmp_path / "home.git"))


def test_previous_report_chains_within_the_invoked_run_root(tmp_path):
    # Regression (develop-run seam): batch N's staged previous_report must be
    # the latest report under the run root THIS run was invoked with (a
    # development run's own updates/ root), never the global config run_root.
    frame, config, batch, _ = make_frame(tmp_path, [good_lead])
    global_root = Path(config["learning"]["update_crew"]["run_root"])
    (global_root / "lr_99991231T235959").mkdir(parents=True)
    (global_root / "lr_99991231T235959" / "report.md").write_text("decoy")
    scoped_root = tmp_path / "develop" / "updates"
    prior = scoped_root / "lr_00000101T000000"
    prior.mkdir(parents=True)
    (prior / "report.md").write_text("batch-1 report")
    run_dir = frame.run_update(batch, str(scoped_root), "crew_v1")
    inputs = yaml.safe_load((run_dir / "inputs.yaml").read_text())
    assert inputs["previous_report"] == str(prior / "report.md")


def test_representation_flip_requires_a_green_run_in_transaction(tmp_path):
    # CD SS2 transaction rule: a text -> code flip without a green codify-run
    # verdict in the SAME transaction is rejected; with the green verdict and
    # the card artifacts (code/, replay/, entrypoint) it passes.
    from tests.test_codify_seeder import bare_bank, entry, log_row, write_proc

    def flipping_lead(workspace):
        run_dir = Path(workspace)
        good_lead(workspace)
        # journal the seeded codify docket row (coverage counts every row)
        worksheet = (run_dir / "work" / "observations.md").read_text()
        journal_path = run_dir / "work" / "journal.md"
        journal = journal_path.read_text()
        for line in worksheet.splitlines():
            if line.startswith("- **dk-") and "[codify]" in line:
                row_id = line.split("**")[1]
                journal += (f"- **{row_id} → CODIFY** — compatible; run "
                            f"folded this transaction. [flip-proc]\n")
        journal_path.write_text(journal)
        proc_dir = run_dir / "bank" / "procedures" / "flip-proc"
        text = (proc_dir / "card.md").read_text()
        text = text.replace("representation: text", "representation: code")
        text = text.replace("entrypoint: null", "entrypoint: run.py")
        text = text.replace("provenance: {version: 1}", "provenance: {version: 2}")
        # log stays append-only: the codify entry goes AFTER the founding one
        text = text.replace(
            "supersedes: null",
            "  - version: 2\n    date: 2026-08-18\n"
            "    commit: lr_flip\n    change: Codified.\nsupersedes: null",
            1,
        )
        (proc_dir / "card.md").write_text(text)
        (proc_dir / "code").mkdir(exist_ok=True)
        (proc_dir / "code" / "run.py").write_text("print('gate')\n")
        (proc_dir / "replay").mkdir(exist_ok=True)
        (proc_dir / "replay" / "eval.py").write_text("assert True\n")

    def seed_flip_bank(tmp_path):
        root = bare_bank(tmp_path / "fixture")
        write_proc(root, "flip-proc", [
            entry("rel-a--t/20260101T000000_lane-a"),
            entry("rel-b--t/20260102T000000_lane-b"),
        ])
        (root / "procedures" / "index.md").write_text(
            "# Procedures\n- [flip-proc](flip-proc/card.md) - fixture\n"
        )
        (root / "insights" / "a-card.md").write_text(card_text("a-card"))
        (root / "insights" / "index.md").write_text(
            "# Insights\n- [a-card](a-card.md) — hero\n"
        )
        home = tmp_path / "bank-home.git"
        init_bank(str(home))
        seed = tmp_path / "seed-flip"
        subprocess.run(["git", "clone", str(home), str(seed)],
                       check=True, capture_output=True)
        shutil.copytree(root, seed, dirs_exist_ok=True)
        subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(seed), "push", "origin", "main"],
                       check=True, capture_output=True)

    # without a green verdict: the flip is rejected after repair
    config = make_config(tmp_path)
    store = TrajectoryStore.from_config(config)
    seed_flip_bank(tmp_path)
    batch = make_batch(tmp_path, store)
    frame = UpdateFrame(store, config,
                        agent_factory=FakeFactory(FakeLead(
                            [flipping_lead, flipping_lead])))
    with pytest.raises(RuntimeError, match="no green run, no flip"):
        frame.run_update(batch, str(tmp_path / "runs-red"), "crew_v1")

    # with the green verdict in the transaction: the flip commits
    def green_flipping_lead(workspace):
        flipping_lead(workspace)
        run_dir = Path(workspace)
        verdict_dir = run_dir / "work" / "codify-runs" / "flip-proc"
        verdict_dir.mkdir(parents=True)
        (verdict_dir / "verdict.yaml").write_text(
            "status: green\niterations: 1\n"
        )

    config2 = make_config(tmp_path / "b")
    store2 = TrajectoryStore.from_config(config2)
    seed_flip_bank(tmp_path / "b")
    batch2 = make_batch(tmp_path / "b", store2)
    frame2 = UpdateFrame(store2, config2,
                         agent_factory=FakeFactory(FakeLead([green_flipping_lead])))
    run_dir = frame2.run_update(batch2, str(tmp_path / "runs-green"), "crew_v1")
    flipped = (run_dir / "bank" / "procedures" / "flip-proc" / "card.md").read_text()
    assert "representation: code" in flipped


def test_outcome_verdict_without_reassessment_is_rejected(tmp_path):
    # B6 live regression: an ATTACH that lands a confirm but freezes the
    # reliability block (no version bump) fails validation — settlements
    # must move or re-affirm the scores, never freeze them.
    def frozen_lead(workspace):
        run_dir = Path(workspace)
        worksheet = (run_dir / "work" / "observations.md").read_text()
        inputs = yaml.safe_load((run_dir / "inputs.yaml").read_text())
        journal = ["# Routing journal", ""]
        for line in worksheet.splitlines():
            if not line.startswith("- **"):
                continue
            row_id = line.split("**")[1]
            if "[seed: lift" in line:
                journal.append(
                    f"- **{row_id} \u2192 ATTACH** (fast-path) \u2014 lifted; "
                    f"delta copied. [mined/it-1/flow-1.md]"
                )
                card_path = run_dir / "bank" / "insights" / "a-card.md"
                text = card_path.read_text()
                append = EVIDENCE_APPEND.format(
                    lr_id=inputs["lr_id"],
                    trajectory=inputs["batch"][0]["trajectory"],
                )
                # evidence appended, reliability FROZEN - no bump, no log
                card_path.write_text(
                    text.replace("reliability:", append + "reliability:", 1)
                )
            elif "[seed: card-candidate]" in line:
                journal.append(
                    f"- **{row_id} \u2192 SIGHTING** \u2014 single observation. "
                    f"[mined/it-1/flow-1.md]"
                )
                sightings = run_dir / "bank" / "sightings.md"
                sightings.write_text(
                    sightings.read_text() + "- 2026-08-18 \u00b7 t \u00b7 x\n"
                )
            elif "[seed: serving-feedback]" in line:
                journal.append(f"- **{row_id} \u2192 NOTE** \u2014 serving-side.")
        (run_dir / "work" / "journal.md").write_text("\n".join(journal) + "\n")
        (run_dir / "work" / "headline.md").write_text("Frozen run.\n")
        (run_dir / "work" / "closing.md").write_text("n/a\n")
        (run_dir / "work" / "critic-findings.md").write_text(
            "- **F-01** [warn] [class: routing] none. Required: n/a\n"
        )

    frame, config, batch, _ = make_frame(tmp_path, [frozen_lead, frozen_lead, frozen_lead])
    with pytest.raises(RuntimeError, match="never reassessed"):
        frame.run_update(batch, str(tmp_path / "runs-frozen"), "crew_v2")
