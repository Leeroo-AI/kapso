# Codify seeder tests (P7.1, CD§1): the frame's layer-1 arithmetic — executed
# verdicts only (a mention never counts), closure through founding references
# with set-semantics dedup, the failed-attempt guard, and the worksheet row
# (Rule 9: each is a wrong-machine-spins-up regression).

from pathlib import Path

from kapso.learning.bank import Bank
from kapso.learning.trajectory_store import TrajectoryStore
from kapso.learning.update_frame import UpdateFrame
from tests.test_update_frame import make_config

PROC_PLAIN = ("promising — executed in 2 campaigns on 2 datasets; no "
              "counter-evidence; untested elsewhere.")

PROC_TMPL = """# {title}: decide candidates on paired deltas

**Rule:** Never accept or reject a fresh candidate on its raw validation
score; decide on the paired, cluster-resampled delta against the incumbent
so shared fold noise cancels out of the decision.

## Is this your situation?

- You have a fresh candidate model and an incumbent champion on the same
  validation split.
- Their scores differ by less than the split's noise.
- You must decide right now whether the candidate ships or dies.

## What to do

1. Score both models on the identical fold set.
2. Compute the paired clustered-bootstrap delta with its standard error.
3. Accept the candidate only when the delta clears the significance gate.
4. Otherwise keep the incumbent unchanged.

## Why believe this

Paired deltas cancel shared fold variance that raw score comparisons
cannot, so the decision reads the models' true difference instead of the
split's noise. In our runs, teams that skipped the pairing routinely
shipped regressions that looked like wins on a single noisy validation
read, while the paired gate kept every such flip out.

**Confidence:** {plain}

---

```yaml
type: procedure
tags: []
timestamp: 2026-08-18T09:00:00Z
scope: domain
scope_conditions: "any"
representation: text
entrypoint: null
evidence:
{evidence}reliability:
  validity: 0.85
  boundary: 0.5
  coverage: 0.4
  score: 0.65
  plain: {plain}
  rationale: >-
    Fixture ledger.
  state: {state}
provenance: {{version: {version}}}
log:
{log}supersedes: null
contradicts: {contradicts}
probe: >-
  Re-run the gate on one fold.
```
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
        version=version, plain=PROC_PLAIN,
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
        plain=PROC_PLAIN,
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


def seed_worksheet_for(tmp_path, root):
    config = make_config(tmp_path)
    frame = UpdateFrame(TrajectoryStore.from_config(config), config)
    run_dir = tmp_path / "run"
    (run_dir / "work").mkdir(parents=True)
    frame._seed_worksheet(run_dir, [], Bank(str(root)))
    return (run_dir / "work" / "observations.md").read_text()


def test_sighting_expiry_row_after_configured_batches(tmp_path):
    # An aged sighting (enough log entries newer than it) seeds an expiry
    # row; a fresh sighting stays quiet.
    root = bare_bank(tmp_path)
    (root / "sightings.md").write_text(
        "# Sightings\n"
        "- 2026-01-01 \u00b7 rel-a--t/20260101T000000_lane-a \u00b7 old lesson\n"
        "- 2026-08-18 \u00b7 rel-b--t/20260818T000000_lane-b \u00b7 new lesson\n"
    )
    (root / "log.md").write_text(
        "# Log\n" + "".join(
            f"- lr_2026061{i}T000000 \u2014 batch {i}\n" for i in range(7)
        )
    )
    worksheet = seed_worksheet_for(tmp_path, root)
    assert "old lesson" in worksheet and "aged 7 batches" in worksheet
    assert "new lesson" not in worksheet


def test_code_freshness_rows(tmp_path):
    # CD SS4: an unstamped code card and a stale one seed expiry rows; a
    # freshly replayed one stays quiet.
    root = bare_bank(tmp_path)
    write_proc(root, "unstamped-code", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-b--t/20260102T000000_lane-b"),
    ])
    write_proc(root, "stale-code", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-b--t/20260102T000000_lane-b"),
    ])
    write_proc(root, "fresh-code", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-b--t/20260102T000000_lane-b"),
    ])
    for name, stamp in (("unstamped-code", None), ("stale-code", "2026-01-01"),
                        ("fresh-code", "2026-08-17")):
        path = root / "procedures" / name / "card.md"
        text = path.read_text().replace(
            "representation: text", "representation: code"
        )
        if stamp:
            text = text.replace(
                "entrypoint: null", f"entrypoint: run.py\nlast_replayed: {stamp}"
            )
        else:
            text = text.replace("entrypoint: null", "entrypoint: run.py")
        path.write_text(text)
    worksheet = seed_worksheet_for(tmp_path, root)
    assert "unstamped-code carries no last_replayed stamp" in worksheet
    assert "stale-code last replayed" in worksheet
    assert "fresh-code" not in "".join(
        l for l in worksheet.splitlines() if "expiry" in l
    )


def test_green_codify_run_stages_flip_and_suppresses_renomination(tmp_path):
    # CD§2 plumbing: the newest green run's verdict + artifacts land in
    # work/codify-runs/<card>/, the seeder emits the flip row instead of
    # re-nominating, and non-green runs stage nothing.
    root = bare_bank(tmp_path)
    write_proc(root, "ready-proc", [
        entry("rel-a--t/20260101T000000_lane-a"),
        entry("rel-b--t/20260102T000000_lane-b"),
    ])
    config = make_config(tmp_path)
    codify_root = Path(config["learning"]["update_crew"]["run_root"]) / "codify"
    green = codify_root / "ready-proc-20260820T000000"
    (green / "workspace" / "code").mkdir(parents=True)
    (green / "workspace" / "code" / "main.py").write_text("x = 1\n")
    (green / "workspace" / "replay").mkdir()
    (green / "workspace" / "replay" / "eval.py").write_text("ok = True\n")
    (green / "verdict.yaml").write_text("status: green\niterations: 1\n")
    red = codify_root / "other-proc-20260820T000000"
    red.mkdir(parents=True)
    (red / "verdict.yaml").write_text("status: failed\niterations: 3\n")

    frame = UpdateFrame(TrajectoryStore.from_config(config), config)
    run_dir = tmp_path / "run"
    (run_dir / "work").mkdir(parents=True)
    bank = Bank(str(root))
    frame._stage_codify_runs(run_dir, bank)
    frame._seed_worksheet(run_dir, [], bank)

    staged = run_dir / "work" / "codify-runs" / "ready-proc"
    assert (staged / "verdict.yaml").is_file()
    assert (staged / "code" / "main.py").is_file()
    assert (staged / "replay" / "eval.py").is_file()
    assert not (run_dir / "work" / "codify-runs" / "other-proc").exists()
    worksheet = (run_dir / "work" / "observations.md").read_text()
    assert "commit the representation flip" in worksheet
    assert not any(
        "executed source campaigns" in line
        for line in worksheet.splitlines() if "ready-proc" in line
    )
