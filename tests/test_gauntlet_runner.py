# Gauntlet-runner tests (P4): the traps black-box a fake crew against a
# synthetic completed development run. The PASS path rides the driver e2e
# (test_develop_driver); these pin the FAIL sides — a spawn twinning an
# existing card fails the duplicate trap, and a rerun that diverges in
# substance fails stability with the difference named (Rule 9: the traps are
# the crew's answer-key regressions; their own failure detection must hold).

import subprocess
from pathlib import Path

import yaml

from kapso.learning.graders.gauntlet import GauntletRunner
from kapso.learning.trajectory_store import TrajectoryStore
from tests.test_develop_driver import develop_config, seed_mined_trajectory
from tests.test_update_frame import EVIDENCE_APPEND, card_text, seed_bank_home

TRAJ_ID = "rel-amazon--user-churn/20260101T000000_lane-t1"

REPORT = """---
trajectory: {trajectory}
bank_head: {head}
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

# TWIN_BODY lexically twins card_text's TEMPLATE_BODY (the duplicate
# trap's Jaccard bait); both bodies follow the format-v2 template.
TWIN_PLAIN = ("tentative — one observation in one campaign; no "
              "counter-evidence; untested elsewhere.")
TWIN_BODY = (
    "# Prefer group-relative signals when rows compete in a pool\n\n"
    "**Rule:** When ranked rows compete inside a shared pool, build "
    "group-relative features (within-group percentiles and z-scores), not "
    "absolute values — the relative view carries the ordering signal.\n\n"
    "## Is this your situation?\n\n"
    "- You are ranking rows that compete inside a shared pool.\n"
    "- Your model consumes absolute per-row features today.\n"
    "- You are choosing which normalization block to build next before "
    "training the next candidate.\n\n"
    "## What to do\n\n"
    "1. Group rows by their competing pool.\n"
    "2. Compute each feature's percentile or z-score within its group.\n"
    "3. Feed those transforms alongside the raw columns.\n"
    "4. Gate the block with your paired significance check before "
    "shipping the change into the current best model.\n\n"
    "## Why believe this\n\n"
    "Competition happens within the pool rather than across it, so "
    "absolute magnitudes mislead exactly when they look informative. In "
    "our runs the relative block separated competing rows at the margin "
    "where the absolute view ranked them identically.\n\n"
    f"**Confidence:** {TWIN_PLAIN}"
)
FRESH_PLAIN = ("tentative — one observation in one campaign; no "
               "counter-evidence; untested elsewhere.")
FRESH_BODY = (
    "# Cache the feature matrix by schema fingerprint\n\n"
    "**Rule:** When the matrix build dominates loop time and its schema "
    "changes rarely, cache the built matrix and version the cache by a "
    "schema fingerprint — rebuild only when the fingerprint moves.\n\n"
    "## Is this your situation?\n\n"
    "- You rebuild the full feature matrix on every search iteration.\n"
    "- The build dominates your loop time.\n"
    "- The matrix schema changes far less often than the models "
    "consuming it.\n\n"
    "## What to do\n\n"
    "1. Fingerprint the matrix schema (columns, dtypes, windows).\n"
    "2. Cache the built matrix keyed by that fingerprint.\n"
    "3. Rebuild only when the fingerprint changes; otherwise load from "
    "disk.\n"
    "4. Let model-side iterations reuse the cached matrix directly.\n\n"
    "## Why believe this\n\n"
    "Fingerprint versioning invalidates the cache exactly when the "
    "features truly changed rather than on every loop, so staleness risk "
    "stays zero while the search spends its budget on models instead of "
    "rebuilds. In our runs the loop time dropped by the full build cost "
    "on every unchanged-schema iteration.\n\n"
    f"**Confidence:** {FRESH_PLAIN}"
)

SPAWN_CARD = """{body}

---

```yaml
type: insight
tags: []
timestamp: 2026-08-18T09:00:00Z
scope: domain
scope_conditions: "any run"
evidence:
  - source:
      learner_run: {lr_id}
      trajectory: {trajectory}
      ref: mined/index.md
      card_version: null
    verdict: exercise
    usage: >-
      Observed independently — never served in that campaign.
    effect: >-
      Recurring pattern noted; no measured delta claimed.
reliability:
  validity: 0.5
  boundary: 0.4
  coverage: 0.2
  score: 0.5
  plain: {plain}
  rationale: >-
    Single-batch observation; untested elsewhere.
  state: candidate
provenance: {{version: 1}}
log:
  - version: 1
    date: 2026-08-18
    commit: {lr_id}
    change: Spawned by the trap-test fake.
supersedes: null
contradicts: []
```
"""


class SpawningLeadFake:
    """An update lead that answers every seeded row but SPAWNS on the
    card-candidate — with a twin of a-card (double-carding) or a genuinely
    fresh card, depending on `twin`."""

    def __init__(self, twin):
        self.twin = twin
        self.cwd = None

    def initialize(self, workspace):
        self.cwd = workspace

    def generate_code(self, prompt, debug_mode=False, timeout_seconds=None):
        if "knowledge-update crew" not in prompt:
            raise AssertionError("the gauntlet spawns update leads only")
        run_dir = Path(self.cwd)
        worksheet = (run_dir / "work" / "observations.md").read_text()
        inputs = yaml.safe_load((run_dir / "inputs.yaml").read_text())
        trajectory = inputs["batch"][0]["trajectory"]
        journal = ["# Routing journal", ""]
        for line in worksheet.splitlines():
            if not line.startswith("- **"):
                continue
            row_id = line.split("**")[1]
            if "[seed: lift" in line:
                journal.append(
                    f"- **{row_id} → ATTACH** — settlement lifted onto "
                    f"a-card. [mined/it-1/flow-1.md]"
                )
                card_path = run_dir / "bank" / "insights" / "a-card.md"
                text = card_path.read_text().replace(
                    "reliability:",
                    EVIDENCE_APPEND.format(
                        lr_id=inputs["lr_id"], trajectory=trajectory
                    ) + "reliability:",
                    1,
                )
                text = text.replace(
                    "Validity from two confirmations; boundary untested; "
                    "coverage thin.",
                    "Reassessed after the lifted settlement: in-scope "
                    "confirm holds validity; boundary still untested.",
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
                    f"- **{row_id} → SPAWN** — carded as b-card. "
                    f"[mined/it-1/flow-1.md]"
                )
                body = TWIN_BODY if self.twin else FRESH_BODY
                plain = TWIN_PLAIN if self.twin else FRESH_PLAIN
                (run_dir / "bank" / "insights" / "b-card.md").write_text(
                    SPAWN_CARD.format(
                        body=body, plain=plain,
                        lr_id=inputs["lr_id"], trajectory=trajectory,
                    )
                )
                index = run_dir / "bank" / "insights" / "index.md"
                index.write_text(
                    index.read_text() + "- [b-card](b-card.md) — hero\n"
                )
            elif "[seed: serving-feedback]" in line:
                journal.append(
                    f"- **{row_id} → NOTE** — uptake failure acknowledged; "
                    f"serving-side, no bank edit."
                )
        (run_dir / "work" / "journal.md").write_text("\n".join(journal) + "\n")
        (run_dir / "work" / "headline.md").write_text("Trap-test run.\n")
        (run_dir / "work" / "closing.md").write_text("Nothing further.\n")
        (run_dir / "work" / "critic-findings.md").write_text(
            "- **F-01** [warn] [class: routing] none — clean pass. Required: n/a\n"
        )

        class Result:
            success = True
            output = "done"

        return Result()


class SpawningFactory:
    def __init__(self, twin):
        self.twin = twin

    def create(self, config):
        return SpawningLeadFake(self.twin)


def synthetic_develop_root(tmp_path):
    """A minimal completed development run: one learn batch, one exam report,
    one update run whose after-bank equals the founding state, a-card home."""
    config = develop_config(tmp_path)
    store = TrajectoryStore.from_config(config)
    seed_mined_trajectory(store, tmp_path / "seed", TRAJ_ID)

    root = Path(config["learning"]["develop"]["run_root"]) / "crew_g1"
    root.mkdir(parents=True)
    home = seed_bank_home(root, {"a-card": card_text("a-card")})
    # Stand in for the batch's update run: one (empty) commit tagged lr_<id>,
    # so the runner's pre-batch head (tag^) is the founding a-card state.
    seed = root / "seed-clone"
    subprocess.run(
        ["git", "-C", str(seed), "commit", "--allow-empty", "-m", "update"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"],
                   check=True, capture_output=True)
    subprocess.run(
        ["git", "--git-dir", str(home), "tag", "lr_00000101T000000", "HEAD"],
        check=True, capture_output=True,
    )
    head = subprocess.run(
        ["git", "--git-dir", str(home), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    slot = root / "exams" / "batch-00" / "exam-0" / "hindcast" / TRAJ_ID.replace("/", "--")
    slot.mkdir(parents=True)
    (slot / "report.md").write_text(REPORT.format(trajectory=TRAJ_ID, head=head))
    with open(slot / "serving-record.yaml", "w") as handle:
        yaml.safe_dump({"bank_head": head, "served": []}, handle)

    original = root / "updates" / "lr_00000101T000000"
    original.mkdir(parents=True)
    (original / "report.md").write_text("prior report\n")
    subprocess.run(
        ["git", "clone", str(home), str(original / "bank")],
        check=True, capture_output=True,
    )
    with open(root / "training-curve.yaml", "w") as handle:
        yaml.safe_dump(
            [{"batch": 0, "trajectory": TRAJ_ID, "hindcast": {}}], handle
        )
    return config, store


def test_duplicate_trap_fails_double_carding(tmp_path):
    # The trap batch is a clone of an already-carded trajectory; a spawn
    # twinning a-card is the exact failure the trap exists to catch.
    config, store = synthetic_develop_root(tmp_path)
    runner = GauntletRunner(store, config, agent_factory=SpawningFactory(twin=True))
    assert runner.run("crew_g1") == "FAIL"
    root = Path(config["learning"]["develop"]["run_root"]) / "crew_g1"
    gauntlet = yaml.safe_load((root / "gauntlet.md").read_text().split("---")[1])
    assert gauntlet["gauntlet"]["duplicate"]["verdict"] == "FAIL"
    assert "twins" in gauntlet["gauntlet"]["duplicate"]["rationale"]
    # the trap clone was cleaned out of the store
    assert not list(store.local.glob("*/*_lane-trap"))


def test_stability_trap_names_substance_divergence(tmp_path):
    # A crew whose rerun spawns a card the original run did not is unstable;
    # the fresh card is dissimilar, so duplicate stays PASS and the FAIL is
    # attributed to stability with the divergence named.
    config, store = synthetic_develop_root(tmp_path)
    runner = GauntletRunner(store, config, agent_factory=SpawningFactory(twin=False))
    assert runner.run("crew_g1") == "FAIL"
    root = Path(config["learning"]["develop"]["run_root"]) / "crew_g1"
    gauntlet = yaml.safe_load((root / "gauntlet.md").read_text().split("---")[1])
    assert gauntlet["gauntlet"]["duplicate"]["verdict"] == "PASS"
    assert gauntlet["gauntlet"]["stability"]["verdict"] == "FAIL"
    assert "only in run B" in gauntlet["gauntlet"]["stability"]["rationale"]
