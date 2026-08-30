# Gauntlet — the two-trap harness: fixtures, substance diff, trap runners,
# verdict assembly.
#
# Design: learn-from-trajectories-grader-scoring.md §2.3 (minimal battery).
# The traps buy an answer key by construction: run the learning step on
# controlled input and diff the result. The mechanical half is the
# duplicate-trap fixture (a clone under a fresh identity), the stability
# substance-diff (touched-card set, states, transitions, scores within
# tolerance; prose free to differ), and gauntlet.md assembly with per-trap
# {verdict, rationale} (no naked tags). GauntletRunner black-boxes the real
# update crew on sandbox bank homes seeded from a completed development run.

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.learning.bank import Bank
from kapso.learning.update_frame import UpdateFrame, _jaccard, _tokens

TRAPS = ("duplicate", "stability")
VERDICTS = ("PASS", "FAIL")


def build_duplicate_fixture(
    mined_view_dir: str, fixture_dir: str, clone_trajectory_id: str
) -> Path:
    """Clone a mined view under a fresh trajectory identity (same run ids,
    same numbers — the disguise rewording is the P4 runner's agent step; the
    identity clone is the mechanical base every disguise starts from)."""
    source = Path(mined_view_dir).expanduser()
    if not (source / "index.md").is_file():
        raise FileNotFoundError(f"{mined_view_dir} is not a mined view")
    target = Path(fixture_dir).expanduser() / clone_trajectory_id / "mined"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def substance_diff(
    bank_a_dir: str, bank_b_dir: str, tolerance: float
) -> List[str]:
    """The stability comparator: two banks must agree in SUBSTANCE — same
    card set, same states, versions, and scores within tolerance. Prose may
    differ freely (§2.3). Every disagreement is a named difference."""
    bank_a, bank_b = Bank(bank_a_dir), Bank(bank_b_dir)
    differences: List[str] = []
    only_a = sorted(set(bank_a.cards) - set(bank_b.cards))
    only_b = sorted(set(bank_b.cards) - set(bank_a.cards))
    for name in only_a:
        differences.append(f"card {name} exists only in run A")
    for name in only_b:
        differences.append(f"card {name} exists only in run B")
    for name in sorted(set(bank_a.cards) & set(bank_b.cards)):
        card_a, card_b = bank_a.cards[name], bank_b.cards[name]
        if card_a.state != card_b.state:
            differences.append(
                f"{name}: state {card_a.state!r} (A) vs {card_b.state!r} (B)"
            )
        version_a = (card_a.frontmatter.get("provenance") or {}).get("version")
        version_b = (card_b.frontmatter.get("provenance") or {}).get("version")
        if version_a != version_b:
            differences.append(f"{name}: version {version_a} (A) vs {version_b} (B)")
        score_a, score_b = card_a.score, card_b.score
        if (score_a is None) != (score_b is None):
            differences.append(f"{name}: score present in one run only")
        elif score_a is not None and abs(score_a - score_b) > tolerance:
            differences.append(
                f"{name}: score {score_a} (A) vs {score_b} (B) exceeds "
                f"tolerance {tolerance}"
            )
    return differences


def assemble_gauntlet(
    trap_results: Dict[str, Dict[str, str]],
    context: Dict[str, Any],
) -> str:
    """gauntlet.md: frontmatter with per-trap {verdict, rationale} and the
    rolled verdict (any FAIL ⇒ FAIL); body sections carry construction +
    proof refs. Rationales are mandatory — no naked tags."""
    for trap, result in trap_results.items():
        if trap not in TRAPS:
            raise ValueError(f"unknown trap {trap!r}: expected one of {TRAPS}")
        if result.get("verdict") not in VERDICTS:
            raise ValueError(f"trap {trap!r} verdict must be PASS or FAIL")
        if not str(result.get("rationale", "")).strip():
            raise ValueError(f"trap {trap!r} has no rationale — no naked tags")
    rolled = "FAIL" if any(
        r["verdict"] == "FAIL" for r in trap_results.values()
    ) else "PASS"
    rolled_rationale = context.get("rolled_rationale", "")
    if not str(rolled_rationale).strip():
        raise ValueError("rolled verdict has no rationale — no naked tags")

    frontmatter = {
        "learner_version": context["learner_version"],
        "bank_head": context["bank_head"],
        "batch": context.get("batch", []),
        "gauntlet": trap_results,
        "verdict": rolled,
        "rationale": rolled_rationale,
    }
    body_sections = [
        f"## {trap} — construction + proof\n{context.get(f'{trap}_proof', 'n/a')}"
        for trap in trap_results
    ]
    return (
        "---\n" + yaml.safe_dump(frontmatter, sort_keys=False)
        + "---\n\n" + "\n\n".join(body_sections) + "\n"
    )


class GauntletRunner:
    """Black-box the real update crew on inputs whose correct transaction is
    known by construction (§2.3), against a completed development run's
    artifacts.

    Stability: the last learn batch re-run from its recorded pre-batch head
    on a disposable home — two independent crew sessions over identical
    inputs must agree in substance.

    Duplicate: an already-ingested trajectory's mined view cloned under a
    trap identity, its admitted exam report retargeted, run against the
    final bank — every lesson is already carded, so a spawned card lexically
    similar to an existing card is the crew double-carding what it knows.
    """

    def __init__(self, store, config: Dict[str, Any], agent_factory=CodingAgentFactory):
        self.store = store
        self.config = config
        learning = config["learning"]
        self.develop_root = Path(learning["develop"]["run_root"]).expanduser()
        self.tolerance = learning["graders"]["gauntlet"]["stability_tolerance"]
        self.dup_threshold = learning["update_crew"]["dup_nominate_jaccard"]
        self.agent_factory = agent_factory

    def run(self, learner_version: str) -> str:
        """Both traps against <develop run_root>/<learner_version>; writes
        gauntlet.md at the run root and returns the rolled verdict."""
        root = self.develop_root / learner_version
        if not (root / "training-curve.yaml").is_file():
            raise FileNotFoundError(
                f"{root} is not a completed development run (no training curve)"
            )
        scratch = root / "gauntlet"
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir()

        stability = self._run_stability(root, learner_version)
        duplicate = self._run_duplicate(root, learner_version)
        trap_results = {
            "duplicate": {k: duplicate[k] for k in ("verdict", "rationale")},
            "stability": {k: stability[k] for k in ("verdict", "rationale")},
        }
        rolled = "FAIL" if any(
            r["verdict"] == "FAIL" for r in trap_results.values()
        ) else "PASS"
        text = assemble_gauntlet(trap_results, {
            "learner_version": learner_version,
            "bank_head": self._rev(root / "bank-home.git", "HEAD"),
            "batch": stability["batch"],
            "rolled_rationale": (
                "Both traps ran the real crew on sandbox homes seeded from "
                "this development run; any FAIL rolls up."
            ),
            "duplicate_proof": duplicate["proof"],
            "stability_proof": stability["proof"],
        })
        (root / "gauntlet.md").write_text(text)
        return rolled

    # ------------------------------------------------------------ stability

    def _run_stability(self, root: Path, learner_version: str) -> Dict[str, Any]:
        batches = self._batches(root)
        last = len(batches) - 1
        batch_ids = batches[last]
        reports = [self._exam_report(root, last, t) for t in batch_ids]
        # The pre-batch head is git truth, never report text: each update run
        # is one commit tagged lr_<id>, so the batch's parent commit is the
        # bank state the batch started from.
        head = self._rev(
            root / "bank-home.git",
            sorted((root / "updates").glob("lr_*"))[last].name + "^",
        )
        home = self._seed_home(root, "stability-home.git", head)
        run_root = root / "gauntlet" / "stability-run"
        originals = sorted((root / "updates").glob("lr_*"))
        if len(originals) != len(batches):
            raise ValueError(
                f"{len(originals)} update runs for {len(batches)} batches — "
                f"the development run is not one-update-per-batch"
            )
        # Input parity: the rerun must see the same previous-report chain the
        # original batch saw (content parity; the path may differ).
        for prior in originals[:last]:
            seed = run_root / prior.name
            seed.mkdir(parents=True)
            shutil.copy2(prior / "report.md", seed / "report.md")
        batch = [
            {"trajectory": t, "hindcast_report": str(r)}
            for t, r in zip(batch_ids, reports)
        ]
        rerun_dir = self._scoped_frame(home).run_update(
            batch, str(run_root), learner_version
        )
        differences = substance_diff(
            str(originals[last] / "bank"), str(rerun_dir / "bank"), self.tolerance
        )
        verdict = "PASS" if not differences else "FAIL"
        rationale = (
            f"batch {last} re-run from head {head} agreed in substance with "
            f"the original transaction (tolerance {self.tolerance})"
            if not differences else
            f"batch {last} re-run diverged in substance: "
            + "; ".join(differences)
        )
        proof = (
            f"- inputs: batch {last} = {batch_ids}, pre-batch head {head}\n"
            f"- run A: {originals[last]}\n- run B: {rerun_dir}\n"
            f"- substance_diff: "
            + ("empty (states, versions, scores agree)" if not differences
               else "\n".join(f"  - {d}" for d in differences))
        )
        return {"verdict": verdict, "rationale": rationale, "proof": proof,
                "batch": batch_ids}

    # ------------------------------------------------------------ duplicate

    def _run_duplicate(self, root: Path, learner_version: str) -> Dict[str, Any]:
        batches = self._batches(root)
        source_id = batches[0][0]
        family_task = source_id.split("/")[0]
        trap_id = f"{family_task}/20260818T235900_lane-trap"
        source_mined = self.store.local / source_id / "mined"
        build_duplicate_fixture(str(source_mined), str(self.store.local), trap_id)
        # The trap needs a manifest so the crew's evidence attachments citing
        # it pass admission (source-resolution gate): the source manifest with
        # the identity rewritten. Written last (manifest-last discipline);
        # removed with the clone once the trap has judged.
        manifest_text = (
            self.store.local / source_id / "trajectory.yaml"
        ).read_text()
        id_line = f"id: {source_id}"
        if not manifest_text.startswith(id_line):
            raise ValueError(f"manifest for {source_id} does not open with its id")
        (self.store.local / trap_id / "trajectory.yaml").write_text(
            manifest_text.replace(id_line, f"id: {trap_id}", 1)
        )

        source_report_path = self._exam_report(root, 0, source_id)
        source_report = source_report_path.read_text()
        needle = f"trajectory: {source_id}"
        if source_report.count(needle) != 1:
            raise ValueError(
                f"exam report for {source_id} does not carry exactly one "
                f"frontmatter trajectory line"
            )
        trap_report = root / "gauntlet" / "duplicate-report.md"
        trap_report.write_text(
            source_report.replace(needle, f"trajectory: {trap_id}")
        )
        source_record = source_report_path.parent / "serving-record.yaml"
        if source_record.is_file():
            shutil.copy2(
                source_record, trap_report.parent / "serving-record.yaml"
            )

        home = self._seed_home(root, "duplicate-home.git", None)
        run_root = root / "gauntlet" / "duplicate-run"
        originals = sorted((root / "updates").glob("lr_*"))
        seed = run_root / originals[-1].name
        seed.mkdir(parents=True)
        shutil.copy2(originals[-1] / "report.md", seed / "report.md")
        run_dir = self._scoped_frame(home).run_update(
            [{"trajectory": trap_id, "hindcast_report": str(trap_report)}],
            str(run_root), learner_version,
        )

        before = Bank(str(run_dir / "bank-before"))
        after = Bank(str(run_dir / "bank"))
        spawned = sorted(set(after.cards) - set(before.cards))
        offenders = []
        for name in spawned:
            card = after.cards[name]
            tokens = _tokens(card.hero + " " + card.body)
            for existing_name, existing in before.cards.items():
                if existing.type != card.type:
                    continue
                similarity = _jaccard(
                    tokens, _tokens(existing.hero + " " + existing.body)
                )
                if similarity >= self.dup_threshold:
                    offenders.append((name, existing_name, similarity))
        attach_count = sum(
            1
            for card in after.cards.values()
            for entry in card.evidence
            if trap_id in str(entry.get("source", ""))
        )
        shutil.rmtree(self.store.local / trap_id)

        verdict = "PASS" if not offenders else "FAIL"
        rationale = (
            f"every lesson in the trap batch was already carded; the crew "
            f"spawned {len(spawned)} card(s), none lexically twinning an "
            f"existing card (threshold {self.dup_threshold}); "
            f"{attach_count} evidence attachment(s) cite the trap identity"
            if not offenders else
            "the crew double-carded known lessons: "
            + "; ".join(
                f"{new} twins {old} (similarity {sim:.2f})"
                for new, old, sim in offenders
            )
        )
        proof = (
            f"- construction: {source_id} mined view cloned as {trap_id}, its "
            f"admitted exam report retargeted, run against the final bank\n"
            f"- run: {run_dir}\n- spawned: {spawned or 'none'}\n"
            f"- attachments citing the trap: {attach_count}\n"
            f"- offenders: "
            + ("none" if not offenders else "; ".join(
                f"{n} ~ {o} ({s:.2f})" for n, o, s in offenders
            ))
        )
        return {"verdict": verdict, "rationale": rationale, "proof": proof}

    # -------------------------------------------------------------- helpers

    def _batches(self, root: Path) -> List[List[str]]:
        curve = yaml.safe_load((root / "training-curve.yaml").read_text())
        batches: List[List[str]] = []
        for row in curve:
            while len(batches) <= row["batch"]:
                batches.append([])
            batches[row["batch"]].append(row["trajectory"])
        if not batches or not all(batches):
            raise ValueError(f"training curve at {root} has empty batches")
        return batches

    def _exam_report(self, root: Path, batch_index: int, trajectory_id: str) -> Path:
        slug = trajectory_id.replace("/", "--")
        matches = list(
            (root / "exams" / f"batch-{batch_index:02d}").glob(
                f"exam-*/hindcast/{slug}/report.md"
            )
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one admitted report for {trajectory_id} in "
                f"batch {batch_index}, found {len(matches)}"
            )
        return matches[0]

    def _rev(self, home: Path, revision: str) -> str:
        return subprocess.run(
            ["git", "--git-dir", str(home), "rev-parse", revision],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def _seed_home(self, root: Path, name: str, head) -> Path:
        """A disposable bare home cloned from the run's home, with main reset
        to the given head (None keeps the final head)."""
        home = root / "gauntlet" / name
        # --no-tags: the disposable home must not inherit the run's lr_ tags —
        # the trap's own transaction mints its tag there.
        subprocess.run(
            ["git", "clone", "--bare", "--no-tags",
             str(root / "bank-home.git"), str(home)],
            check=True, capture_output=True,
        )
        if head is not None:
            subprocess.run(
                ["git", "--git-dir", str(home), "update-ref",
                 "refs/heads/main", head],
                check=True, capture_output=True,
            )
        return home

    def _scoped_frame(self, bank_home: Path) -> UpdateFrame:
        scoped = {
            **self.config,
            "learning": {
                **self.config["learning"],
                "bank": {"local_path": str(bank_home)},
            },
        }
        return UpdateFrame(self.store, scoped, agent_factory=self.agent_factory)
