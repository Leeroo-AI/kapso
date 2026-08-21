# Bank transaction validation tests (P4): each invariant class trips exactly
# (Rule 9 — the regression is a corrupted bank state admitted silently).

import shutil

import pytest
import yaml

from kapso.learning.bank import Bank
from kapso.learning.bank_invariants import (
    BankTransactionValidator,
    evidence_admission_findings,
)
from kapso.learning.trajectory_store import TrajectoryStore, save_trajectory
from tests.test_bank_retriever import build_bank, card_text
from tests.test_trajectory_store import TRAJECTORY_ID, build_work_dir


def clone_bank(tmp_path, cards, mutate=None):
    """Build `before`, copy to `after`, apply mutate(after_root)."""
    before_root = build_bank(tmp_path / "before", cards)
    after_root = tmp_path / "after" / "bank"
    shutil.copytree(before_root, after_root)
    if mutate:
        mutate(after_root)
    return Bank(str(before_root)), Bank(str(after_root))


def read_card(root, name):
    return (root / "insights" / f"{name}.md").read_text()


def write_card(root, name, text):
    (root / "insights" / f"{name}.md").write_text(text)


def test_clean_transaction_has_no_findings(tmp_path):
    before, after = clone_bank(tmp_path, {"a-card": card_text("a-card")})
    assert BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate() == []


def test_body_contract_names_missing_reader_sections(tmp_path):
    # The reader contract (2026-08-21): a body without the three engineer-
    # facing intents is an unfinished card and must trip by name.
    def mutate(root):
        text = read_card(root, "a-card")
        _, delim, ledger = text.partition("\n---\n")
        write_card(root, "a-card", "A mechanism paragraph only." + delim + ledger)

    before, after = clone_bank(tmp_path, {"a-card": card_text("a-card")}, mutate)
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any(
        "card-template gaps" in f and "Is this your situation?" in f
        for f in findings
    )


def test_evidence_append_only(tmp_path):
    # Regression: rewriting an existing evidence entry must trip.
    def mutate(root):
        text = read_card(root, "a-card").replace(
            "effect: KEPT at +0.0032", "effect: KEPT at +0.9999"
        )
        write_card(root, "a-card", text)

    before, after = clone_bank(tmp_path, {"a-card": card_text("a-card")}, mutate)
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("evidence is append-only" in f for f in findings)


def test_claim_change_requires_version_and_log(tmp_path):
    # Regression: a scope edit without a version bump + one log entry trips
    # both ways (bump without change; change without bump).
    def scope_edit_no_bump(root):
        text = read_card(root, "a-card").replace(
            "scope: domain", "scope: [family:entity_binary_classification]"
        )
        write_card(root, "a-card", text)

    before, after = clone_bank(
        tmp_path / "x", {"a-card": card_text("a-card")}, scope_edit_no_bump
    )
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("claim-layer change without exactly one version bump" in f
               for f in findings)
    assert any("exactly one log entry" in f for f in findings)

    def bump_no_change(root):
        text = read_card(root, "a-card").replace(
            "provenance: {version: 1}", "provenance: {version: 2}"
        )
        write_card(root, "a-card", text)

    before, after = clone_bank(
        tmp_path / "y", {"a-card": card_text("a-card")}, bump_no_change
    )
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("version bumped without a claim-layer change" in f for f in findings)


def test_retirement_is_a_move(tmp_path):
    # Regression: deleting a card outright must trip; moving it to retired/
    # must not.
    def delete(root):
        (root / "insights" / "a-card.md").unlink()

    before, after = clone_bank(
        tmp_path / "x",
        {"a-card": card_text("a-card"), "b-card": card_text("b-card")},
        delete,
    )
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("without a move to retired/" in f for f in findings)

    def move(root):
        (root / "retired" / "insights").mkdir(parents=True)
        text = read_card(root, "a-card").replace("state: active", "state: retired")
        (root / "retired" / "insights" / "a-card.md").write_text(text)
        (root / "insights" / "a-card.md").unlink()

    before, after = clone_bank(
        tmp_path / "y",
        {"a-card": card_text("a-card"), "b-card": card_text("b-card")},
        move,
    )
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert not any("retired" in f for f in findings)


def test_decoys_untouchable(tmp_path):
    # Regression: any credit or edit landing on a decoy trips (the standing
    # decoy invariant — rung 1's per-commit audit).
    def setup_and_touch(root):
        text = read_card(root, "decoy-card").replace("score: 0.7", "score: 0.9")
        write_card(root, "decoy-card", text)

    def build(tmp):
        before_root = build_bank(tmp / "before", {
            "decoy-card": card_text("decoy-card"),
            "real-card": card_text("real-card"),
        })
        (before_root / ".decoys.yaml").write_text("- decoy-card\n")
        after_root = tmp / "after" / "bank"
        shutil.copytree(before_root, after_root)
        setup_and_touch(after_root)
        return Bank(str(before_root)), Bank(str(after_root))

    before, after = build(tmp_path)
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("decoy decoy-card: modified" in f for f in findings)


def test_contradicts_lands_on_both_cards(tmp_path):
    # Regression: a one-sided contradicts edge trips.
    def one_sided(root):
        text = read_card(root, "a-card").replace(
            "contradicts: []", "contradicts: [b-card]"
        ).replace("provenance: {version: 1}", "provenance: {version: 2}")
        text = text.replace(
            "log:", "log:\n  - {version: 2, date: 2026-08-15, commit: lr_2, change: edge}",
            1,
        )
        write_card(root, "a-card", text)

    before, after = clone_bank(
        tmp_path, {"a-card": card_text("a-card"), "b-card": card_text("b-card")},
        one_sided,
    )
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("but not vice versa" in f for f in findings)


def test_merge_shape(tmp_path):
    # Regression: a merge successor with copied (unreferenced) evidence, an
    # unretired parent, or a missing forward link trips.
    def merge(root):
        successor = card_text("merged-card")
        successor = successor.replace("supersedes: null", "supersedes: [a-card, b-card]")
        write_card(root, "merged-card", successor)
        index = root / "insights" / "index.md"
        index.write_text(index.read_text() + "\n- [merged-card](merged-card.md) — hero")
        # parent a: properly retired with forward link + state
        (root / "retired" / "insights").mkdir(parents=True)
        text_a = read_card(root, "a-card").replace("state: active", "state: superseded")
        text_a = text_a.replace("supersedes: null",
                                "supersedes: null\nsuperseded_by: merged-card")
        (root / "retired" / "insights" / "a-card.md").write_text(text_a)
        (root / "insights" / "a-card.md").unlink()
        # parent b: left active — a violation

    before, after = clone_bank(
        tmp_path, {"a-card": card_text("a-card"), "b-card": card_text("b-card")},
        merge,
    )
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("supersedes b-card but it is not in retired/" in f for f in findings)
    # founding evidence never referenced parent a's ledger
    assert any("merge evidence is by reference" in f for f in findings)


def test_generalize_born_candidate_with_probe(tmp_path):
    # Regression: a domain-broadening successor born active or probe-less
    # trips — upward moves are born as predictions with their test attached.
    def generalize(root):
        successor = card_text("general-card", scope="domain", state="active")
        successor = successor.replace(
            "probe: >-\n  Ablate the grouped-rank block on one forward fold; "
            "keep the clustered delta.",
            "probe: null",
        )
        successor = successor.replace("supersedes: null", "supersedes: [a-card]")
        write_card(root, "general-card", successor)
        (root / "retired" / "insights").mkdir(parents=True)
        text_a = read_card(root, "a-card").replace("state: active", "state: superseded")
        text_a = text_a.replace("scope: domain", "scope: [family:entity_binary_classification]")
        text_a = text_a.replace("supersedes: null",
                                "supersedes: null\nsuperseded_by: general-card")
        # reference the parent ledger so only the generalize findings remain
        (root / "retired" / "insights" / "a-card.md").write_text(text_a)
        (root / "insights" / "a-card.md").unlink()

    before, after = clone_bank(tmp_path, {"a-card": card_text("a-card")}, generalize)
    findings = BankTransactionValidator(before, after, body_floors={"rule": 35, "section": 25, "confidence": 8}).validate()
    assert any("must be born candidate" in f for f in findings)
    assert any("unseen-family probe" in f for f in findings)


def make_store_with_trajectory(tmp_path):
    store = TrajectoryStore(local=str(tmp_path / "store"))
    work_dir, log = build_work_dir(tmp_path)
    save_trajectory(store, TRAJECTORY_ID, work_dir=str(work_dir), campaign_log=str(log))
    return store


def entry(verdict="confirm", trajectory=TRAJECTORY_ID,
          ref="runs/run_0001/metrics.json",
          usage="Independent evidence — the campaign never saw the card.",
          effect="KEPT at +0.7136 ≈ 3.6 clustered SE on the validation split."):
    return {
        "source": {"learner_run": "lr_x", "trajectory": trajectory, "ref": ref,
                   "card_version": None},
        "verdict": verdict, "usage": usage, "effect": effect,
    }


def test_evidence_admission_source_resolution(tmp_path):
    # Regression (§5.2 check 1): unknown trajectory / unresolvable ref /
    # numbers that do not re-grep are all named findings.
    store = make_store_with_trajectory(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    card = Bank(str(bank_root)).cards["a-card"]
    good = entry()
    findings = evidence_admission_findings(card, [good], store, {})
    assert findings == []
    findings = evidence_admission_findings(
        card, [entry(trajectory="rel-x--y/20260101T000000_lane-z")], store, {}
    )
    assert any("is not in the store" in f for f in findings)
    findings = evidence_admission_findings(
        card, [entry(ref="runs/run_0009/ghost.json")], store, {}
    )
    assert any("resolves in neither" in f for f in findings)
    findings = evidence_admission_findings(
        card, [entry(effect="KEPT at +0.9944 ≈ 3 SE.")], store, {}
    )
    assert any("does not re-grep" in f for f in findings)


def test_evidence_admission_usage_vs_record(tmp_path):
    # Regression (§5.2 check 2): claimed serving without a record — and
    # claimed independence WITH one — both trip.
    store = make_store_with_trajectory(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    card = Bank(str(bank_root)).cards["a-card"]
    served_record = {TRAJECTORY_ID: {"served": [{"card": "a-card"}]}}
    findings = evidence_admission_findings(
        card, [entry(usage="The card was served and cited by the spec.")],
        store, {},
    )
    assert any("serving record does not carry" in f for f in findings)
    findings = evidence_admission_findings(
        card, [entry(usage="Independent rediscovery — never served.")],
        store, served_record,
    )
    assert any("claims independence but" in f for f in findings)


def test_evidence_admission_verdict_earnable(tmp_path):
    # Regression (§5.2 check 3): outcome verdicts need delta + significance;
    # sub-threshold effects carry exercise only.
    store = make_store_with_trajectory(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    card = Bank(str(bank_root)).cards["a-card"]
    findings = evidence_admission_findings(
        card, [entry(effect="It helped somewhat, direction positive.")],
        store, {},
    )
    assert any("needs a measured delta with a significance mark" in f
               for f in findings)
    findings = evidence_admission_findings(
        card, [entry(verdict="exercise",
                     effect="Directional only, under threshold — the metric "
                            "held at 0.7136.")],
        store, {},
    )
    assert findings == []


def test_retired_cards_are_frozen_history(tmp_path):
    # Regression: a card already in retired/ before the run must never change
    # again — merge founding references point into it.
    def setup(root):
        (root / "retired" / "insights").mkdir(parents=True)
        text = read_card(root, "a-card").replace("state: active", "state: retired")
        (root / "retired" / "insights" / "a-card.md").write_text(text)
        (root / "insights" / "a-card.md").unlink()

    before_root = build_bank(tmp_path / "before", {
        "a-card": card_text("a-card"), "b-card": card_text("b-card")})
    setup(before_root)
    after_root = tmp_path / "after" / "bank"
    shutil.copytree(before_root, after_root)
    retired = after_root / "retired" / "insights" / "a-card.md"
    retired.write_text(retired.read_text().replace("score: 0.7", "score: 0.1"))
    findings = BankTransactionValidator(
        Bank(str(before_root)), Bank(str(after_root)),
        body_floors={"rule": 35, "section": 25, "confidence": 8},
    ).validate()
    assert any("modified after retirement" in f for f in findings)


def test_negated_serving_is_independence_not_a_claim(tmp_path):
    # Regression (founding-bank self-review): "never served" / "served
    # nowhere" prose must read as independence, not as a serving claim — and
    # it trips when the record shows the card WAS served.
    store = make_store_with_trajectory(tmp_path)
    bank_root = build_bank(tmp_path, {"a-card": card_text("a-card")})
    card = Bank(str(bank_root)).cards["a-card"]
    findings = evidence_admission_findings(
        card, [entry(usage="Served nowhere (pre-bank campaign); the lane "
                           "applied the move independently.")],
        store, {},
    )
    assert findings == []
    served_record = {TRAJECTORY_ID: {"served": [{"card": "a-card"}]}}
    findings = evidence_admission_findings(
        card, [entry(usage="Never served in that campaign.")],
        store, served_record,
    )
    assert any("claims independence but" in f for f in findings)
