# Bank transaction validation — diff invariants + evidence admission.
#
# Design: update-crew doc §7 (the six-step validation's mechanical heart) +
# main doc §4.3/§5.2. The proposal medium is the working tree: the crew edits
# card files directly, and trust lives HERE — the frame validates the final
# diff against these invariants and rejects the whole transaction on any
# violation. Every check is a named finding.
#
# Scope notes, stated honestly: sightings changes are validated as
# set-additions/removals (an in-place edit is indistinguishable from
# remove+add at this layer; the critic owns entry-rewrite honesty). The
# verdict-earnability checks are v1 mechanical proxies (delta + significance
# marks present); semantic honesty is the critic's and assessor's job.

import re
from typing import Any, Dict, List, Optional

from kapso.learning.bank import (
    EVIDENCE_VERDICTS,
    Bank,
    Card,
    RELIABILITY_STATES,
)
from kapso.learning.trajectory_store import TrajectoryStore

# The claim layer: changes here are versioned (one bump + one log entry per
# transaction). Evidence appends and reliability re-scores do not bump — the
# version identifies the claim text, and evidence entries pin card_version.
CLAIM_LAYER_FIELDS = (
    "type", "title", "description", "tags", "scope", "scope_conditions",
    "probe", "supersedes", "contradicts", "representation", "entrypoint",
    "preconditions", "replay",
    # The reliability block is served with the card — a reassessment (score,
    # state, or rationale movement) is a claim-layer event (B6 live finding:
    # settlements must move or re-affirm the scores, and the re-affirmation
    # itself versions).
    "reliability",
)

_DELTA_PATTERN = re.compile(r"[+\-−±]\s?\d+\.\d+")
_NUMBER_PATTERN = re.compile(r"\d+\.\d{2,}")

# Negated serving ("never served", "served nowhere", "not in any brief") is
# an independence statement, not a participation claim (founding-bank
# self-review finding). Shared by evidence admission and the probe queue's
# exposure count.
NEGATED_SERVING_PHRASES = ("never served", "served nowhere", "not served",
                           "no brief", "not in any brief", "pre-bank")


def usage_negates_serving(usage: str) -> bool:
    lower = usage.lower()
    return any(phrase in lower for phrase in NEGATED_SERVING_PHRASES)


def usage_claims_serving(usage: str) -> bool:
    """Does this usage prose claim the card participated (served, cited, or
    probe-tested)? Negation-aware."""
    if usage_negates_serving(usage):
        return False
    lower = usage.lower()
    return any(word in lower for word in ("served", "cited", "probe"))


# The reader contract (user decisions 2026-08-21): three engineer-facing
# intents per body, each with enough substance to act on — a missing
# section is an unwritten card, a slogan-thin one is an unfinished card.
BODY_SECTIONS = ("## When you're here", "## Do this", "## What you gain")


def body_contract_gaps(body: str, min_words: int) -> List[str]:
    """Missing or slogan-thin reader sections, named. Shared by the
    transaction validator (touched cards must conform) and the seeder
    (non-conforming cards queue for rewrite)."""
    gaps: List[str] = []
    for index, section in enumerate(BODY_SECTIONS):
        if section not in body:
            gaps.append(f"missing {section!r}")
            continue
        tail = body.split(section, 1)[1]
        for later in BODY_SECTIONS[index + 1:]:
            tail = tail.split(later, 1)[0]
        words = len(tail.split())
        if words < min_words:
            gaps.append(
                f"{section!r} is slogan-thin ({words} words < {min_words})"
            )
    return gaps


def _claim_signature(card: Card) -> Dict[str, Any]:
    signature = {field: card.frontmatter.get(field) for field in CLAIM_LAYER_FIELDS}
    signature["body"] = card.body
    return signature


def _version(card: Card) -> Optional[int]:
    return (card.frontmatter.get("provenance") or {}).get("version")


class BankTransactionValidator:
    """Validate `after` (the working tree) against `before` (the committed
    state). Pure function of two Bank views + context; findings, never
    repairs."""

    def __init__(self, before: Bank, after: Bank, body_section_min_words: int):
        self.before = before
        self.after = after
        self.body_section_min_words = body_section_min_words

    def validate(self) -> List[str]:
        findings: List[str] = []
        findings += self._check_removals_are_moves()
        findings += self._check_decoys_untouched()
        for name in sorted(set(self.before.cards) & set(self.after.cards)):
            findings += self._check_card_evolution(
                self.before.cards[name], self.after.cards[name]
            )
        findings += self._check_contradicts_symmetry()
        findings += self._check_supersede_shapes()
        findings += self._check_sightings()
        findings += self._check_body_contract()
        return findings

    # ------------------------------------------------------- body contract

    def _check_body_contract(self) -> List[str]:
        """Cards written or rewritten in THIS transaction carry the three
        reader-intent sections with real substance — a card that cannot
        tell an engineer when it applies, what to do, and what it buys
        them (and why) has not finished being written. Untouched legacy
        cards pass: conformance rides each card's next edit (plus the
        seeder's capped rewrite rows), so the bank migrates incrementally
        instead of blocking every transaction on a full rewrite."""
        findings: List[str] = []
        decoys = self.after.decoy_names
        for name, card in sorted(self.after.cards.items()):
            if name in decoys:
                continue  # decoys are frozen bait — crews may not touch them
            previous = self.before.cards.get(name)
            if previous is not None and previous.body == card.body:
                continue  # untouched this transaction — legacy passes
            gaps = body_contract_gaps(card.body, self.body_section_min_words)
            if gaps:
                findings.append(
                    f"{name}: reader-contract body gaps — "
                    f"{'; '.join(gaps)} — write each section for an ML "
                    "engineer mid-task with enough substance to act on "
                    "(no length cap; abstract gains with their why; "
                    "numbers stay in evidence)"
                )
        return findings

    # ------------------------------------------------------ moves & decoys

    def _check_removals_are_moves(self) -> List[str]:
        findings = []
        retired_names = set(self.after.retired_cards)
        for name in sorted(set(self.before.cards) - set(self.after.cards)):
            if name not in retired_names:
                findings.append(
                    f"{name}: removed from the bank without a move to retired/ "
                    f"— retirement is a move, never a delete"
                )
        for name in sorted(set(self.before.retired_cards) - retired_names):
            findings.append(f"retired/{name}: a retired card was deleted")
        # Cards already retired before this run are frozen history: their
        # files never change again (merge founding references point INTO
        # them, so their stability is load-bearing).
        for name in sorted(set(self.before.retired_cards) & retired_names):
            before_card = self.before.retired_cards[name]
            after_card = self.after.retired_cards[name]
            if (
                before_card.frontmatter != after_card.frontmatter
                or before_card.body != after_card.body
            ):
                findings.append(
                    f"retired/{name}: modified after retirement — retired "
                    f"cards are frozen history"
                )
        return findings

    def _check_decoys_untouched(self) -> List[str]:
        findings = []
        decoys = self.before.decoy_names
        if decoys != self.after.decoy_names:
            findings.append("the decoy registry itself was modified")
        for name in sorted(decoys):
            before_card = self.before.cards.get(name)
            after_card = self.after.cards.get(name)
            if before_card is None or after_card is None:
                findings.append(f"decoy {name}: moved or removed")
            elif (
                before_card.frontmatter != after_card.frontmatter
                or before_card.body != after_card.body
            ):
                findings.append(f"decoy {name}: modified — decoys are untouchable")
        return findings

    # ------------------------------------------------------- per-card rules

    def _check_card_evolution(self, before: Card, after: Card) -> List[str]:
        findings = []
        name = after.name

        before_evidence = before.evidence
        after_evidence = after.evidence
        if after_evidence[: len(before_evidence)] != before_evidence:
            findings.append(f"{name}: evidence is append-only — existing entries changed")
        before_log = before.frontmatter.get("log") or []
        after_log = after.frontmatter.get("log") or []
        if after_log[: len(before_log)] != before_log:
            findings.append(f"{name}: log is append-only — existing entries changed")

        version_before, version_after = _version(before), _version(after)
        claim_changed = _claim_signature(before) != _claim_signature(after)
        new_log_entries = len(after_log) - len(before_log)
        if claim_changed:
            if version_after != (version_before or 0) + 1:
                findings.append(
                    f"{name}: claim-layer change without exactly one version "
                    f"bump ({version_before} -> {version_after})"
                )
            if new_log_entries != 1:
                findings.append(
                    f"{name}: claim-layer change must write exactly one log "
                    f"entry (got {new_log_entries})"
                )
        else:
            if version_after != version_before:
                findings.append(
                    f"{name}: version bumped without a claim-layer change"
                )
            if new_log_entries:
                findings.append(
                    f"{name}: log entries appended without a claim-layer change"
                )
        if version_after is not None and len(after_log) != version_after:
            findings.append(
                f"{name}: provenance version {version_after} has "
                f"{len(after_log)} log entries — one-to-one violated"
            )
        if after.state not in RELIABILITY_STATES:
            findings.append(f"{name}: reliability state {after.state!r} invalid")
        return findings

    # ---------------------------------------------------------- structural

    def _check_contradicts_symmetry(self) -> List[str]:
        findings = []
        for name, card in sorted(self.after.cards.items()):
            for other in card.frontmatter.get("contradicts") or []:
                other_card = self.after.cards.get(str(other))
                if other_card is None:
                    findings.append(
                        f"{name}: contradicts {other!r} which is not an active card"
                    )
                elif name not in [str(x) for x in
                                  (other_card.frontmatter.get("contradicts") or [])]:
                    findings.append(
                        f"{name} contradicts {other} but not vice versa — the "
                        f"edge lands on both cards"
                    )
        return findings

    def _check_supersede_shapes(self) -> List[str]:
        """MERGE/GENERALIZE successors (§3.3): parents retired + linked both
        ways; founding evidence BY REFERENCE into the parents' ledgers, never
        copied; a domain-broadening successor is born candidate with its
        probe present."""
        findings = []
        for name, card in sorted(self.after.cards.items()):
            parents = card.frontmatter.get("supersedes")
            if not parents or not isinstance(parents, list):
                continue
            parent_names = [str(p) for p in parents]
            parent_scopes = []
            for parent in parent_names:
                retired = self.after.retired_cards.get(parent)
                if retired is None:
                    findings.append(
                        f"{name}: supersedes {parent} but it is not in retired/"
                    )
                    continue
                parent_scopes.append(retired.scope)
                if retired.state not in ("superseded", "retired"):
                    findings.append(
                        f"{name}: superseded parent {parent} has state "
                        f"{retired.state!r}"
                    )
                if str(retired.frontmatter.get("superseded_by")) != name:
                    findings.append(
                        f"{name}: parent {parent} lacks the forward link "
                        f"(superseded_by) — superseded links both ways"
                    )
                referenced = any(
                    str((entry.get("source") or {}).get("ref", "")).startswith(
                        f"retired/"
                    ) and parent in str((entry.get("source") or {}).get("ref", ""))
                    for entry in card.evidence
                )
                if len(parent_names) >= 2 and not referenced:
                    findings.append(
                        f"{name}: no founding evidence references parent "
                        f"{parent}'s ledger — merge evidence is by reference, "
                        f"never copied"
                    )
            # The generalize shape: a successor claiming MORE than its parents
            # is a prediction — born candidate with its test attached.
            if card.scope == "domain" and parent_scopes and all(
                scope != "domain" for scope in parent_scopes
            ):
                if card.state != "candidate":
                    findings.append(
                        f"{name}: a generalizing successor must be born "
                        f"candidate (state is {card.state!r})"
                    )
                if not str(card.frontmatter.get("probe") or "").strip():
                    findings.append(
                        f"{name}: a generalizing successor must carry its "
                        f"unseen-family probe"
                    )
        return findings

    def _check_sightings(self) -> List[str]:
        """Set semantics: additions and removals are both legal (append /
        expiry); nothing else is checkable at this layer."""
        return []


def evidence_admission_findings(
    card: Card,
    new_entries: List[Dict[str, Any]],
    store: TrajectoryStore,
    serving_records: Dict[str, Dict[str, Any]],
) -> List[str]:
    """§5.2's three checks for each new evidence entry, mechanical layer:

    1. Source resolves — the trajectory exists in the store and the ref is in
       its inventory (or is a merge founding reference into retired/); the
       effect's quoted decimal numbers re-grep in the referenced artifact.
    2. Usage vs the record — a claimed serving/citation requires the card in
       the campaign's serving record; claimed independence requires absence.
    3. Verdict earnable — outcome verdicts carry a measured delta with a
       significance mark; `exercise` is the only sub-threshold verdict
       (v1 proxies; semantic honesty is the critic's job).
    """
    findings: List[str] = []
    for index, entry in enumerate(new_entries):
        label = f"{card.name} evidence[{index}]"
        source = entry.get("source") or {}
        trajectory = source.get("trajectory")
        ref = str(source.get("ref") or "")
        verdict = entry.get("verdict")
        usage = str(entry.get("usage") or "")
        effect = str(entry.get("effect") or "")

        if verdict not in EVIDENCE_VERDICTS:
            findings.append(f"{label}: verdict {verdict!r} invalid")
        if not usage.strip() or not effect.strip():
            findings.append(f"{label}: usage and effect prose are required")

        if ref.startswith("retired/"):
            pass  # merge founding reference — shape-checked by the validator
        elif not trajectory:
            findings.append(f"{label}: source.trajectory missing")
        else:
            manifest_path = store.local / str(trajectory) / "trajectory.yaml"
            if not manifest_path.is_file():
                findings.append(
                    f"{label}: trajectory {trajectory} is not in the store"
                )
            else:
                manifest = store.manifest(str(trajectory))
                inventory = manifest["inventory"]["sha256"]
                ref_path = ref.partition("#")[0]
                in_derived = (store.local / str(trajectory) / ref_path).is_file()
                if ref_path and ref_path not in inventory and not in_derived:
                    findings.append(
                        f"{label}: ref {ref_path!r} resolves in neither the "
                        f"raw inventory nor the derived view of {trajectory}"
                    )
                elif ref_path:
                    numbers = _NUMBER_PATTERN.findall(effect)
                    target = store.local / str(trajectory) / ref_path
                    if numbers and target.is_file():
                        content = target.read_text(errors="replace")
                        for number in numbers[:3]:
                            if number not in content:
                                findings.append(
                                    f"{label}: effect quotes {number} which "
                                    f"does not re-grep in {ref_path}"
                                )

            record = serving_records.get(str(trajectory)) or {}
            served_names = {row["card"] for row in record.get("served", [])}
            claims_use = usage_claims_serving(usage)
            claims_independence = (
                usage_negates_serving(usage) or "independen" in usage.lower()
            )
            if claims_use and card.name not in served_names:
                findings.append(
                    f"{label}: usage claims serving/citation but the serving "
                    f"record does not carry {card.name}"
                )
            if claims_independence and card.name in served_names:
                findings.append(
                    f"{label}: usage claims independence but {card.name} was "
                    f"served in that campaign"
                )

        if verdict in ("confirm", "weaken", "refute"):
            if not _DELTA_PATTERN.search(effect) or not (
                "SE" in effect or "±" in effect
            ):
                findings.append(
                    f"{label}: a {verdict} verdict needs a measured delta with "
                    f"a significance mark — sub-threshold effects carry "
                    f"`exercise` only"
                )
        if verdict == "spawn" and "[" not in effect + usage:
            findings.append(
                f"{label}: a spawn verdict must name the sibling card it founds"
            )
    return findings
