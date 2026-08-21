# Bank — the knowledge repo's card model and read layer.
#
# Design: learn-from-trajectories-design.md §3.1 (the OKF bundle), §3.2 (the
# card: two types, OKF reserved fields + kapso extensions, body as THE FACT).
# This module is the read side: parsing, scope eligibility, and conformance
# findings — what the retriever's push core and the graders consume. The write
# side (diff invariants, indexes, lr_ tags) lands with the update crew (P4).

import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CARD_TYPES = ("insight", "procedure")
RELIABILITY_STATES = ("candidate", "active", "cold", "retired", "superseded")
EVIDENCE_VERDICTS = ("confirm", "weaken", "refine", "refute", "spawn", "exercise")
REPRESENTATIONS = ("text", "code")
SERVABLE_STATES = ("candidate", "active")  # cold/retired/superseded never serve
# Outcome/exercise verdicts imply execution (CD§1): admission demands effect
# numbers that re-grep in a run where the method actually ran — a mere
# mention can never earn one.
EXECUTED_VERDICTS = ("confirm", "weaken", "refute", "refine", "exercise")
# A failed codify attempt is recorded as a log entry carrying this marker;
# the seeder skips the card until new executed evidence arrives (CD§1/§2).
CODIFY_FAILED_MARKER = "codify attempt failed"

_SCOPE_COORD_PATTERN = re.compile(r"^(family|dataset):(.+)$")
DECOY_REGISTRY_NAME = ".decoys.yaml"
SIGHTINGS_NAME = "sightings.md"
_RULE_PATTERN = re.compile(r"\*\*Rule:\*\*\s*(.+?)(?:\n\s*\n|\n#|\Z)", re.S)


def split_card_text(text: str) -> tuple:
    """Card file format v2 (user decision 2026-08-21): the engineer-facing
    body FIRST, then one line `---`, then the machine ledger as a fenced
    ```yaml block (so GitHub renders it as code, not markdown soup).
    Returns (body, ledger_yaml_text); raises on any other shape."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip() == "---":
            body = "\n".join(lines[:index]).strip()
            tail = [l for l in lines[index + 1:]]
            while tail and not tail[0].strip():
                tail.pop(0)
            while tail and not tail[-1].strip():
                tail.pop()
            if not tail or tail[0].strip() != "```yaml" \
                    or tail[-1].strip() != "```":
                raise ValueError(
                    "card ledger must be a fenced ```yaml block below the "
                    "`---` delimiter (format v2)"
                )
            return body, "\n".join(tail[1:-1])
    raise ValueError("card has no `---` ledger delimiter (format v2: body "
                     "first, machine ledger below)")


class _LedgerDumper(yaml.SafeDumper):
    """House style for ledger YAML: folded blocks for long prose, flow
    style for leaf collections, ISO-8601 Z timestamps."""


def _represent_str(dumper, value):
    if "\n" in value or len(value) > 76:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


def _represent_datetime(dumper, value):
    text = value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", text)


_LedgerDumper.add_representer(str, _represent_str)
_LedgerDumper.add_representer(datetime.datetime, _represent_datetime)


def dump_ledger(ledger: Dict[str, Any]) -> str:
    """Serialize a card's machine ledger in house style."""
    return yaml.dump(ledger, Dumper=_LedgerDumper, sort_keys=False,
                     allow_unicode=True, width=78, default_flow_style=None)


class Card:
    """One parsed card: the engineer-facing body (THE CARD) + the machine
    ledger. `frontmatter` keeps its historical name but lives BELOW the
    body on disk since format v2."""

    def __init__(self, name: str, kind_dir: str, frontmatter: Dict[str, Any], body: str):
        self.name = name              # the slug — path is identity
        self.kind_dir = kind_dir      # "insights" | "procedures"
        self.frontmatter = frontmatter
        self.body = body.strip()

    @classmethod
    def parse(cls, path: Path, bank_root: Path) -> "Card":
        body, ledger_text = split_card_text(path.read_text())
        frontmatter = yaml.safe_load(ledger_text)
        if not isinstance(frontmatter, dict):
            raise ValueError(f"card {path} ledger is not a mapping")
        rel = path.relative_to(bank_root)
        kind_dir = rel.parts[0]
        name = rel.parts[1] if kind_dir == "procedures" else rel.stem
        return cls(name, kind_dir, frontmatter, body)

    # ------------------------------------------------------------ accessors

    @property
    def type(self) -> Optional[str]:
        return self.frontmatter.get("type")

    @property
    def title(self) -> str:
        """The H1 heading — the rule as a plain sentence. Cards still
        awaiting the template rewrite fall back to the humanized slug."""
        first = self.body.split("\n", 1)[0]
        if first.startswith("# "):
            return first[2:].strip()
        return self.name.replace("-", " ").capitalize()

    @property
    def hero(self) -> str:
        """One-line summary for indexes and shortlists: the **Rule:**
        paragraph, collapsed. Pre-template bodies fall back to their first
        non-heading line."""
        match = _RULE_PATTERN.search(self.body)
        if match:
            return " ".join(match.group(1).split())
        for line in self.body.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
        return ""

    @property
    def scope(self) -> Any:
        return self.frontmatter.get("scope")

    @property
    def reliability(self) -> Dict[str, Any]:
        block = self.frontmatter.get("reliability")
        return block if isinstance(block, dict) else {}

    @property
    def state(self) -> Optional[str]:
        return self.reliability.get("state")

    @property
    def score(self) -> Optional[float]:
        return self.reliability.get("score")

    @property
    def evidence(self) -> List[Dict[str, Any]]:
        entries = self.frontmatter.get("evidence")
        return entries if isinstance(entries, list) else []

    @property
    def representation(self) -> str:
        return self.frontmatter.get("representation", "text")

    def eligible_for(self, task_coords: Dict[str, str]) -> bool:
        """Serving eligibility (§5.1): task ∈ scope. `domain` covers every
        task; a coordinate list covers tasks matching any family:/dataset:
        coordinate. Tags never participate."""
        scope = self.scope
        if scope == "domain":
            return True
        if not isinstance(scope, list):
            return False
        for coordinate in scope:
            match = _SCOPE_COORD_PATTERN.match(str(coordinate))
            if match and task_coords.get(match.group(1)) == match.group(2):
                return True
        return False

    def conformance_findings(self) -> List[str]:
        """Card-level schema checks (admission raises live in the write side;
        these are the read-side findings the OKF conformance gate reports)."""
        findings = []
        front = self.frontmatter
        if front.get("type") not in CARD_TYPES:
            findings.append(f"{self.name}: type {front.get('type')!r} invalid")
        if not self.hero:
            findings.append(f"{self.name}: no hero derivable — body is empty "
                            f"of prose")
        scope = front.get("scope")
        if scope != "domain" and not (
            isinstance(scope, list) and scope
            and all(_SCOPE_COORD_PATTERN.match(str(c)) for c in scope)
        ):
            findings.append(f"{self.name}: scope {scope!r} is neither `domain` "
                            f"nor a family:/dataset: coordinate list")
        if not self.evidence:
            findings.append(f"{self.name}: evidence ledger is empty")
        for entry in self.evidence:
            if entry.get("verdict") not in EVIDENCE_VERDICTS:
                findings.append(
                    f"{self.name}: evidence verdict {entry.get('verdict')!r} invalid"
                )
        if self.state not in RELIABILITY_STATES:
            findings.append(f"{self.name}: reliability state {self.state!r} invalid")
        if not str(self.reliability.get("rationale", "")).strip():
            findings.append(f"{self.name}: reliability rationale missing — no naked scores")
        log = front.get("log")
        version = (front.get("provenance") or {}).get("version")
        if not isinstance(log, list) or not log:
            findings.append(f"{self.name}: log is missing")
        elif version is not None and len(log) != version:
            findings.append(
                f"{self.name}: provenance version {version} has {len(log)} log "
                f"entries — version ⇔ log-entry must be one-to-one"
            )
        if not self.body:
            findings.append(f"{self.name}: body (THE FACT) is empty")
        if self.type == "procedure" and self.representation not in REPRESENTATIONS:
            findings.append(
                f"{self.name}: representation {self.representation!r} invalid"
            )
        return findings


class Bank:
    """A read-only view over a bank checkout: cards, sightings, decoys."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"bank checkout {root} does not exist")
        self.cards: Dict[str, Card] = {}
        for path in sorted(self.root.glob("insights/*.md")):
            if path.name != "index.md":
                card = Card.parse(path, self.root)
                self.cards[card.name] = card
        for path in sorted(self.root.glob("procedures/*/card.md")):
            card = Card.parse(path, self.root)
            self.cards[card.name] = card
        # retired/ mirrors the active layout; moved, never deleted (§3.1).
        self.retired_cards: Dict[str, Card] = {}
        for path in sorted(self.root.glob("retired/insights/*.md")):
            if path.name != "index.md":
                card = Card.parse(path, self.root / "retired")
                self.retired_cards[card.name] = card
        for path in sorted(self.root.glob("retired/procedures/*/card.md")):
            card = Card.parse(path, self.root / "retired")
            self.retired_cards[card.name] = card

    @property
    def decoy_names(self) -> set:
        """The quarantine registry (frame-side knowledge; gauntlet + retriever
        both exclude these). A missing registry means no decoys — documented
        default, not an error."""
        registry = self.root / DECOY_REGISTRY_NAME
        if not registry.is_file():
            return set()
        names = yaml.safe_load(registry.read_text())
        if not isinstance(names, list):
            raise ValueError(f"{DECOY_REGISTRY_NAME} must be a list of card names")
        return set(names)

    def servable(self) -> List[Card]:
        """Cards the retriever may serve: servable state, never decoys
        (quarantine is law — §5.1)."""
        decoys = self.decoy_names
        return [
            card for card in self.cards.values()
            if card.state in SERVABLE_STATES and card.name not in decoys
        ]

    def sightings_text(self) -> str:
        path = self.root / SIGHTINGS_NAME
        return path.read_text() if path.is_file() else ""

    def log_text(self) -> str:
        path = self.root / "log.md"
        return path.read_text() if path.is_file() else ""

    def retired_by_founding_ref(self, ref: str) -> Card:
        """Resolve a merge/generalize founding reference (`retired/...`) to
        the retired parent card. A dangling founding reference is a corrupt
        bank — raise, never skip."""
        parts = Path(ref.partition("#")[0]).parts
        if len(parts) < 2 or parts[0] != "retired":
            raise ValueError(f"{ref!r} is not a retired/ founding reference")
        name = parts[2] if parts[1] == "procedures" else Path(parts[-1]).stem
        card = self.retired_cards.get(name)
        if card is None:
            raise ValueError(
                f"founding reference {ref!r} points at no retired card"
            )
        return card

    def executed_closure(self, card: Card) -> List[Dict[str, Any]]:
        """CD§1: the card's executed evidence entries, with merge/generalize
        founding references expanded into the retired parents' ledgers,
        recursively (supersedes only points backward — terminates)."""
        entries: List[Dict[str, Any]] = []
        seen: set = set()

        def walk(current: Card) -> None:
            if current.name in seen:
                return
            seen.add(current.name)
            for entry in current.evidence:
                ref = str((entry.get("source") or {}).get("ref") or "")
                if ref.startswith("retired/"):
                    walk(self.retired_by_founding_ref(ref))
                    continue
                if entry.get("verdict") in EXECUTED_VERDICTS:
                    entries.append(entry)

        walk(card)
        return entries

    def codify_recurrence(self, card: Card) -> int:
        """Distinct source campaigns across the executed closure — set
        semantics: same-campaign entries collapse, and the union across
        merged parents dedups a shared source campaign."""
        return len({
            str((entry.get("source") or {}).get("trajectory"))
            for entry in self.executed_closure(card)
            if (entry.get("source") or {}).get("trajectory")
        })

    def codify_blocked_by_failed_attempt(self, card: Card) -> bool:
        """A hopeless card is not retried until new evidence arrives: blocked
        iff a codify-failure log entry is at least as recent (by lr_ id) as
        the card's own latest executed evidence."""
        failures = [
            str(row.get("commit") or "")
            for row in (card.frontmatter.get("log") or [])
            if CODIFY_FAILED_MARKER in str(row.get("change") or "")
        ]
        if not failures:
            return False
        executed_runs = [
            str((entry.get("source") or {}).get("learner_run") or "")
            for entry in card.evidence
            if entry.get("verdict") in EXECUTED_VERDICTS
        ]
        return max(failures) >= max(executed_runs or [""])

    def conformance_findings(self) -> List[str]:
        """Bank-level OKF conformance: per-card checks plus index coverage.

        Decoy-registry cards are skipped: they are frame-owned instruments,
        evidence-free by construction (the gauntlet's standing bait), and the
        registry — not the schema — is their contract.
        """
        findings: List[str] = []
        decoys = self.decoy_names
        for card in self.cards.values():
            if card.name in decoys:
                continue
            findings += card.conformance_findings()
        for section in ("insights", "procedures"):
            section_dir = self.root / section
            if not section_dir.is_dir():
                continue
            index_path = section_dir / "index.md"
            if not index_path.is_file():
                findings.append(f"{section}/index.md is missing")
                continue
            index_text = index_path.read_text()
            for card in self.cards.values():
                if card.kind_dir == section and card.name not in index_text:
                    findings.append(f"{section}/index.md does not list {card.name}")
        return findings
