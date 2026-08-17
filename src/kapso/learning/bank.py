# Bank — the knowledge repo's card model and read layer.
#
# Design: learn-from-trajectories-design.md §3.1 (the OKF bundle), §3.2 (the
# card: two types, OKF reserved fields + kapso extensions, body as THE FACT).
# This module is the read side: parsing, scope eligibility, and conformance
# findings — what the retriever's push core and the graders consume. The write
# side (diff invariants, indexes, lr_ tags) lands with the update crew (P4).

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CARD_TYPES = ("insight", "procedure")
RELIABILITY_STATES = ("candidate", "active", "cold", "retired", "superseded")
EVIDENCE_VERDICTS = ("confirm", "weaken", "refine", "refute", "spawn", "exercise")
REPRESENTATIONS = ("text", "code")
SERVABLE_STATES = ("candidate", "active")  # cold/retired/superseded never serve

_SCOPE_COORD_PATTERN = re.compile(r"^(family|dataset):(.+)$")
DECOY_REGISTRY_NAME = ".decoys.yaml"
SIGHTINGS_NAME = "sightings.md"


class Card:
    """One parsed card: frontmatter fields + the body (THE FACT)."""

    def __init__(self, name: str, kind_dir: str, frontmatter: Dict[str, Any], body: str):
        self.name = name              # the slug — path is identity
        self.kind_dir = kind_dir      # "insights" | "procedures"
        self.frontmatter = frontmatter
        self.body = body.strip()

    @classmethod
    def parse(cls, path: Path, bank_root: Path) -> "Card":
        text = path.read_text()
        if not text.startswith("---\n") or "\n---" not in text[4:]:
            raise ValueError(f"card {path} has no frontmatter block")
        end = text.index("\n---", 4)
        frontmatter = yaml.safe_load(text[4:end])
        if not isinstance(frontmatter, dict):
            raise ValueError(f"card {path} frontmatter is not a mapping")
        rel = path.relative_to(bank_root)
        kind_dir = rel.parts[0]
        name = rel.parts[1] if kind_dir == "procedures" else rel.stem
        return cls(name, kind_dir, frontmatter, text[end + 4:])

    # ------------------------------------------------------------ accessors

    @property
    def type(self) -> Optional[str]:
        return self.frontmatter.get("type")

    @property
    def hero(self) -> str:
        return str(self.frontmatter.get("description", "")).strip()

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
            findings.append(f"{self.name}: missing hero `description`")
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

    def conformance_findings(self) -> List[str]:
        """Bank-level OKF conformance: per-card checks plus index coverage."""
        findings: List[str] = []
        for card in self.cards.values():
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
