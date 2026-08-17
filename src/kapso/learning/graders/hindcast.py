# Hindcast report — parsing, corridors, and admission checks.
#
# Design: learn-from-trajectories-grader-scoring.md §1 (the report: evidence
# layer + scoring layer) and §0 (the idiom: agent-written, frame-bounded,
# agent-read). The frame computes crude corridor centers from marker counts
# and rejects scores outside center ± band — the tether that keeps judgment
# from contradicting the evidence layer. Null is a verdict (empty evidence
# base), never a gap; fabricating a number over an empty base rejects the
# report, as does hiding an existing base behind a null.

import re
from typing import Any, Callable, Dict, List, Optional

import yaml

from kapso.learning.refs import extract_refs

EXTRACTION_MARKERS = ("HIT-SERVED", "HIT-UNSERVED", "MISS-UNCARDED", "MISS-NOVEL")
CLAIMS_MARKERS = ("AGREED", "CONTRADICTED", "OUT-OF-SCOPE", "THIN")
SERVING_MARKERS = ("SERVED-USED", "UPTAKE-FAIL", "SERVE-MISS", "SERVE-NOISE")

SECTION_MARKERS = {
    "extraction": EXTRACTION_MARKERS,
    "claims": CLAIMS_MARKERS,
    "serving": SERVING_MARKERS,
}
# Section headings as the report writes them (§1: three body sections).
SECTION_HEADINGS = {
    "extraction": "extraction",
    "claims": "claims settlement",
    "serving": "serving",
}
DIMENSIONS = ("foresight", "accuracy", "serving")

_ENTRY_PATTERN = re.compile(r"^- \*\*([A-Z-]+)\*\*(?:\s*(?:—|-)\s*)(.*)$", re.DOTALL)
_CARD_REF_PATTERN = re.compile(r"\[(insight|procedure):\s*([A-Za-z0-9._/-]+)\]")
# A measured delta (−0.004, +0.006, ±0.001): sign + decimal. Matched on
# ref-stripped text so path hyphens (flow-1.md) never count as deltas.
_SIGNED_NUMBER_PATTERN = re.compile(r"[+\-−±]\s?\d+\.\d+")
_BRACKET_STRIP_PATTERN = re.compile(r"\[[^\[\]]*\]")


class HindcastReport:
    """A parsed hindcast report: frontmatter scoring layer + marker entries."""

    def __init__(self, frontmatter: Dict[str, Any], sections: Dict[str, List[Dict[str, Any]]]):
        self.frontmatter = frontmatter
        self.sections = sections

    @classmethod
    def parse(cls, text: str) -> "HindcastReport":
        """Parse the report; a structurally broken document raises (there is
        nothing to admit) while entry-level defects become findings later."""
        if not text.startswith("---\n") or "\n---" not in text[4:]:
            raise ValueError("hindcast report has no frontmatter block")
        end = text.index("\n---", 4)
        frontmatter = yaml.safe_load(text[4:end])
        if not isinstance(frontmatter, dict):
            raise ValueError("hindcast report frontmatter is not a mapping")
        body = text[end + 4:]

        sections: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SECTION_MARKERS}
        current: Optional[str] = None
        entry_lines: List[str] = []

        def flush():
            if current and entry_lines:
                raw = "\n".join(entry_lines)
                match = _ENTRY_PATTERN.match(raw)
                sections[current].append({
                    "marker": match.group(1) if match else None,
                    "text": raw,
                    "links": extract_refs(raw),
                    "card_refs": _CARD_REF_PATTERN.findall(raw),
                })
            entry_lines.clear()

        for line in body.splitlines():
            if line.startswith("## "):
                flush()
                heading = line[3:].strip().lower()
                current = next(
                    (key for key, name in SECTION_HEADINGS.items()
                     if heading.startswith(name)),
                    None,
                )
            elif line.startswith("- ") and current:
                flush()
                entry_lines.append(line)
            elif entry_lines and current:
                entry_lines.append(line)
        flush()
        return cls(frontmatter, sections)

    def counts(self) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for section, entries in self.sections.items():
            for entry in entries:
                marker = entry["marker"]
                if marker in SECTION_MARKERS[section]:
                    tally[marker] = tally.get(marker, 0) + 1
        return tally


def corridor_centers(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    """The crude centers (§1.2–1.4); None where the evidence base is empty."""
    def get(*names: str) -> int:
        return sum(counts.get(n, 0) for n in names)

    centers: Dict[str, Optional[float]] = {}
    hits = get("HIT-SERVED", "HIT-UNSERVED")
    learnable = hits + get("MISS-UNCARDED")  # MISS-NOVEL excluded by design
    centers["foresight"] = hits / learnable if learnable else None

    settled = get("AGREED") + get("CONTRADICTED")
    centers["accuracy"] = get("AGREED") / settled if settled else None

    served_base = get("SERVED-USED", "UPTAKE-FAIL", "SERVE-MISS", "SERVE-NOISE")
    if served_base:
        delivery = get("SERVED-USED", "UPTAKE-FAIL", "SERVE-MISS")
        hit_rate = get("SERVED-USED") / delivery if delivery else 0.0
        noise_base = get("SERVED-USED", "UPTAKE-FAIL", "SERVE-NOISE")
        noise_share = get("SERVE-NOISE") / noise_base if noise_base else 0.0
        centers["serving"] = hit_rate * (1.0 - noise_share)
    else:
        centers["serving"] = None
    return centers


def _is_two_decimal(value: float) -> bool:
    return abs(value * 100 - round(value * 100)) < 1e-9


class HindcastValidator:
    """Admission checks (§1.6) over a parsed report.

    Context the frame supplies: `ref_exists(path) -> bool` resolves refs
    against the trajectory (mined view + raw bundle), and `known_cards` is the
    card-name set at the graded bank head. Every violation is a named finding;
    an admitted report has none.
    """

    def __init__(
        self,
        graders_config: Dict[str, Any],
        ref_exists: Callable[[str], bool],
        known_cards: set,
    ):
        self.band = graders_config["score_band"]
        self.min_settlements = graders_config["min_settlements"]
        self.ref_exists = ref_exists
        self.known_cards = known_cards

    def validate(self, report: HindcastReport) -> List[str]:
        findings: List[str] = []
        findings += self._check_frontmatter(report)
        findings += self._check_markers(report)
        findings += self._check_refs(report)
        findings += self._check_liftable(report)
        findings += self._check_scores(report)
        findings += self._check_rationale(report)
        return findings

    # ------------------------------------------------------------ structure

    def _check_frontmatter(self, report: HindcastReport) -> List[str]:
        findings = []
        for key in ("trajectory", "bank_head", "hindcast"):
            if key not in report.frontmatter:
                findings.append(f"frontmatter missing `{key}`")
        block = report.frontmatter.get("hindcast")
        if not isinstance(block, dict):
            findings.append("`hindcast` block is not a mapping")
        return findings

    def _check_markers(self, report: HindcastReport) -> List[str]:
        findings = []
        for section, entries in report.sections.items():
            for entry in entries:
                if entry["marker"] not in SECTION_MARKERS[section]:
                    findings.append(
                        f"{section}: entry marker {entry['marker']!r} is not in "
                        f"{SECTION_MARKERS[section]}"
                    )
        return findings

    # ----------------------------------------------------------------- refs

    def _check_refs(self, report: HindcastReport) -> List[str]:
        findings = []
        for section, entries in report.sections.items():
            for entry in entries:
                for link in entry["links"]:
                    path = link.partition("#")[0]
                    if path and not self.ref_exists(path):
                        findings.append(
                            f"{section}: ref {link!r} does not resolve in the "
                            f"trajectory"
                        )
                for kind, name in entry["card_refs"]:
                    if name not in self.known_cards:
                        findings.append(
                            f"{section}: [{kind}: {name}] is not a card at the "
                            f"graded bank head"
                        )
                if entry["marker"] == "MISS-UNCARDED" and not entry["links"]:
                    findings.append(
                        "extraction: a MISS-UNCARDED entry cites no learn-set "
                        "source ref"
                    )
        return findings

    # ------------------------------------------------------------- liftable

    def _check_liftable(self, report: HindcastReport) -> List[str]:
        """AGREED/CONTRADICTED entries must be liftable into evidence (§4):
        a measured delta plus at least one ref."""
        findings = []
        for entry in report.sections["claims"]:
            if entry["marker"] in ("AGREED", "CONTRADICTED"):
                prose = _BRACKET_STRIP_PATTERN.sub("", entry["text"])
                if not _SIGNED_NUMBER_PATTERN.search(prose):
                    findings.append(
                        f"claims: a {entry['marker']} entry carries no measured "
                        f"delta — not liftable"
                    )
                if not entry["links"]:
                    findings.append(
                        f"claims: a {entry['marker']} entry carries no ref — "
                        f"not liftable"
                    )
        return findings

    # --------------------------------------------------------------- scores

    def _check_scores(self, report: HindcastReport) -> List[str]:
        findings: List[str] = []
        block = report.frontmatter.get("hindcast")
        if not isinstance(block, dict):
            return findings  # already a structure finding
        centers = corridor_centers(report.counts())
        # accuracy nulls below the settlement floor (§1.3)
        settled = report.counts().get("AGREED", 0) + report.counts().get("CONTRADICTED", 0)
        if settled < self.min_settlements:
            centers["accuracy"] = None

        non_null: List[float] = []
        for dim in DIMENSIONS:
            if dim not in block:
                findings.append(f"hindcast block missing `{dim}`")
                continue
            value = block[dim]
            center = centers[dim]
            if value is None:
                if center is not None:
                    findings.append(
                        f"`{dim}` is null but its evidence base is non-empty "
                        f"(center {center:.2f})"
                    )
                continue
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                findings.append(f"`{dim}` must be a float in [0, 1] or null")
                continue
            if not _is_two_decimal(float(value)):
                findings.append(f"`{dim}` has more than two decimals — false precision")
            if center is None:
                findings.append(
                    f"`{dim}` carries a number but its evidence base is empty — "
                    f"null is the verdict"
                )
                continue
            if abs(value - center) > self.band + 1e-9:
                findings.append(
                    f"`{dim}` {value:.2f} is outside its corridor "
                    f"({center:.2f} ± {self.band:.2f})"
                )
            non_null.append(float(value))

        overall = block.get("score")
        if overall is None:
            if non_null:
                findings.append(
                    "`score` is null but at least one dimension is scored"
                )
        elif not isinstance(overall, (int, float)) or not 0 <= overall <= 1:
            findings.append("`score` must be a float in [0, 1] or null")
        elif non_null:
            low = max(0.0, min(non_null) - self.band)
            high = min(1.0, max(non_null) + self.band)
            if not low - 1e-9 <= overall <= high + 1e-9:
                findings.append(
                    f"`score` {overall:.2f} escapes the span of its own "
                    f"dimensions [{low:.2f}, {high:.2f}]"
                )
        return findings

    # ------------------------------------------------------------ rationale

    def _check_rationale(self, report: HindcastReport) -> List[str]:
        """Presence duties (§1.5); content depth is the verifier agent's job."""
        findings = []
        block = report.frontmatter.get("hindcast")
        rationale = block.get("rationale") if isinstance(block, dict) else None
        if not isinstance(rationale, str) or not rationale.strip():
            findings.append("hindcast rationale is missing — no naked scores")
            return findings
        if report.counts().get("MISS-NOVEL", 0) and "novel" not in rationale.lower():
            findings.append(
                "rationale does not state the novel share despite MISS-NOVEL "
                "entries"
            )
        return findings
