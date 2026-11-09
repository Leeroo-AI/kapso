#!/usr/bin/env python3
"""Fail when a public page breaks one of the two claim rules
(docs/plans/docs-and-examples-redesign.md, Checks).

1. A phrase of the words-out list (scripts/words_out.txt, from the
   redesign's rules for every public sentence) appears in prose — outside
   fenced code and inline code — as whole words, an `s` or `ly` ending
   allowed, across a line break. The list has two sections: phrases banned
   everywhere, and phrases banned on the front pages only (README.md,
   docs/index.mdx), where a word is also the name of something the inner
   pages must describe (the judge role, a host's trust line). Prose for
   this rule is every visible string: the paragraphs, the headings, a
   page's front matter (its title, sidebar title, description and
   keywords, folded or quoted over several lines or not), the title,
   description, caption and alt text of a card, frame or image tag, the
   strings of the plugin's manifests and the Context7 file, the package's
   description and the CLI's own help line.
2. A result — a number with a decimal part, a percentage, or an integer
   beside a rating word (Elo, rating, rank) — sits beside a metric word
   (AUROC, accuracy, F1, MAE, NMAE, MAP, log loss, score, rank, rating,
   ...) in a paragraph that carries no provenance. Provenance is a form,
   not a word: a verb of measurement (ran, measured, recorded, published,
   submitted, graded, scored by, the evaluator, official, ...) together
   with who or when — an actor (by us, by Leeroo, by the benchmark's
   grader, we ran, our campaign), a date (2026-08-26, July 2026, in 2026)
   or a link to the record. A table whose header names a provenance column
   carries provenance per row, in the rows whose cell is not empty.
   Headings are not paragraphs; a number marked beside itself as a target,
   expected or not measured is not a result, and the mark covers its own
   clause; a number beside a tolerance, a band, a floor, a budget or an
   axis origin is a rule, not a result.

Scans every .mdx under docs/, the repository README, the examples' README
and the examples' and the benchmarks' READMEs and the plugin's skills.
Code fences and inline code are not prose; a `text` or `md` fence is (a
prompt the user pastes is the most-copied sentence on a page), for the
words-out rule only. A hit prints the file, the line and the rule; the
exit status is the number of hits. Standard library only: the docs
workflow runs it on the runner's own Python.
"""

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Iterator, List, Tuple

REPO = Path(__file__).resolve().parent.parent
WORDS_OUT = REPO / "scripts" / "words_out.txt"
METRIC_WORDS = (
    "auroc", "auc", "accuracy", "f1", "mae", "rmse", "nmae", "map", "mrr", "log loss", "logloss", "score",
    "scores", "scored", "rank", "ranked", "rating", "elo", "sharpe", "percent", "medal", "leaderboard",
)
# A rating word beside which an integer is a result too (1909 Elo, rank 38, a rating of 1363).
RATING_WORDS = ("elo", "rating", "ratings", "rank", "ranked")
RATING_REACH = 14                                      # characters between an integer and its rating word
# A verb of measurement; provenance needs one of these with an actor, a date or a link.
PROVENANCE_WORDS = (
    "ran", "run by", "runs by", "measured", "recorded", "fetched", "published", "reported", "submitted",
    "graded", "scored by", "evaluator", "evaluated", "official", "read on",
)
_ACTOR = re.compile(
    r"\b(?:by\s+(?:us|the\s+\w+|[A-Z][\w'-]*)|we\s+(?:ran|measured|recorded|graded|submitted|published)"
    r"|our\s+(?:run|runs|campaign|campaigns|measurement|measurements))\b"
)
_DATE = re.compile(
    r"\b(?:20\d\d-\d\d(?:-\d\d)?|(?:in|on|of|since)\s+20\d\d|(?:january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\s+(?:\d{1,2},?\s+)?20\d\d)\b",
    re.IGNORECASE,
)
# A number is a target, not a result, when one of these sits beside it.
_EXPECTED = re.compile(r"\b(expected|not measured|not recorded|target|asks for|toward|reachable)\b", re.IGNORECASE)
# A number beside one of these is a rule (a tolerance, a band, a floor, a budget), not a result.
_RULE = re.compile(r"(±|\b(?:budget|floor|tolerance|band|threshold|within|seed|axis|origin)\b)", re.IGNORECASE)
_RESULT = re.compile(r"(?<![A-Za-z0-9_.-])(?:\d+\.\d+|\d+(?:\.\d+)?%)(?![A-Za-z0-9_])")
_INTEGER = re.compile(r"(?<![A-Za-z0-9_.-])\d{1,6}(?![A-Za-z0-9_.%])")
WINDOW = 60                                            # characters either side of a number that count as "beside"
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
FRONT_KEYS = ("title", "sidebarTitle", "description", "keywords")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LINK_URL = re.compile(r"\]\([^)]*\)")
# A card is its own paragraph: the tag lines of a card group bound it.
_CARD_TAG = re.compile(r"\s*</?Card(?:Group)?\b")
# The visible strings of a tag: a card's or frame's title, description or caption, an image's alt text.
_ATTRIBUTE = re.compile(r"\b(?:title|description|caption|alt)=\"([^\"]*)\"")
_PROSE_FENCES = ("text", "md", "markdown", "")
# A line that is only tags (one, or several with nothing but space between
# them, as a centred image: <p align="center"><img …></p>), a tag's
# attributes, a tag's close or a Markdown image: its visible text is read
# for the words-out rule by visible_strings; it is not a paragraph, since
# the paragraph with the provenance is the one the picture or frame
# illustrates.
_TAG_LINE = re.compile(r"^\s*(?:(?:<[^>]*>?\s*)+|[\w-]+=\"[^\"]*\"|/>|!\[[^\]]*\]\([^)]*\))\s*$")
# A clause: a mark (a target, a rule) covers the numbers in its own clause.
# A period between digits is a decimal point, not a boundary.
_CLAUSE_BOUNDARY = re.compile(r"(?<!\d)\.|\.(?!\d)|[;:,()\[\]]")
FRONT_PAGES = ("README.md", "docs/index.mdx")
# The manifests whose strings a store or an agent reads before any page.
MANIFESTS = (
    ".claude-plugin/marketplace.json", "plugin/.claude-plugin/plugin.json", "plugin/.codex-plugin/plugin.json",
    "plugin/.cursor-plugin/plugin.json", "context7.json",
)
PYPROJECT = "pyproject.toml"
CLI_MODULE = "src/kapso/cli/__init__.py"
_CLI_DESCRIPTION = re.compile(r"description=\"([^\"]*)\"")


def pages() -> List[Path]:
    return (
        [REPO / "README.md"] + sorted((REPO / "docs").rglob("*.mdx"))
        + [REPO / "examples" / "README.md"] + sorted((REPO / "examples").glob("*/README.md"))
        + [REPO / "examples" / "harnesses" / "README.md"]
        + sorted((REPO / "examples" / "harnesses").glob("*/README.md")) + sorted((REPO / "benchmarks").glob("*/README.md"))
        + sorted((REPO / "plugin" / "skills").glob("*/SKILL.md"))
    )


def _strings(node, key: str) -> Iterator[Tuple[str, str]]:
    """(key path, text) for every string leaf of a JSON document."""
    if isinstance(node, dict):
        for name, value in node.items():
            yield from _strings(value, f"{key}.{name}" if key else name)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings(value, f"{key}[{index}]")
    elif isinstance(node, str):
        yield key, node


def manifest_strings() -> List[Tuple[Path, str, str]]:
    """(file, key, text) for every string the manifests carry, at any depth
    (a plugin's description, the interface's short and long descriptions
    and its prompts, the marketplace's plugin descriptions, the Context7
    rules), the package's description and the CLI's own help line."""
    found = []
    for relative in MANIFESTS:
        path = REPO / relative
        found += [(path, key, text) for key, text in _strings(json.loads(path.read_text(encoding="utf-8")), "")]
    pyproject = REPO / PYPROJECT
    found.append((pyproject, "project.description", str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["description"])))
    cli = REPO / CLI_MODULE
    description = _CLI_DESCRIPTION.search(cli.read_text(encoding="utf-8"))
    if description is None:
        raise ValueError(f"{cli}: no parser description found")
    found.append((cli, "parser description", description.group(1)))
    return found


def front_matter_fields(text: str) -> Iterator[Tuple[int, str, str]]:
    """(line number, key, value) for the front matter's title, sidebar
    title, description and keywords; a value folded or quoted over several
    lines, or a list, is read whole (its continuation lines are indented)."""
    front = _FRONT_MATTER.match(text)
    if not front:
        return
    current = None
    for number, line in enumerate(front.group(0).split("\n"), start=1):
        if current is not None and line.startswith((" ", "\t")):
            current[2].append(line.strip())
            continue
        if current is not None:
            yield current[0], current[1], " ".join(part for part in current[2] if part)
            current = None
        key, separator, value = line.partition(":")
        if separator and key.strip() in FRONT_KEYS:
            value = value.strip()
            current = (number, key.strip(), [] if value in (">", "|", ">-", "|-", ">+", "|+") else [value])
    if current is not None:
        yield current[0], current[1], " ".join(part for part in current[2] if part)


def visible_strings(text: str) -> Iterator[Tuple[int, str]]:
    """(line number, text) for the visible strings that are not paragraphs:
    the headings, and the title, description, caption and alt text of a
    tag, outside code fences."""
    front = _FRONT_MATTER.match(text)
    offset = text[:front.end()].count("\n") if front else 0
    body = text[front.end():] if front else text
    in_fence = False
    for number, line in enumerate(body.split("\n"), start=offset + 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.lstrip().startswith("#"):
            yield number, line.lstrip("# ").strip()
        for value in _ATTRIBUTE.findall(line):
            yield number, value


def prose_paragraphs(text: str) -> Iterator[Tuple[int, Tuple[str, bool]]]:
    """(first line number, (paragraph, in a prose fence)) for every prose
    paragraph: code fences dropped whole, a text fence kept as prose,
    inline code and link targets blanked, front matter skipped. A
    paragraph that runs into a fence is flushed when the fence opens; a
    card's text is a paragraph of its own (its neighbour's provenance is
    not its own)."""
    front = _FRONT_MATTER.match(text)
    offset = text[:front.end()].count("\n") if front else 0
    body = text[front.end():] if front else text
    in_fence = False
    prose_fence = False
    paragraph: List[str] = []
    start = offset + 1
    for index, line in enumerate(body.split("\n"), start=offset + 1):
        if line.lstrip().startswith("```"):
            if paragraph:
                yield start, ("\n".join(paragraph), in_fence)
                paragraph = []
            if in_fence:
                in_fence = prose_fence = False
            else:
                in_fence = True
                prose_fence = line.strip()[3:].split(" ")[0].lower() in _PROSE_FENCES
            continue
        if in_fence and not prose_fence:
            continue
        if not line.strip() or _CARD_TAG.match(line):
            if paragraph:
                yield start, ("\n".join(paragraph), in_fence)
                paragraph = []
            continue
        if line.lstrip().startswith("#") or _TAG_LINE.match(line):
            continue                                   # a heading or a tag is not a paragraph (visible_strings reads it)
        if not paragraph:
            start = index
        paragraph.append(_LINK_URL.sub("](url)", _INLINE_CODE.sub("`code`", line)))
    if paragraph:
        yield start, ("\n".join(paragraph), in_fence)


def words_out() -> dict:
    """{"everywhere": [...], "front": [...]} from the list's two sections."""
    sections: dict = {"everywhere": [], "front": []}
    current = "everywhere"
    for line in WORDS_OUT.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("[") and entry.endswith("]"):
            current = entry[1:-1]
            if current not in sections:
                raise ValueError(f"{WORDS_OUT}: unknown section {current!r}")
            continue
        sections[current].append(entry)
    return sections


def phrase_pattern(phrase: str) -> re.Pattern:
    """The phrase as whole words, an `s` or `ly` ending allowed (agents,
    autonomously), its spaces matching a line break too."""
    words = re.escape(phrase).replace("\\ ", r"\s+")
    return re.compile(r"(?<![A-Za-z0-9-])" + words + r"(?:s|ly)?(?![A-Za-z0-9-])", re.IGNORECASE)


def provenance_column(header: str) -> int:
    """The index of the header cell that names a provenance column, or -1."""
    cells = [cell.strip().lower() for cell in header.strip().strip("|").split("|")]
    for index, cell in enumerate(cells):
        if any(word in cell for word in ("provenance", "source", "who ran", "run by", "fetched", "entry", "recorded", "best other")):
            return index
    return -1


def _has_word(text: str, word: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text) is not None


def has_provenance(paragraph: str) -> bool:
    """A verb of measurement with who or when: an actor, a date or a link
    to the record. "Measured." alone names nobody and no time."""
    lowered = paragraph.lower()
    if not any(_has_word(lowered, word) for word in PROVENANCE_WORDS):
        return False
    return bool(_ACTOR.search(paragraph) or _DATE.search(paragraph) or "](url)" in paragraph)


def _results(paragraph: str) -> List[Tuple[int, int, str]]:
    """(start, end, text) of every result in the paragraph: a decimal or a
    percentage, and an integer within reach of a rating word."""
    found = [(m.start(), m.end(), m.group(0)) for m in _RESULT.finditer(paragraph)]
    lowered = paragraph.lower()
    for match in _INTEGER.finditer(paragraph):
        reach = lowered[max(0, match.start() - RATING_REACH):match.end() + RATING_REACH]
        if any(_has_word(reach, word) for word in RATING_WORDS):
            found.append((match.start(), match.end(), match.group(0)))
    return sorted(set(found))


def _marked(paragraph: str, start: int, end: int) -> bool:
    """Whether a target or rule word covers the number: it sits in the
    number's own clause (between two punctuation marks), within the
    window. "An accuracy of 0.30, toward 0.50" marks 0.50 and not 0.30."""
    clause_start = max((m.end() for m in _CLAUSE_BOUNDARY.finditer(paragraph, 0, start)), default=0)
    after = _CLAUSE_BOUNDARY.search(paragraph, end)
    clause_end = after.start() if after else len(paragraph)
    clause = paragraph[max(clause_start, start - WINDOW):min(clause_end, end + WINDOW)]
    return bool(_EXPECTED.search(clause) or _RULE.search(clause))


def _claims_in(text: str) -> List[str]:
    hits = []
    for start, end, number in _results(text):
        if _marked(text, start, end):
            continue
        window = text[max(0, start - WINDOW):end + WINDOW].lower()
        metric = next((word for word in METRIC_WORDS if _has_word(window, word)), None)
        if metric:
            hits.append(f"{number} beside {metric!r}")
    return hits


def number_claims(paragraph: str) -> List[str]:
    """The metric numbers in a paragraph with no provenance, each read with
    the text beside it: a target or a rule is not a result. A table with a
    provenance column is read row by row: a row whose cell is empty has
    none."""
    lines = paragraph.split("\n")
    if lines[0].lstrip().startswith("|"):
        column = provenance_column(lines[0])
        if column >= 0:
            hits = []
            for row in lines[2:]:
                cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
                if column >= len(cells) or not cells[column]:
                    hits += _claims_in(row)
            return hits
    if has_provenance(paragraph):
        return []
    return _claims_in(paragraph)


def scan(path: Path, lists: dict) -> List[str]:
    findings = []
    relative = path.relative_to(REPO)
    phrases = list(lists["everywhere"])
    if str(relative) in FRONT_PAGES:
        phrases += lists["front"]
    patterns = [(phrase, phrase_pattern(phrase)) for phrase in phrases]
    text = path.read_text(encoding="utf-8")
    for line_number, key, value in front_matter_fields(text):
        for phrase, pattern in patterns:
            if pattern.search(value):
                findings.append(f"{relative}:{line_number}: words-out {phrase!r} in the front matter's {key}")
    for line_number, value in visible_strings(text):
        for phrase, pattern in patterns:
            if pattern.search(value):
                findings.append(f"{relative}:{line_number}: words-out {phrase!r} in a heading or a tag's text")
    for line_number, (paragraph, in_prose_fence) in prose_paragraphs(text):
        for phrase, pattern in patterns:
            if pattern.search(paragraph):
                findings.append(f"{relative}:{line_number}: words-out {phrase!r}")
        if in_prose_fence:
            continue                                   # a pasted prompt's numbers are its own
        for claim in number_claims(paragraph):
            findings.append(f"{relative}:{line_number}: a number with no provenance: {claim}")
    return findings


def scan_manifests(lists: dict) -> List[str]:
    patterns = [(phrase, phrase_pattern(phrase)) for phrase in lists["everywhere"]]
    findings = []
    for path, key, value in manifest_strings():
        for phrase, pattern in patterns:
            if pattern.search(value):
                findings.append(f"{path.relative_to(REPO)}: words-out {phrase!r} in {key}")
    return findings


def main() -> int:
    lists = words_out()
    findings: List[str] = scan_manifests(lists)
    for page in pages():
        findings += scan(page, lists)
    for finding in findings:
        print(finding)
    print(f"{len(findings)} finding(s) over {len(pages())} pages and {len(MANIFESTS)} manifests")
    return min(len(findings), 125)


if __name__ == "__main__":
    sys.exit(main())
