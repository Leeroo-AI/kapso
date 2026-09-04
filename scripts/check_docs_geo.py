#!/usr/bin/env python3
"""Fail the docs build on the mechanical GEO rules, and report the rest.

Mintlify's GEO guidance splits cleanly into rules a machine can judge and
rules only a writer can. This gates the first kind and prints the second.

Gated, because each is objectively true or false:
  * a page opens with prose, not a heading — an answer engine quoting the top
    of a page needs something to quote
  * every page carries an inline Markdown link to at least one other page —
    topic clusters that survive the Markdown rendition AI agents fetch
  * every page names the repository URL — the citation an answer engine needs
  * no banned marketing word, per Mintlify's own style guide
  * no banned terminology alias, so one concept keeps one name

Reported only:
  * the share of headings phrased as questions

That last one is deliberately not a gate. Gating a percentage produces
"How do I DeploymentFactory?" — it optimises the number at the cost of the
docs, which is the failure mode the guidance warns about by name.
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS_CONFIG = REPO / "docs.json"

REPO_URL = "https://github.com/Leeroo-AI/kapso"

BANNED_WORDS = ("powerful", "seamless", "robust", "cutting-edge", "blazingly")

# alias -> the name to use instead. Deliberately narrow: only aliases for
# concepts this project actually owns.
BANNED_ALIASES = {
    "knowledge system": "knowledge graph",
    "knowledge bank": "lesson bank",
    "developer agent": "coding agent",
}

QUESTION = re.compile(
    r"^(how|what|why|when|where|which|can|do|does|should|is|are)\b", re.I
)


def navigation_pages(config: dict) -> list[str]:
    pages: list[str] = []

    def walk(group: dict) -> None:
        for entry in group.get("pages", []):
            if isinstance(entry, str):
                pages.append(entry)
            else:
                walk(entry)

    for tab in config["navigation"]["tabs"]:
        for group in tab.get("groups", []):
            walk(group)
    return pages


def strip_code(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def main() -> int:
    config = json.loads(DOCS_CONFIG.read_text(encoding="utf-8"))
    failures: list[str] = []
    question_headings = 0
    total_headings = 0
    checked = 0

    for page in navigation_pages(config):
        path = REPO / f"{page}.mdx"
        if not path.is_file():
            # a missing page is mint validate's job, not this check's
            continue
        checked += 1
        raw = path.read_text(encoding="utf-8")
        front = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
        body = raw[front.end():] if front else raw
        prose = strip_code(body)

        if body.lstrip().startswith("#"):
            failures.append(f"{page}: opens with a heading; lead with a sentence")

        # Mintlify serves a Markdown rendition of each page to AI agents (and to
        # anyone sending Accept: text/markdown). <Card href> links do not survive
        # that rendition; inline Markdown links do. So the gate asks for inline.
        if "](/docs/" not in body:
            failures.append(
                f"{page}: no inline Markdown link to another page; "
                "add a 'Related pages:' line (Card links vanish for agent readers)"
            )

        if REPO_URL not in body:
            failures.append(f"{page}: does not name the repository; keep the project line at the end")

        lowered = prose.lower()
        for word in BANNED_WORDS:
            if re.search(rf"\b{word}\b", lowered):
                failures.append(f"{page}: uses marketing word {word!r}")
        for alias, prefer in BANNED_ALIASES.items():
            if alias in lowered:
                failures.append(f"{page}: says {alias!r}; use {prefer!r}")

        for heading in re.findall(r"^#{2,3}\s+(.+)$", prose, re.M):
            total_headings += 1
            if QUESTION.match(heading.strip()):
                question_headings += 1

    share = 100 * question_headings // max(total_headings, 1)
    print(f"Checked {checked} pages, {total_headings} headings.")
    print(f"Headings phrased as questions: {question_headings} ({share}%) — reported, not gated.")

    if failures:
        print(f"\n{len(failures)} GEO rule violations:\n")
        for line in failures:
            print(f"  {line}")
        return 1

    print("All gated GEO rules pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
