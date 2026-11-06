#!/usr/bin/env python3
"""Keep the translated READMEs in step with README.md.

A translation drifts silently: a score changes in README.md, a link is added,
a command is edited, and the translation keeps saying the old thing. Prose
cannot be checked mechanically, but everything a reader acts on can. This
check fails when a docs/translations/README.<lang>.md disagrees with
README.md on any of:

  - links: the same set of URLs and relative targets (a translation's
    relative targets carry a ../../ prefix, which is normalised away; the
    language switcher's own links are excepted);
  - numbers: the same numeric tokens, the same number of times — scores,
    versions, counts — so a translation can never cite a stale figure;
  - code: fenced blocks identical line for line once comments are stripped,
    since only the comments are translated;
  - the switcher: README.md's language switcher links every translation
    that exists, and every translation links README.md and its siblings.

The sync rule it enforces: README.md and its translations change in the same
commit. Stdlib only, like the other checks under scripts/.

  python3 scripts/check_readme_translations.py    # CI: exit 1 on drift
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "README.md"
TRANSLATIONS_DIR = ROOT / "docs" / "translations"
PREFIX = "../../"  # where a translation's relative links point back to

LINK = re.compile(r'\]\(([^)\s]+)\)|(?:href|src)="([^"]+)"')
SWITCHER_LINK = re.compile(r"(?:\.\./\.\./)?(?:docs/translations/)?README(?:\.[A-Za-z-]+)?\.md")
SWITCHER_BLOCK = re.compile(r"<!-- translations:start -->(.*?)<!-- translations:end -->", re.S)
TRANSLATION_NAME = re.compile(r"README\.([A-Za-z]{2,3}-[A-Za-z]{2,4})\.md")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M)
COMMENT = re.compile(r"(^|\s)#.*$")


def translations() -> list[pathlib.Path]:
    if not TRANSLATIONS_DIR.is_dir():
        return []
    return sorted(p for p in TRANSLATIONS_DIR.glob("README.*.md") if TRANSLATION_NAME.fullmatch(p.name))


def links(text: str, strip_prefix: bool) -> set[str]:
    targets = {a or b for a, b in LINK.findall(text)}
    targets = {t for t in targets if not SWITCHER_LINK.fullmatch(t)}
    if strip_prefix:
        targets = {t[len(PREFIX):] if t.startswith(PREFIX) else t for t in targets}
    return targets


def numbers(text: str) -> collections.Counter[str]:
    return collections.Counter(NUMBER.findall(text))


def code(text: str) -> list[str]:
    lines = []
    for block in FENCE.findall(text):
        for line in block.splitlines():
            stripped = COMMENT.sub("", line).rstrip()
            if stripped:
                lines.append(stripped)
    return lines


def switcher_targets(text: str) -> set[str]:
    """The link targets inside the switcher block, or an empty set when the
    block is missing (reported by the caller)."""
    match = SWITCHER_BLOCK.search(text)
    if match is None:
        return set()
    return {a or b for a, b in LINK.findall(match.group(1))}


def compare(source: str, translation: str) -> list[str]:
    problems = []
    src_links, tr_links = links(source, strip_prefix=False), links(translation, strip_prefix=True)
    for missing in sorted(src_links - tr_links):
        problems.append(f"link missing from translation: {missing}")
    for extra in sorted(tr_links - src_links):
        problems.append(f"link not in README.md: {extra}")
    src_numbers, tr_numbers = numbers(source), numbers(translation)
    for token, count in sorted((src_numbers - tr_numbers).items()):
        problems.append(f"number {token!r} appears {count} more time(s) in README.md than in the translation")
    for token, count in sorted((tr_numbers - src_numbers).items()):
        problems.append(f"number {token!r} appears {count} more time(s) in the translation than in README.md")
    src_code, tr_code = code(source), code(translation)
    if src_code != tr_code:
        first = next((i for i, (a, b) in enumerate(zip(src_code, tr_code)) if a != b), min(len(src_code), len(tr_code)))
        have = tr_code[first] if first < len(tr_code) else "<end of code>"
        want = src_code[first] if first < len(src_code) else "<end of code>"
        problems.append(f"code line {first + 1} differs: translation has {have!r}, README.md has {want!r}")
    return problems


def check_switchers(source: str, files: list[pathlib.Path]) -> list[str]:
    """README.md links every translation; every translation links README.md
    and every sibling (itself appears as plain bold text, not a link)."""
    problems = []
    names = [p.name for p in files]
    source_targets = switcher_targets(source)
    if not source_targets:
        problems.append("README.md: no language switcher block between <!-- translations:start --> and <!-- translations:end -->")
    for name in names:
        if f"docs/translations/{name}" not in source_targets:
            problems.append(f"README.md: the switcher does not link docs/translations/{name}")
    for path in files:
        targets = switcher_targets(path.read_text(encoding="utf-8"))
        if not targets:
            problems.append(f"{path.name}: no language switcher block")
            continue
        if "../../README.md" not in targets:
            problems.append(f"{path.name}: the switcher does not link ../../README.md")
        for name in names:
            if name != path.name and name not in targets:
                problems.append(f"{path.name}: the switcher does not link {name}")
    return problems


def main() -> int:
    files = translations()
    if not files:
        print("no translations under docs/translations/; nothing to check")
        return 0
    source = SOURCE.read_text(encoding="utf-8")
    failed = False
    for path in files:
        problems = compare(source, path.read_text(encoding="utf-8"))
        for p in problems:
            print(f"{path.name}: {p}")
        if problems:
            failed = True
        else:
            print(f"{path.name}: links, numbers and code match README.md")
    for p in check_switchers(source, files):
        print(p)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
