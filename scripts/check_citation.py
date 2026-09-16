#!/usr/bin/env python3
"""Keep CITATION.cff in step with the released version.

GitHub's "Cite this repository" button, Zenodo and Zotero all read CITATION.cff,
so a stale `version` there ends up as a wrong citation in someone's paper. This
check fails when CITATION.cff's `version` differs from the one in pyproject.toml;
`--fix` rewrites `version` and `date-released` (today, UTC) for a release.

Stdlib only, like the other checks under scripts/.

  python3 scripts/check_citation.py          # CI: exit 1 on drift
  python3 scripts/check_citation.py --fix    # at release time, then commit
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"


def pyproject_version() -> str:
    text = PYPROJECT.read_text()
    # The [project] table's own version line — not a dependency pin, not a tool table.
    table = re.search(r"^\[project\]\n(.*?)(?=^\[)", text, re.S | re.M)
    m = re.search(r'^version\s*=\s*"([^"]+)"', table.group(1) if table else text, re.M)
    if not m:
        sys.exit("pyproject.toml: no version under [project]")
    return m.group(1)


def citation_field(text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip().strip("\"'") if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="set version from pyproject.toml and date-released to today")
    args = ap.parse_args()

    want = pyproject_version()
    text = CITATION.read_text()
    have = citation_field(text, "version")
    date = citation_field(text, "date-released")

    if args.fix:
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        text = re.sub(r"^version:.*$", f"version: {want}", text, count=1, flags=re.M)
        text = re.sub(r"^date-released:.*$", f"date-released: {today}", text, count=1, flags=re.M)
        CITATION.write_text(text)
        print(f"CITATION.cff: version {have} -> {want}, date-released {date} -> {today}")
        return 0

    problems = []
    if have != want:
        problems.append(f"version is {have!r} but pyproject.toml says {want!r} — run: python3 scripts/check_citation.py --fix")
    if not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        problems.append(f"date-released {date!r} is not a YYYY-MM-DD date")
    for p in problems:
        print("CITATION.cff:", p)
    if not problems:
        print(f"CITATION.cff: version {have} matches pyproject.toml")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
