#!/usr/bin/env python3
"""Fail when the reference docs do not mention part of the public surface.

The CLI reference sat at one of eight commands for seven months because
nothing connected `cli.py` to `docs/reference/`. This closes that loop: add a
command or a public method and the docs build fails until it is written up.

Reads argparse subparser names out of cli.py and public method names out of
kapso.py, then checks each one appears in the matching reference page. Text
presence is a deliberately shallow test — it cannot tell whether the prose is
any good, only whether the surface was acknowledged at all, which is the
failure that actually happened.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLI_SOURCE = REPO / "src" / "kapso" / "cli.py"
API_SOURCE = REPO / "src" / "kapso" / "kapso.py"
CLI_DOC = REPO / "docs" / "reference" / "cli.mdx"
API_DOC = REPO / "docs" / "reference" / "kapso-api.mdx"

# Each entry maps the argparse variable holding a subparser group to the
# command prefix its names appear under in the docs.
SUBPARSER_GROUPS = (
    ("subparsers", "kapso"),
    ("learn_sub", "kapso learn"),
    ("bank_sub", "kapso bank"),
)


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"expected {path} to exist")
    return path.read_text(encoding="utf-8")


def subcommands(source: str, variable: str) -> list[str]:
    pattern = re.escape(variable) + r'\.add_parser\(\s*\n?\s*"([a-z][a-z_-]*)"'
    return sorted(set(re.findall(pattern, source)))


def public_methods(source: str) -> list[str]:
    found = re.findall(r"\n    def ([a-z][a-zA-Z_]*)\(", source)
    return [name for name in found if not name.startswith("_")]


def main() -> int:
    cli_source = read(CLI_SOURCE)
    api_source = read(API_SOURCE)
    cli_doc = read(CLI_DOC)
    api_doc = read(API_DOC)

    missing: list[str] = []
    checked = 0

    for variable, prefix in SUBPARSER_GROUPS:
        for name in subcommands(cli_source, variable):
            checked += 1
            if f"{prefix} {name}" not in cli_doc:
                missing.append(f"{CLI_DOC.name} does not mention `{prefix} {name}`")

    for name in public_methods(api_source):
        checked += 1
        if name not in api_doc:
            missing.append(f"{API_DOC.name} does not mention `{name}`")

    if missing:
        print(f"Reference docs are missing {len(missing)} of {checked} public names:\n")
        for line in missing:
            print(f"  {line}")
        print("\nDocument the new surface in docs/reference/, then rerun.")
        return 1

    print(f"Reference covers all {checked} public commands and methods.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
