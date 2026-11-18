"""`kapso harness`: author, check and fetch harnesses, the domain layer a
build loads (docs/research/harness-design.md, onboarding.md section 1)."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from kapso.core.config import load_config
from kapso.harness import (
    BUNDLED_ROOT,
    HarnessStore,
    check_harness,
    parse_reference,
    read_manifest,
)
from kapso.harness.export import HOSTS, export_harness
from kapso.harness.manifest import MANIFEST_FILE, valid_name

NAME = "harness"
TEMPLATES = Path(__file__).resolve().parent.parent / "harness" / "templates"


def cmd_harness(args) -> None:
    settings = load_config(args.config)["harness"]
    store = HarnessStore(Path(settings["store"]))
    command = args.harness_command
    if command == "new":
        _new(args.name, Path(args.dir).expanduser() if args.dir else Path.cwd())
        return
    if command == "list":
        _list(store, settings, as_json=args.json)
        return
    reference = parse_reference(args.ref)
    if command == "remove":
        print(f"removed {store.remove(reference)}")
        return
    if command == "update":
        before, after = store.update(reference)
        if before == after:
            print(f"at {after[:12]}: a tag or a commit never moves")
        else:
            print(f"now at {after[:12]} (was {before[:12]})")
    # check, add, update and export: resolve (fetching when needed, and
    # saying so), then the report; a failure exits 1, what the machine
    # lacks never does. An export shows the report only when it refuses.
    fetched_before = reference.kind != "git" or store.is_fetched(reference)
    path = store.resolve(reference)
    if not fetched_before:
        print(f"fetched {path}")
    report = check_harness(path, page_warning_lines=int(settings["page_warning_lines"]))
    if command == "export":
        if not report.ok():
            print(report.render())
            sys.exit(1)
        print(export_harness(path, args.host, Path(args.to).expanduser() if args.to else Path.cwd()).render())
        return
    print(report.render())
    if report.manifest is not None:
        print(f"name     {report.manifest.name} {report.manifest.version}")
    if not report.ok():
        sys.exit(1)


def _new(name: str, parent: Path) -> None:
    """The two-file starter the authoring guide begins with."""
    if not valid_name(name):
        print(f"Error: a harness name matches [a-z0-9][a-z0-9-]*, not {name!r}", file=sys.stderr)
        sys.exit(1)
    root = parent / name
    if root.exists():
        print(f"Error: {root} exists; choose another name or directory", file=sys.stderr)
        sys.exit(1)
    (root / "instructions").mkdir(parents=True)
    (root / MANIFEST_FILE).write_text(
        (TEMPLATES / "harness.yaml").read_text(encoding="utf-8").replace("{name}", name), encoding="utf-8",
    )
    (root / "instructions" / "guardrails.md").write_text(
        (TEMPLATES / "guardrails.md").read_text(encoding="utf-8").replace("{title}", name.replace("-", " ").capitalize()),
        encoding="utf-8",
    )
    print(f"created {root}")
    print(f"  {MANIFEST_FILE}                 name, version, description; the module lists to fill")
    print("  instructions/guardrails.md   the rules every module's sessions read")
    print(f"next: edit the two files, then  kapso harness check {_shown(root)}")


def _list(store: HarnessStore, settings, *, as_json: bool) -> None:
    rows = _listing(store, settings)
    if as_json:
        print(json.dumps(rows, indent=1))
        return
    if not rows:
        print("no harnesses yet: write one with `kapso harness new <name>`, or fetch one with "
              "`kapso harness add <git reference>`; a local directory loads by path, no add needed")
        return
    bundled = [row for row in rows if row["source"] == "bundled"]
    fetched = [row for row in rows if row["source"] == "fetched"]
    if bundled:
        print("bundled")
        for row in bundled:
            print(f"  {row['name']:<24} {row['version']:<10} {_needs(row)}")
    else:
        print("bundled    (none ship in this release)")
    if fetched:
        print("fetched")
        for row in fetched:
            print(f"  {row['name']:<24} {row['version']:<10} {row['reference']}  {_needs(row)}")


def _listing(store: HarnessStore, settings) -> List[Dict[str, Any]]:
    """Every bundled and fetched harness with the state its check gives."""
    lines = int(settings["page_warning_lines"])
    bundled = sorted(p for p in BUNDLED_ROOT.iterdir() if (p / MANIFEST_FILE).is_file()) if BUNDLED_ROOT.is_dir() else []
    harnesses = []
    for path in bundled:
        manifest = read_manifest(path)
        harnesses.append((manifest.name, manifest.version, "bundled", None, path))
    harnesses += [(entry.name, entry.version, "fetched", entry.reference, entry.path) for entry in store.entries()]
    rows = []
    for name, version, source, reference, path in harnesses:
        report = check_harness(path, page_warning_lines=lines)
        rows.append({
            "name": name,
            "version": version,
            "source": source,
            "reference": reference,
            "path": str(path),
            "state": "fails" if not report.ok() else "needs" if report.missing else "ready",
            "needs": [row.label for row in report.missing],
            "failures": len(report.failures),
        })
    return rows


def _needs(row: Dict[str, Any]) -> str:
    if row["state"] == "fails":
        return f"fails check ({row['failures']} failure(s))"
    if row["state"] == "needs":
        return "needs " + ", ".join(row["needs"])
    return "ready"


def _shown(path: Path) -> str:
    """A path as a person would type it: relative when it is under the
    working directory, else absolute; a bare name would be read as a
    bundled one."""
    cwd = Path.cwd()
    if path.resolve().is_relative_to(cwd):
        return f"./{path.resolve().relative_to(cwd).as_posix()}"
    return str(path.resolve())


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "harness",
        help="Author, check and fetch harnesses: the domain layer a build loads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        description=(
            "A harness is one directory per domain: instruction pages, skills, "
            "tool servers, delegates and hooks, listed per module in harness.yaml. "
            "A build loads one by bundled name, by path, or by git reference."
        ),
        epilog="""
Examples:
  kapso harness new trading                       # harness.yaml + instructions/guardrails.md
  kapso harness check ./trading                   # failures, warnings, what this machine lacks
  kapso harness add acme/harnesses@v3#trading     # fetch into the store and check
  kapso harness list                              # bundled and fetched harnesses
  kapso harness export ./trading --host codex     # its skills, delegates and servers as a session's files
  kapso build --harness ./trading --brief kapso-brief.md ...
""",
    )
    sub = parser.add_subparsers(dest="harness_command", required=True)
    new = sub.add_parser("new", help="Write the two-file starter for a new harness")
    new.add_argument("name", type=str, help="The harness name: [a-z0-9][a-z0-9-]*")
    new.add_argument("--dir", type=str, default=None, help="Parent directory (default: the working directory)")
    for verb, text in (
        ("check", "Check a harness: failures, warnings, and what this machine lacks"),
        ("add", "Fetch a git reference into the store (a path is checked in place) and check it"),
        ("update", "Re-fetch a branch or HEAD reference; a tag or a commit never moves"),
        ("remove", "Delete one fetched clone, every harness in it; a path is refused"),
    ):
        command = sub.add_parser(verb, help=text)
        command.add_argument("ref", type=str, help="A bundled name, a directory, or a git reference")
        command.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    listing = sub.add_parser("list", help="The bundled and fetched harnesses, with what each needs")
    listing.add_argument("--json", action="store_true", help="Print the list as JSON: name, version, source, reference, path, state, needs, failures")
    listing.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    export = sub.add_parser(
        "export",
        help="Write a harness's skills, delegates and servers as the files an ordinary session of one host discovers",
    )
    export.add_argument("ref", type=str, help="A bundled name, a directory, or a git reference")
    export.add_argument("--host", type=str, required=True, choices=HOSTS, help="The host whose files to write")
    export.add_argument("--to", type=str, default=None, help="The directory to write into (default: the working directory)")
    export.add_argument("--config", type=str, default=None, help="Config path (default: packaged config.yaml)")
    new.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS)
    return cmd_harness
