"""Open one GitHub issue per ticket of the backlog, labelled by area, from the repository's own ticket files.

Optional: the build works from the ticket files, and the issues are for the people who read the backlog on
GitHub. An issue's title is the ticket's first heading and its body the rest of the file, with a footer naming
the ticket file and the check command. A ticket whose title already has an open issue is skipped, so a rerun
adds nothing. Needs the `gh` CLI logged in with write access to the repository.

Usage: python open_issues.py <owner/repository> <repository directory> [--dry-run]
"""
import json
import pathlib
import subprocess
import sys

AREAS = {"revenue": "0e8a16", "customer": "1d76db", "inventory": "fbca04", "marketing": "d93f0b"}

arguments = [argument for argument in sys.argv[1:] if argument != "--dry-run"]
dry_run = "--dry-run" in sys.argv[1:]
if len(arguments) != 2:
    raise SystemExit("usage: python open_issues.py <owner/repository> <repository directory> [--dry-run]")
repository, tree = arguments[0], pathlib.Path(arguments[1])


def gh(arguments, text=None):
    completed = subprocess.run(["gh", *arguments], input=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"gh {' '.join(arguments[:3])} failed: {completed.stderr.strip()[:300]}")
    return completed.stdout


backlog = json.loads((tree / "checks" / "tickets.json").read_text())["tickets"]
existing_labels = {row["name"] for row in json.loads(gh(["label", "list", "--repo", repository, "--json", "name", "--limit", "100"]))}
open_titles = {row["title"] for row in json.loads(gh(["issue", "list", "--repo", repository, "--state", "open", "--json", "title", "--limit", "200"]))}
for area, colour in AREAS.items():
    if area not in existing_labels and not dry_run:
        gh(["label", "create", area, "--repo", repository, "--color", colour, "--description", f"{area.capitalize()} reporting"])
for ticket in backlog:
    text = (tree / "tickets" / f"{ticket['key']}.md").read_text()
    heading, _, body = text.partition("\n")
    title = heading.lstrip("# ").strip()
    if title != ticket["title"]:
        raise SystemExit(f"{ticket['key']}: the ticket file's heading and checks/tickets.json disagree")
    body = body.strip() + f"\n\n---\nTicket file: `tickets/{ticket['key']}.md` · check: `python checks/run_checks.py --ticket {ticket['key']}`\n"
    if title in open_titles:
        print(f"exists   {ticket['key']}")
        continue
    if dry_run:
        print(f"would open [{ticket['area']}] {title} ({len(body)} characters)")
        continue
    url = gh(["issue", "create", "--repo", repository, "--title", title, "--label", ticket["area"], "--body-file", "-"], text=body).strip()
    print(f"opened   {ticket['key']}  {url}")
