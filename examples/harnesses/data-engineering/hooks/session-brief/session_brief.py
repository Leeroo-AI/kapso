"""SessionStart hook: what the setup session pinned, and where this attempt stands.

It reads prepared/environment.json (which targets and commands the evidence scripts will use), for one ticket
prepared/baseline/summary.json (backend, model, grain, the kind of change, the downstream count, the prepared
spec) and results/proof.json, and for a backlog the inventory under prepared/backlog/ and the ledger
results/backlog.json (how many tickets, how many in each status). It adds a short brief to the session's
context. Before setup has run there is nothing to say, and the hook says nothing. A file that is not valid
JSON raises, which the host reports as the hook's error; a file of another shape (a row that is not an object,
a ticket named twice, a status that is not a word) is reported as unreadable so the session inspects it.
"""
import json
import sys
from pathlib import Path

ENVIRONMENT = Path("prepared/environment.json")
SUMMARY = Path("prepared/baseline/summary.json")
PROOF = Path("results/proof.json")
INVENTORY = Path("prepared/backlog/inventory.json")
LEDGER = Path("results/backlog.json")


def ticket_rows(document):
    """The rows of an inventory or a ledger when every one is an object with its own non-empty string
    `ticket`; None when the list has another shape."""
    rows = document.get("tickets") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        return None
    tickets = [row.get("ticket") if isinstance(row, dict) else None for row in rows]
    if not all(isinstance(ticket, str) and ticket for ticket in tickets) or len(set(tickets)) != len(tickets):
        return None
    return rows


event = json.load(sys.stdin)
parts = []
if ENVIRONMENT.is_file():
    pinned = json.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    if isinstance(pinned, dict):
        parts.append(
            f"Environment (prepared/environment.json): builds on target `{pinned.get('build_target')}`, production is read through "
            f"`{pinned.get('reference_target')}`; a reset command is {'pinned' if pinned.get('reset_command') else 'not pinned'}, "
            f"an acceptance check per ticket is {'pinned' if pinned.get('acceptance_command') else 'not pinned'}, "
            f"a release command is {'pinned' if pinned.get('release_command') else 'not pinned'}."
        )
    else:
        parts.append("prepared/environment.json has an unexpected shape; inspect it before measuring anything.")
if SUMMARY.is_file():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if isinstance(summary, dict) and summary.get("model"):
        parts.append(
            f"Prepared for one ticket: model {summary.get('model')} on {summary.get('backend')}, grain {summary.get('grain')}, "
            f"change kind {summary.get('change_kind') or 'unstated'}, {summary.get('downstream_models')} downstream dataset(s); "
            f"the spec is {summary.get('spec')}, and the implementation takes the baseline before it edits a file."
        )
    else:
        parts.append("prepared/baseline/summary.json has an unexpected shape; inspect it before relying on it.")
if INVENTORY.is_file():
    rows = ticket_rows(json.loads(INVENTORY.read_text(encoding="utf-8")))
    if rows is not None:
        parts.append(f"Backlog: {len(rows)} ticket(s) in prepared/backlog/inventory.json; each ticket's baseline is taken when it is worked.")
    else:
        parts.append("prepared/backlog/inventory.json has an unexpected shape; inspect it before relying on the inventory.")
if LEDGER.is_file():
    rows = ticket_rows(json.loads(LEDGER.read_text(encoding="utf-8")))
    statuses = [row.get("status") for row in rows] if rows is not None else None
    if statuses is not None and all(isinstance(status, str) for status in statuses):
        counts = {status: statuses.count(status) for status in sorted(set(statuses))}
        parts.append("Ledger (results/backlog.json): " + ", ".join(f"{count} {status}" for status, count in counts.items()) + ".")
    else:
        parts.append("results/backlog.json has an unexpected shape; inspect it before relying on the ledger.")
if PROOF.is_file():
    proof = json.loads(PROOF.read_text(encoding="utf-8"))
    cause = proof.get("root_cause") if isinstance(proof, dict) else None
    unknowns = proof.get("unknowns", []) if isinstance(proof, dict) else None
    if isinstance(unknowns, list):
        parts.append(
            f"This attempt's proof: label `{proof.get('label')}` ({proof.get('change_kind')}), "
            f"cause in {cause.get('file') if isinstance(cause, dict) else None}, {len(unknowns)} question(s) for the author."
        )
    else:
        parts.append("results/proof.json has an unexpected shape; inspect it before relying on its label.")
if not parts:
    sys.exit(0)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": " ".join(parts)}}))
