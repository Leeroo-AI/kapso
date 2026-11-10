"""Stop reminder: an implementation session is told once, before it ends, what its proofs lack and which
label its evidence earns. It catches a mistake; it is not a gate (the host lets the session end on its second
stop), and what a pull request may claim is decided again by verification and at publication.

The validation-report skill says every attempt writes results/proof.json with the kind of change, the
validation list and the cause, and that the label is computed from what was measured (evidence.py baseline
before the change, prove after it, label --write). A backlog (ticket-backlog) writes one such proof per closed
ticket under results/proofs/ and a ledger, results/backlog.json, with one row for every ticket of the
inventory; each closed row names the proof of its own ticket, of the kind its status says, and every row
carries the acceptance verdict as last run.

When the session tries to stop the hook checks, for every proof it must find: the fields and their shape (the
shape the pull-request body is rendered from, so a proof that passes here renders), that the ticket was
measured at all, and that the proof's label is the one evidence.py computes. A proof calling itself safe to
merge without the measurements is named with what is missing. The hook asks once: when the host reports that
the session is already continuing because of a stop hook, it lets the session end, so a session that truly
cannot finish is not trapped; the label a pull request carries is computed again at deployment whatever the
proof says. A file that is not valid JSON raises, which the host reports as the hook's error.
"""
import json
import subprocess
import sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[2] / "skills" / "validation-report" / "scripts" / "evidence.py"
PROOF = Path("results/proof.json")
LEDGER = Path("results/backlog.json")
PROOFS_DIR = Path("results/proofs")
BACKLOG_DIR = Path("prepared/backlog")
INVENTORY = BACKLOG_DIR / "inventory.json"
REQUIRED = ("model", "change_kind", "label", "validated", "changed_files")
LABELS = ("safe to merge", "unsure", "won't do")
KINDS = ("fix", "refactor", "performance", "new", "migration", "no_change")
STATUSES = ("done", "no_change", "parked", "open")
CLOSED = ("done", "no_change")
ACCEPTANCE = ("pass", "fail", "not_run")
REPORTED_PROBLEMS = 12
SINGLE_TICKET = "change"


def existing_file(value):
    return isinstance(value, str) and bool(value) and Path(value).is_file() and Path(value).stat().st_size > 0


def computed_label(ticket):
    """(label, reasons) as evidence.py computes them, or (None, why) when it could not run."""
    completed = subprocess.run([sys.executable, str(EVIDENCE), "label", "--ticket", ticket, "--json"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        return None, [completed.stderr.strip()[-300:] or "evidence.py label failed"]
    entry = json.loads(completed.stdout)[ticket]
    return entry["label"], entry["reasons"]


def proof_problems(path, ticket=None, status=None):
    """What one proof file lacks, each named with the file; for a ledger row, also that it is that ticket's
    proof and of the kind the row's status says."""
    proof = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(proof, dict):
        return [f"{path} is not an object"]
    problems = [f"{path}: `{key}` is missing" for key in REQUIRED if key not in proof]
    if proof.get("label") not in LABELS:
        problems.append(f"{path}: `label` must be one of {', '.join(LABELS)}")
    kind = proof.get("change_kind")
    if kind not in KINDS:
        problems.append(f"{path}: `change_kind` must be one of {', '.join(KINDS)}")
    if ticket is not None and proof.get("ticket") != ticket:
        problems.append(f"{path}: its `ticket` is {proof.get('ticket')!r}, but the ledger names it as the proof of `{ticket}`")
    if status is not None and (status == "no_change") != (kind == "no_change"):
        problems.append(f"{path}: the ledger calls `{ticket}` {status}, but the proof's `change_kind` is {kind!r}")
    validated = proof.get("validated")
    if not isinstance(validated, dict):
        problems.append(f"{path}: `validated` must be an object (commands, outcomes, not_inspected, skipped, assumptions)")
    else:
        for key in ("commands", "not_inspected", "skipped", "assumptions"):
            if key in validated and not isinstance(validated[key], list):
                problems.append(f"{path}: `validated.{key}` must be a list")
        skipped = validated.get("skipped") if isinstance(validated.get("skipped"), list) else []
        if not all(isinstance(item, dict) and item.get("check") and item.get("reason") for item in skipped):
            problems.append(f"{path}: every `validated.skipped` entry is an object with `check` and `reason`")
    blast = proof.get("blast_radius")
    if blast is not None and not (isinstance(blast, dict) and isinstance(blast.get("changed_models", []), list)):
        problems.append(f"{path}: `blast_radius` is an object whose `changed_models` lists {{model, reason}} entries")
    if kind == "no_change":
        if proof.get("changed_files") != []:
            problems.append(f"{path}: a no_change outcome has an empty `changed_files`")
        evidence = proof.get("decision_evidence")
        if not isinstance(evidence, list) or not evidence or not all(existing_file(item) for item in evidence):
            problems.append(f"{path}: a no_change outcome lists `decision_evidence` naming existing files")
        return problems
    if kind == "fix":
        cause = proof.get("root_cause")
        if not isinstance(cause, dict) or not cause.get("file") or not cause.get("reproducing_query"):
            problems.append(f"{path}: a fix names `root_cause.file` and `root_cause.reproducing_query`")
    name = ticket or proof.get("ticket") or SINGLE_TICKET
    if not Path("results", name, "evidence.json").is_file():
        problems.append(f"{path}: `{name}` was never measured (no results/{name}/evidence.json): evidence.py baseline before the change, "
                        "prove after it; a ticket already changed is measured from where it stands and labelled by what that shows")
        return problems
    label, reasons = computed_label(name)
    if label is None:
        problems.append(f"{path}: the label could not be computed ({reasons[0]})")
    elif proof.get("label") != label:
        problems.append(f"{path}: `label` says {proof.get('label')!r} and the evidence earns {label!r}"
                        + (f" ({reasons[0][:200]})" if reasons and label != "safe to merge" else "")
                        + f": measure what is missing, or run evidence.py label --ticket {name} --write")
    return problems


def ticket_rows(path, document):
    """The rows of an inventory or a ledger as (ticket, row) pairs, and what is wrong with the list itself:
    a row that is not an object, a ticket key that is not a non-empty string, a key that appears twice."""
    rows = document.get("tickets") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        return [], [f"{path} holds no non-empty list of tickets"]
    pairs, problems, seen = [], [], set()
    for position, row in enumerate(rows, 1):
        ticket = row.get("ticket") if isinstance(row, dict) else None
        if not isinstance(ticket, str) or not ticket:
            problems.append(f"{path}: row {position} is not an object with a non-empty string `ticket`")
        elif ticket in seen:
            problems.append(f"{path}: ticket `{ticket}` appears more than once")
        else:
            seen.add(ticket)
            pairs.append((ticket, row))
    return pairs, problems


def backlog_problems():
    """What the inventory, the ledger and the per-ticket proofs lack."""
    if not INVENTORY.is_file():
        return [f"{INVENTORY} does not exist, though this attempt has backlog files (the setup session writes the inventory)"]
    inventory, problems = ticket_rows(INVENTORY, json.loads(INVENTORY.read_text(encoding="utf-8")))
    if not LEDGER.is_file():
        return problems + [f"{LEDGER} does not exist (one row per ticket of the inventory: status, acceptance, proof)"]
    ledger, ledger_problems = ticket_rows(LEDGER, json.loads(LEDGER.read_text(encoding="utf-8")))
    problems += ledger_problems
    expected = {ticket for ticket, _ in inventory}
    listed = {ticket for ticket, _ in ledger}
    problems += [f"{LEDGER}: no row for ticket `{ticket}`" for ticket, _ in inventory if ticket not in listed]
    problems += [f"{LEDGER}: `{ticket}` is not a ticket of the inventory" for ticket, _ in ledger if ticket not in expected]
    for ticket, entry in ledger:
        status = entry.get("status")
        if status not in STATUSES:
            problems.append(f"{LEDGER}: `{ticket}` has status {status!r}, not one of {', '.join(STATUSES)}")
            continue
        if entry.get("acceptance") not in ACCEPTANCE:
            problems.append(f"{LEDGER}: `{ticket}` needs `acceptance` as last run: one of {', '.join(ACCEPTANCE)}")
        if status in CLOSED:
            proof = entry.get("proof")
            if not existing_file(proof):
                problems.append(f"{LEDGER}: `{ticket}` is {status} and names no existing proof file")
            else:
                problems += proof_problems(Path(proof), ticket=ticket, status=status)
        elif status == "parked" and not entry.get("note"):
            problems.append(f"{LEDGER}: `{ticket}` is parked without a note (what was reproduced, ruled out, the next question)")
    return problems


event = json.load(sys.stdin)
if event.get("stop_hook_active"):
    sys.exit(0)
if INVENTORY.is_file() or BACKLOG_DIR.is_dir() or LEDGER.is_file() or PROOFS_DIR.is_dir():
    problems = backlog_problems()
    what = ("results/backlog.json with one row for every ticket (status, acceptance) and results/proofs/<ticket>.json "
            "for each ticket the ledger calls done or no_change")
elif PROOF.is_file():
    problems = proof_problems(PROOF)
    what = "results/proof.json"
else:
    problems = ["results/proof.json does not exist"]
    what = "results/proof.json"
if not problems:
    sys.exit(0)
shown = problems[:REPORTED_PROBLEMS] + ([f"and {len(problems) - REPORTED_PROBLEMS} more"] if len(problems) > REPORTED_PROBLEMS else [])
print(json.dumps({
    "decision": "block",
    "reason": (
        "The attempt has no complete proof: " + "; ".join(shown) + f". Write {what} as the validation-report skill "
        "specifies (the kind of change, what was validated and how, the cause with file and lines or the decision "
        "evidence, the measurement context, the unknowns), with the label evidence.py computes from what was "
        "measured, then finish. A label is never chosen: `unsure` with what is missing is the honest state of a "
        "change that passes its check and has not been measured; a ticket of a backlog that was not closed stays "
        "`open` or `parked` in the ledger with its note."
    ),
}))
