"""PostToolUse hook on the file tools: a change to a shared layer is a
decision with a blast radius.

Advisory, not a denial: the write has happened. Staging models and sources
are what every consumer reads, snapshots carry history that configuration
cannot recreate, seeds are data, contracts and schemas bind other teams,
migrations and DAG definitions change what runs, and an existing test changed
without independent evidence is a finding in verification. The hook names the
file and what the proof must now carry, so the session decides while the
change is fresh rather than meeting it in verification. It names possible
impact; the measured comparison establishes what changed.

Both hosts' event shapes: Claude Code's file tools carry `file_path`; the
Codex `apply_patch` tool carries the patch text in `command`, whose header
lines name the files written.
"""
import json
import re
import sys
from pathlib import Path

PATCH_FILE = re.compile(r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", re.MULTILINE)
NOTES = {
    "staging": "a shared transformation: identify its downstream consumers and measure which outputs change; the proof's blast radius covers the whole downstream set",
    "sources": "a source definition: confirm the resolved input relation and assess the affected consumers; the proof shows the source's row count and keys before and after",
    "snapshots": "a history-bearing dataset: preserve the captured history, verify key uniqueness and non-overlapping validity intervals, and identify a retained history source before any rebuild",
    "seeds": "reference data: identify its consumers and explain the authoritative basis for the change",
    "contracts": "a contract: other teams depend on it; the proof names the consumers and the migration",
    "schemas": "a schema definition: types and columns bind the consumers; the proof shows the structure before and after",
    "migrations": "a migration: it runs once in production; the proof carries the runbook line and the rollback",
    "dags": "an orchestration definition: it changes what runs and when; the proof states the schedule and the rerun safety",
    "tests": "a test: a change to an existing expectation needs independent evidence for the corrected contract, in the proof; a test changed to make a failure go away is a finding",
}

event = json.load(sys.stdin)
tool_input = event.get("tool_input", {})
written = [tool_input["file_path"]] if tool_input.get("file_path") else PATCH_FILE.findall(tool_input.get("command", ""))
notes = []
for file_path in written:
    parts = Path(file_path).parts
    for directory, note in NOTES.items():
        if directory in parts:
            notes.append(f"{file_path} is {note}")
            break
if not notes:
    sys.exit(0)
print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "shared-layer-notice: " + " ".join(notes)}}))
