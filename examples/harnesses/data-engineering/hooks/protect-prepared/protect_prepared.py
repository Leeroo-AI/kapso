"""PreToolUse hook on the file tools: prepared resources are read-only.

The baseline invariants and fingerprints under prepared/ are what a fix is
measured against. A session that wants new measurements writes them under
results/. This hook catches writes made through the file tools and says why;
it does not prevent shell writes or establish filesystem isolation, and it is
not mounted on setup, which creates prepared/.

Both hosts' event shapes: Claude Code's file tools carry `file_path`; the
Codex `apply_patch` tool carries the patch text in `command`, whose header
lines name the files. Paths are resolved on both sides, so a `prepared` that
is a symlink into the campaign still matches.
"""
import json
import re
import sys
from pathlib import Path

PREPARED = (Path.cwd() / "prepared").resolve()
PATCH_FILE = re.compile(r"^\*\*\* (?:(?:Add|Update|Delete) File|Move to): (.+)$", re.MULTILINE)

event = json.load(sys.stdin)
tool_input = event.get("tool_input", {})
paths = [tool_input["file_path"]] if tool_input.get("file_path") else PATCH_FILE.findall(tool_input.get("command", ""))
touched = [
    path for path in paths
    if (target := Path.cwd().joinpath(path).resolve()) == PREPARED or PREPARED in target.parents
]
if not touched:
    sys.exit(0)
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"{', '.join(touched)} is under prepared/, which is read-only: the baseline invariants and "
            "fingerprints are what the fix is measured against. Write this attempt's measurements under results/."
        ),
    }
}))
