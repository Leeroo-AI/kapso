"""Record that production stands at the tip of the repository's base branch as it is now.

Run it once after the repository is pushed and production is created, when production holds exactly what that
commit builds (the untouched image), and again whenever the repository is started over from one commit. Refuses
while a release is recorded, since production then holds more than the starting commit. The state file is
`release.state_file` of the config; a missing one is production before any release.

Usage: python set_production_commit.py
"""
import json
import pathlib
import subprocess

from pool import load_config

SETTINGS = load_config()["release"]
STATE = pathlib.Path(SETTINGS["state_file"])

state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"releases": []}
if state["releases"]:
    raise SystemExit(f"{STATE} records {len(state['releases'])} release(s): roll them back before resetting the commit")
completed = subprocess.run(["gh", "api", f"repos/{SETTINGS['repository']}/commits/{SETTINGS['base_branch']}", "--jq", ".sha"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
if completed.returncode != 0:
    raise SystemExit(f"could not read {SETTINGS['base_branch']} of {SETTINGS['repository']}: {completed.stderr.strip()}")
state["released_commit"] = completed.stdout.strip()
STATE.parent.mkdir(parents=True, exist_ok=True)
STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
print(f"production stands at {state['released_commit']}")
