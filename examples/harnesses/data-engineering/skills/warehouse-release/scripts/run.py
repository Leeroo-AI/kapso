"""One call of the warehouse-release deployment: every released ticket's checks on production as it stands.

Reads the call's inputs as one JSON document on stdin (none are used) and writes one JSON document on stdout:
the release, its commit, the commit production stands at, and per released ticket how many checks pass and
which do not. The work is release.py's `run`; this file is the entry the engine calls.
"""
import subprocess
import sys
from pathlib import Path

completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "release.py"), "run"], stdin=sys.stdin)
sys.exit(completed.returncode)
