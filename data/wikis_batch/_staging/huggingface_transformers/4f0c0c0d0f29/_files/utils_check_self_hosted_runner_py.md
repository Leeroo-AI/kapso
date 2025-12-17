# File: `utils/check_self_hosted_runner.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 57 |
| Functions | `get_runner_status` |
| Imports | argparse, json, subprocess |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Monitors GitHub Actions self-hosted runner availability for CI infrastructure.

**Mechanism:** Uses curl to query GitHub API endpoint `/repos/huggingface/transformers/actions/runners`, compares runner names against a target list provided via command line, and identifies offline runners. Saves results to `offline_runners.txt` for Slack notifications and raises an exception if any target runners are offline.

**Significance:** Infrastructure monitoring tool ensuring CI pipeline reliability by detecting when self-hosted GitHub Actions runners go offline, enabling proactive alerting before test failures occur.
