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

**Purpose:** Monitors GitHub Actions self-hosted runners to ensure they remain online and available for CI/CD workflows.

**Mechanism:** Uses GitHub's REST API to query the status of specified self-hosted runners for the Transformers repository. Checks each target runner's status field and raises an error if any are offline, saving the list of offline runners to a file for Slack notification integration.

**Significance:** Prevents CI/CD pipeline failures due to unavailable self-hosted runners. Critical for maintaining reliable continuous integration when using custom hardware (like specific GPUs) that aren't available in GitHub's standard hosted runners.
