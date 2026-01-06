# NotificationServiceDocTests

## Metadata
| Key | Value |
|-----|-------|
| **Source** | `utils/notification_service_doc_tests.py` |
| **Repository** | huggingface/transformers |
| **Domains** | CI/CD, Testing, Notifications |
| **Last Updated** | 2025-12-18 |

## Overview

The NotificationServiceDocTests module is responsible for posting documentation test results to Slack. It retrieves test artifacts from GitHub Actions, parses test results, and creates formatted Slack messages with detailed failure information. The module is designed to run as part of the Transformers CI/CD pipeline after documentation tests complete.

## Description

This implementation provides a comprehensive notification system for documentation test results:

**Core Capabilities:**
- Retrieves and parses documentation test artifacts from GitHub Actions
- Extracts test statistics (passed, failed, time spent) from test output
- Identifies specific test failures with error messages from doctest output
- Creates formatted Slack messages with failure summaries and details
- Posts main notification and threaded replies with detailed failure information
- Links to GitHub Actions job pages for deeper investigation

**Key Components:**
- `Message` class: Constructs and posts Slack messages with test results
- `handle_test_results()`: Parses pytest output to extract test statistics
- `extract_first_line_failure()`: Extracts the first line of each doctest failure
- `retrieve_artifact()`: Loads test result files from artifact directories
- `retrieve_available_artifacts()`: Scans for available test artifact directories

**Integration Points:**
- GitHub Actions workflow environment variables
- Slack API via `slack_sdk.WebClient`
- CI error statistics via `get_ci_error_statistics.get_jobs()`
- Test artifact files (stats, failures_short, summary_short)

## Usage

### Command Line Usage

```bash
# Set required environment variables
export CI_SLACK_BOT_TOKEN="<slack-bot-token>"
export SLACK_REPORT_CHANNEL="<channel-id>"
export GITHUB_RUN_ID="<run-id>"
export ACCESS_REPO_INFO_TOKEN="<github-token>"

# Run from the directory containing test artifacts
python utils/notification_service_doc_tests.py
```

### Typical CI Workflow Integration

```yaml
# In GitHub Actions workflow
- name: Download test artifacts
  uses: actions/download-artifact@v3
  with:
    path: artifacts/

- name: Send test notifications
  env:
    CI_SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
    SLACK_REPORT_CHANNEL: ${{ secrets.SLACK_CHANNEL }}
    GITHUB_RUN_ID: ${{ github.run_id }}
    ACCESS_REPO_INFO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    cd artifacts
    python ../utils/notification_service_doc_tests.py
```

## Code Reference

### Main Functions

```python
def handle_test_results(test_results: str) -> tuple[int, int, str]:
    """
    Parse pytest output to extract test statistics.

    Args:
        test_results: Raw test output string containing pass/fail counts

    Returns:
        Tuple of (failed_count, success_count, time_spent)
    """

def extract_first_line_failure(failures_short_lines: str) -> dict[str, str]:
    """
    Extract the first line of each doctest failure.

    Args:
        failures_short_lines: Concatenated failure output from pytest

    Returns:
        Dictionary mapping test names to first line of error message
    """

def retrieve_artifact(name: str) -> dict[str, str]:
    """
    Load all text files from an artifact directory.

    Args:
        name: Directory name containing artifact files

    Returns:
        Dictionary mapping filename (without extension) to file contents
    """

def retrieve_available_artifacts() -> dict[str, Artifact]:
    """
    Scan current directory for available test artifacts.

    Returns:
        Dictionary of artifact names to Artifact objects
    """
```

### Message Class

```python
class Message:
    """Constructs and posts Slack messages for test results."""

    def __init__(self, title: str, doc_test_results: dict):
        """
        Initialize message with test results.

        Args:
            title: Header text for the Slack message
            doc_test_results: Dictionary of job results with failure details
        """

    @property
    def time(self) -> str:
        """Calculate total time spent across all test jobs."""

    @property
    def header(self) -> dict:
        """Generate Slack header block."""

    @property
    def no_failures(self) -> dict:
        """Generate success message block."""

    @property
    def failures(self) -> dict:
        """Generate failure summary block."""

    @property
    def category_failures(self) -> list[dict]:
        """Generate detailed failure blocks by category."""

    @property
    def payload(self) -> str:
        """Generate complete Slack message payload."""

    def post(self):
        """Post main notification message to Slack."""

    def post_reply(self):
        """Post detailed failure information as thread replies."""

    @staticmethod
    def error_out():
        """Post generic error message when test execution fails."""
```

### Key Imports

```python
import json
import os
import re
import time

from get_ci_error_statistics import get_jobs
from slack_sdk import WebClient
```

## I/O Contract

### Inputs

| Input | Type | Source | Description |
|-------|------|--------|-------------|
| `CI_SLACK_BOT_TOKEN` | Environment Variable | GitHub Secrets | Slack bot authentication token |
| `SLACK_REPORT_CHANNEL` | Environment Variable | GitHub Secrets | Target Slack channel ID |
| `GITHUB_RUN_ID` | Environment Variable | GitHub Actions | Current workflow run ID |
| `ACCESS_REPO_INFO_TOKEN` | Environment Variable | GitHub Secrets | GitHub API token for job information |
| Artifact Directories | Directory | Download Artifact Action | Directories containing test result files |
| `stats` file | Text File | Pytest | Test execution statistics |
| `failures_short` file | Text File | Pytest | Short failure descriptions |
| `summary_short` file | Text File | Pytest | Test summary with FAILED lines |

### Outputs

| Output | Type | Destination | Description |
|--------|------|-------------|-------------|
| Slack Messages | API Call | Slack Channel | Formatted test result notifications |
| Thread Replies | API Call | Slack Thread | Detailed failure information per job |
| `doc_test_results.json` | JSON File | Artifact Directory | Structured test results for upload |
| Console Output | stdout | CI Logs | Payload previews and status messages |

### Side Effects

- Creates `doc_test_results/` directory in working directory
- Posts messages to configured Slack channel
- Adds 1-second delays between thread replies to avoid rate limits
- Reads GitHub Actions job information via GitHub API

## Usage Examples

### Example 1: Parsing Test Results

```python
# Test output string from pytest
test_output = "== 45 passed, 3 failed in 1:23:45 =="

failed, success, time_spent = handle_test_results(test_output)
# failed = 3
# success = 45
# time_spent = "1:23:45"
```

### Example 2: Extracting Failures

```python
# Failure output from doctest
failure_text = """
FAILED test_model.py::test_forward [doctest] test_model.py::test_forward
123 Expected: tensor([1, 2, 3])
124 Got: tensor([1, 2, 4])
"""

failures = extract_first_line_failure(failure_text)
# failures = {"test_model.py::test_forward": "Expected: tensor([1, 2, 3])"}
```

### Example 3: Creating and Posting Message

```python
# Build test results dictionary
doc_test_results = {
    "src/transformers/models/bert": {
        "n_failures": 2,
        "n_success": 48,
        "time_spent": "5:34.21, ",
        "failed": ["test_bert_model", "test_bert_tokenizer"],
        "failures": {
            "test_bert_model": "AssertionError: shapes do not match",
            "test_bert_tokenizer": "KeyError: 'vocab_file'"
        },
        "job_link": "https://github.com/huggingface/transformers/actions/runs/123/jobs/456",
        "category": "Python Examples"
    }
}

# Create and post message
message = Message("[INFO] Results of the doc tests.", doc_test_results)
message.post()  # Posts main message
message.post_reply()  # Posts detailed failures in thread
```

### Example 4: Artifact Retrieval

```python
# Retrieve artifacts from a specific directory
artifact = retrieve_artifact("doc_tests_gpu_test_reports_src_models_bert")
# artifact = {
#     "stats": "== 45 passed, 3 failed in 1:23:45 ==",
#     "failures_short": "...",
#     "summary_short": "..."
# }

# Get all available artifacts
available = retrieve_available_artifacts()
for artifact_obj in available.values():
    print(f"Found artifact: {artifact_obj.name}")
    for path_info in artifact_obj.paths:
        print(f"  Path: {path_info['path']}")
```

### Example 5: Main Execution Flow

```python
# Main execution (from __main__ block)
SLACK_REPORT_CHANNEL_ID = os.environ["SLACK_REPORT_CHANNEL"]

# Get GitHub Actions job information
github_actions_jobs = get_jobs(
    workflow_run_id=os.environ["GITHUB_RUN_ID"],
    token=os.environ["ACCESS_REPO_INFO_TOKEN"]
)

# Map artifact names to jobs
artifact_name_to_job_map = {}
for job in github_actions_jobs:
    for step in job["steps"]:
        if step["name"].startswith("Test suite reports artifacts: "):
            artifact_name = step["name"][len("Test suite reports artifacts: "):]
            artifact_name_to_job_map[artifact_name] = job
            break

# Process all artifacts
available_artifacts = retrieve_available_artifacts()
doc_test_results = {}

for artifact_obj in available_artifacts.values():
    artifact_path = artifact_obj.paths[0]
    if not artifact_path["path"].startswith("doc_tests_gpu_test_reports_"):
        continue

    # Extract job name from artifact path
    job_name = artifact_path["path"].replace("doc_tests_gpu_test_reports_", "").replace("_", "/")

    # Process results...
    job_result = {}
    doc_test_results[job_name] = job_result

# Create and post notification
message = Message("[INFO] Results of the doc tests.", doc_test_results)
message.post()
message.post_reply()
```

## Related Pages

- [Tests Fetcher Implementation](/wikis/huggingface_transformers_TestsFetcher.md)
- [Update Metadata Implementation](/wikis/huggingface_transformers_UpdateMetadata.md)
