# NotificationService Implementation

## Metadata

| Attribute | Value |
|-----------|-------|
| **Source File** | `/tmp/praxium_repo_d5p6fp4d/utils/notification_service.py` |
| **Repository** | huggingface_transformers |
| **Commit** | f9f6619c2cf7 |
| **Lines of Code** | 1622 |
| **Primary Domain** | CI/CD Reporting |
| **Secondary Domains** | Slack Integration, GitHub Actions, Test Result Analysis |
| **Last Updated** | 2025-12-18 |

## Overview

The `notification_service.py` module is a comprehensive CI/CD reporting system that aggregates test results from GitHub Actions workflows and posts detailed reports to Slack channels. It processes test artifacts, compares results with previous runs, tracks regressions, and provides rich formatted notifications with links to failed tests and detailed failure information.

The service supports multiple test categories (modeling, pipelines, examples, DeepSpeed, quantization, kernels), handles both single-GPU and multi-GPU test results, and can compare results across different CI runs to identify new failures and track trends over time.

## Description

### Core Functionality

The notification service provides several key capabilities:

1. **Test Result Aggregation**: Collects and processes test results from multiple GitHub Actions artifacts, parsing test output files to extract failure counts, success rates, and error traces

2. **Multi-Category Reporting**: Handles different test categories (PyTorch, Tokenizers, Pipelines, Trainer, ONNX, Auto, Quantization) with separate tracking and reporting

3. **Slack Integration**: Posts formatted messages to Slack channels with rich formatting, including buttons linking to GitHub Actions jobs, color-coded severity indicators, and threaded replies with detailed failure information

4. **Regression Detection**: Compares current test results with previous CI runs to identify new failures, changed failure counts, and unique failures across different hardware configurations

5. **Hub Integration**: Uploads detailed test results, failure reports, and comparison data to HuggingFace Hub datasets for historical tracking and analysis

6. **Multi-GPU Support**: Separately tracks and reports results for single-GPU and multi-GPU test runs

7. **Warning Tracking**: Collects and reports on warnings generated during CI runs, including special handling for `huggingface_hub` warnings

8. **CI Event Awareness**: Adapts reporting format and content based on CI trigger (push, schedule, pull request comment, workflow run)

### Architecture Components

**Core Classes**:
- `Message`: Main class encapsulating test results, formatting Slack messages, and handling posting logic
  - Aggregates model and additional test results
  - Generates formatted Slack blocks (header, failures, warnings, etc.)
  - Handles threaded replies with detailed failure information
  - Computes and displays differences from previous runs

**Key Functions**:
- `handle_test_results()`: Parses pytest output to extract test statistics
- `handle_stacktraces()`: Extracts error messages and line numbers from test failures
- `retrieve_artifact()`: Loads test result artifacts from downloaded directories
- `retrieve_available_artifacts()`: Discovers all available test artifacts in the workspace
- `prepare_reports()`: Formats failure reports with truncation for Slack limits

**Data Flow**:
1. GitHub Actions uploads test artifacts
2. Notification service downloads and processes artifacts
3. Results are aggregated by test category and device type
4. Comparisons with previous runs are computed
5. Results are uploaded to Hub dataset
6. Formatted messages are posted to Slack
7. Detailed failures are posted as threaded replies

### Key Data Structures

**Result Structure**:
```python
{
    "failed": {
        "PyTorch": {"single": 5, "multi": 2, "unclassified": 0},
        "Tokenizers": {"single": 1, "multi": 0, "unclassified": 0},
        # ... other categories
    },
    "errors": 3,
    "success": 1542,
    "skipped": 87,
    "time_spent": [123.45, 67.89, ...],
    "error": False,
    "failures": {
        "single": [
            {"line": "tests/models/bert/test_modeling.py::test_forward", "trace": "AssertionError: ..."},
            # ... more failures
        ],
        "multi": [...]
    },
    "job_link": {
        "single": "https://github.com/huggingface/transformers/actions/runs/.../job/...",
        "multi": "https://github.com/huggingface/transformers/actions/runs/.../job/..."
    },
    "captured_info": {
        "single": "https://github.com/huggingface/transformers/actions/runs/.../job/...#step:5:1"
    }
}
```

### Reporting Features

1. **Summary Statistics**: Total tests, failures, successes, execution time
2. **Category Breakdown**: Failures grouped by test category (modeling, pipelines, etc.)
3. **Device-specific Reporting**: Separate counts for single-GPU and multi-GPU runs
4. **Model-level Granularity**: Per-model failure counts for model testing job
5. **Change Detection**: Highlights changes in failure counts compared to previous runs
6. **New Failure Identification**: Lists tests that failed in current run but not in previous
7. **Unique Failure Analysis**: For multi-hardware setups, identifies failures unique to one platform
8. **Warning Aggregation**: Collects and reports selected warnings from CI runs

## Usage

### Running as GitHub Actions Step

The service is typically invoked as a step in GitHub Actions workflows:

```yaml
- name: Send results to Slack
  if: always()
  env:
    CI_SLACK_BOT_TOKEN: ${{ secrets.CI_SLACK_BOT_TOKEN }}
    ACCESS_REPO_INFO_TOKEN: ${{ secrets.ACCESS_REPO_INFO_TOKEN }}
    SLACK_REPORT_CHANNEL: ${{ secrets.SLACK_REPORT_CHANNEL }}
    CI_EVENT: ${{ github.event_name }}
    CI_TEST_JOB: run_models_gpu
    GITHUB_RUN_ID: ${{ github.run_id }}
  run: |
    python utils/notification_service.py "${{ toJson(matrix.folders) }}"
```

### Command Line Usage

Direct invocation (typically from CI):

```bash
# With job matrix
python utils/notification_service.py '["models_bert", "models_gpt2"]'

# Without job matrix (for non-matrix jobs)
python utils/notification_service.py ""
```

## Code Reference

### Main Class

```python
class Message:
    """
    Message class for formatting and posting CI test results to Slack.

    Attributes:
        title: Message title
        ci_title: CI event title with PR/commit info
        model_results: Dictionary of model test results
        additional_results: Dictionary of additional test results
        selected_warnings: List of selected warnings to report
        prev_ci_artifacts: Tuple of (workflow_run_id, artifacts) from previous run
        other_ci_artifacts: List of tuples for comparison with other runs
    """

    def __init__(
        self,
        title: str,
        ci_title: str,
        model_results: dict,
        additional_results: dict,
        selected_warnings: list | None = None,
        prev_ci_artifacts=None,
        other_ci_artifacts=None,
    ):
        """
        Initialize message with test results and configuration.

        Args:
            title: Message title for Slack
            ci_title: Formatted CI event title with links
            model_results: Dictionary mapping model names to test results
            additional_results: Dictionary of non-model test results
            selected_warnings: List of warnings to highlight
            prev_ci_artifacts: Previous CI run artifacts for comparison
            other_ci_artifacts: Additional CI runs for comparison
        """

    @property
    def time(self) -> str:
        """Calculate total time spent across all tests in format 'XhYmZs'."""

    @property
    def payload(self) -> str:
        """Generate complete Slack message payload as JSON string."""

    @staticmethod
    def error_out(title, ci_title="", runner_not_available=False, runner_failed=False, setup_failed=False):
        """
        Post error message to Slack when CI fails to run properly.

        Args:
            title: Error message title
            ci_title: CI event title
            runner_not_available: If True, runners are offline
            runner_failed: If True, runner environment failed
            setup_failed: If True, setup job failed
        """

    def post(self):
        """Post the main message to Slack and store thread timestamp."""

    def post_reply(self):
        """Post detailed failure information as threaded replies."""

    def get_reply_blocks(self, job_name, job_result, failures, device, text):
        """
        Generate Slack blocks for a reply containing failure details.

        Args:
            job_name: Name of the test job
            job_result: Test result dictionary
            failures: List of failure dictionaries with 'line' and 'trace' keys
            device: Device type ('single', 'multi', or None)
            text: Summary text for the reply

        Returns:
            List of Slack block dictionaries
        """

    def get_new_model_failure_blocks(self, prev_ci_artifacts, with_header=True, to_truncate=True):
        """
        Generate blocks showing failures that are new compared to previous run.

        Args:
            prev_ci_artifacts: Artifacts from previous CI run
            with_header: If True, include header block
            to_truncate: If True, truncate output to fit Slack limits

        Returns:
            List of Slack block dictionaries
        """
```

### Utility Functions

```python
def handle_test_results(test_results: str) -> tuple[int, int, int, int, str]:
    """
    Parse pytest summary output to extract test statistics.

    Args:
        test_results: Raw pytest output string

    Returns:
        Tuple of (failed, errors, success, skipped, time_spent)
    """

def handle_stacktraces(test_results: str) -> list[str]:
    """
    Extract error messages from pytest failure output.

    Args:
        test_results: Raw failure output

    Returns:
        List of formatted error messages with line numbers
    """

def retrieve_artifact(artifact_path: str, gpu: str | None) -> dict:
    """
    Load test artifact files from a directory.

    Args:
        artifact_path: Path to artifact directory
        gpu: GPU type ('single', 'multi', or None)

    Returns:
        Dictionary mapping filenames (without extension) to contents
    """

def retrieve_available_artifacts() -> dict[str, Artifact]:
    """
    Discover all test artifacts in the current directory.

    Returns:
        Dictionary mapping artifact names to Artifact objects
    """

def prepare_reports(title: str, header: str, reports: list, to_truncate: bool = True) -> str:
    """
    Format a list of reports into a Slack-compatible message with header.

    Args:
        title: Section title
        header: Table header
        reports: List of report lines
        to_truncate: If True, truncate to fit Slack limits

    Returns:
        Formatted report string
    """

def dicts_to_sum(objects: dict[str, dict] | list[dict]) -> collections.Counter:
    """
    Sum multiple dictionaries of counts.

    Args:
        objects: Dictionary of dictionaries or list of dictionaries

    Returns:
        Counter with summed values
    """
```

### Configuration Constants

```python
# Job name to test category mapping
job_to_test_map = {
    "run_models_gpu": "Models",
    "run_trainer_and_fsdp_gpu": "Trainer & FSDP",
    "run_pipelines_torch_gpu": "PyTorch pipelines",
    "run_examples_gpu": "Examples directory",
    "run_torch_cuda_extensions_gpu": "DeepSpeed",
    "run_quantization_torch_gpu": "Quantization",
    "run_kernels_gpu": "Kernels",
}

# Test category to result filename mapping
test_to_result_name = {
    "Models": "model",
    "Trainer & FSDP": "trainer_and_fsdp",
    "PyTorch pipelines": "torch_pipeline",
    "Examples directory": "example",
    "DeepSpeed": "deepspeed",
    "Quantization": "quantization",
    "Kernels": "kernels",
}

# Non-model test modules
NON_MODEL_TEST_MODULES = [
    "deepspeed",
    "extended",
    "fixtures",
    "generation",
    "onnx",
    "optimization",
    "pipelines",
    "sagemaker",
    "trainer",
    "utils",
    "fsdp",
    "quantization",
    "kernels",
]
```

## I/O Contract

### Input Specifications

| Input | Type | Description | Required | Default |
|-------|------|-------------|----------|---------|
| Job Matrix | `str` (JSON) | List of test folders/modules as CLI argument | Yes | N/A |
| CI_SLACK_BOT_TOKEN | env var | Slack bot authentication token | Yes | N/A |
| ACCESS_REPO_INFO_TOKEN | env var | GitHub token for API access | Yes | N/A |
| SLACK_REPORT_CHANNEL | env var | Slack channel ID for reports | Yes | N/A |
| CI_EVENT | env var | CI event type (push, schedule, etc.) | Yes | N/A |
| CI_TEST_JOB | env var | Name of the test job | Yes | N/A |
| GITHUB_RUN_ID | env var | GitHub Actions run ID | Yes | N/A |
| CI_TITLE | env var | Commit/PR title | No | `""` |
| CI_SHA | env var | Commit SHA | No | `None` |
| SETUP_STATUS | env var | Status of setup job | No | `"success"` |
| PREV_WORKFLOW_RUN_ID | env var | Previous workflow run ID | No | `""` |
| OTHER_WORKFLOW_RUN_ID | env var | Other workflow run ID for comparison | No | `""` |

**Test Artifact Requirements**:
- Artifacts must contain files: `stats`, `summary_short`, `failures_line`
- Optional files: `captured_info` (for debug output links)
- Artifact names must follow pattern: `[single-gpu|multi-gpu]_<test_name>_test_reports`

### Output Specifications

| Output | Type | Description |
|--------|------|-------------|
| Slack Message | Posted message | Main test results summary |
| Slack Replies | Posted messages | Threaded replies with failure details |
| Hub Upload | JSON files | Detailed results uploaded to dataset |
| Console Output | Text | Formatted JSON payloads for debugging |

**Slack Message Components**:
- Header with CI event type
- CI title with PR/commit links and author
- Overall statistics (failures/successes/time)
- Category-level failure breakdown
- Model-level failure breakdown
- New failures section (if previous run available)
- Test result diff section (for AMD CI)
- Buttons linking to GitHub Actions jobs

**Hub Uploaded Files**:
- `<category>_results.json`: Complete test results
- `<category>_results_extra.json`: Extended information with captured output links
- `model_failures_report.txt`: Formatted model failure table
- `module_failures_report.txt`: Formatted module failure table
- `changed_model_failures_report.txt`: Diff from previous run
- `new_failures.txt` / `new_failures.json`: New failures in current run
- `new_failures_against_<run_id>.txt/.json`: Unique failures vs. other runs
- `test_results_diff.json`: Comparison with previous run (AMD only)
- `job_links.json`: Mapping of jobs to GitHub Actions URLs

### Side Effects

1. **Network Operations**:
   - Posts messages to Slack via Web API
   - Uploads result files to HuggingFace Hub
   - Fetches GitHub Actions job information via API
   - Downloads previous CI artifacts from GitHub

2. **File System**:
   - Creates `ci_results_<job_name>/` directory
   - Writes multiple result files to disk
   - Reads artifact directories in current working directory

3. **External Services**:
   - Slack workspace receives notifications
   - HuggingFace Hub dataset receives new files
   - GitHub API is queried for job and workflow information

## Usage Examples

### Example 1: Basic CI Integration

In GitHub Actions workflow:

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        folder: [models_bert, models_gpt2]
    steps:
      - name: Run tests
        run: pytest tests/${{ matrix.folder }}

      - name: Upload results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test_reports_${{ matrix.folder }}
          path: test_results/

  report:
    needs: test
    runs-on: ubuntu-latest
    if: always()
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v3

      - name: Send to Slack
        env:
          CI_SLACK_BOT_TOKEN: ${{ secrets.CI_SLACK_BOT_TOKEN }}
          CI_TEST_JOB: run_models_gpu
        run: |
          python utils/notification_service.py "${{ toJson(matrix.folder) }}"
```

### Example 2: Handling Multiple Test Types

```python
import os
import json
from notification_service import Message

# Aggregate results from different test types
model_results = {
    "models_bert": {
        "failed": {"PyTorch": {"single": 2, "multi": 0, "unclassified": 0}},
        "success": 150,
        "error": False,
        # ... more fields
    }
}

additional_results = {
    "PyTorch pipelines": {
        "failed": {"single": 1, "multi": 0, "unclassified": 0},
        "success": 45,
        "error": False,
        # ... more fields
    }
}

# Create and post message
message = Message(
    title="[INFO] Results of Nightly CI",
    ci_title="<https://github.com/huggingface/transformers/pull/12345|Add new model>",
    model_results=model_results,
    additional_results=additional_results,
)

message.post()
message.post_reply()
```

### Example 3: Error Reporting

```python
from notification_service import Message

# Report when CI infrastructure fails
Message.error_out(
    title="[FAIL] Nightly CI",
    ci_title="Infrastructure Issue",
    runner_not_available=True
)
```

### Example 4: Comparing with Previous Run

```python
from get_previous_daily_ci import get_last_daily_ci_reports

# Get previous run artifacts
prev_workflow_run_id = "123456789"
prev_artifacts = get_last_daily_ci_reports(
    artifact_names=["ci_results_run_models_gpu"],
    output_dir="./previous_reports",
    token=os.environ["ACCESS_REPO_INFO_TOKEN"],
    workflow_run_id=prev_workflow_run_id,
)

# Create message with comparison
message = Message(
    title="[INFO] Results of Scheduled CI",
    ci_title="Daily CI Run",
    model_results=current_results,
    additional_results={},
    prev_ci_artifacts=(prev_workflow_run_id, prev_artifacts),
)

message.post()  # Will include new failures section
```

### Example 5: Processing Test Artifacts

```python
from notification_service import retrieve_artifact, handle_test_results, handle_stacktraces

# Load artifact
artifact_path = "single-gpu_models_bert_test_reports"
artifact = retrieve_artifact(artifact_path, gpu="single")

# Parse results
if "stats" in artifact:
    failed, errors, success, skipped, time_spent = handle_test_results(artifact["stats"])
    print(f"Results: {success} passed, {failed} failed in {time_spent}")

# Extract failures
if "failures_line" in artifact:
    stacktraces = handle_stacktraces(artifact["failures_line"])
    for trace in stacktraces:
        print(f"  - {trace}")
```

### Example 6: Custom Report Formatting

```python
from notification_service import prepare_reports

# Format failure reports
model_failures = [
    "     5 |      2 | bert",
    "     3 |      0 | gpt2",
    "     1 |      1 | vit",
]

report = prepare_reports(
    title="Model failures detected",
    header="Single |  Multi | Model\n",
    reports=model_failures,
    to_truncate=True
)

print(report)
# Output:
# Model failures detected:
# ```
# Single |  Multi | Model
#      5 |      2 | bert
#      3 |      0 | gpt2
#      1 |      1 | vit
# ```
```

### Example 7: Analyzing New Failures

```python
# Inside Message class usage
message = Message(...)

# Get new failure blocks (truncated for Slack)
new_failure_blocks = message.get_new_model_failure_blocks(
    prev_ci_artifacts=prev_artifacts,
    with_header=True,
    to_truncate=True
)

# Get complete new failures (for file upload)
complete_failures = message.get_new_model_failure_blocks(
    prev_ci_artifacts=prev_artifacts,
    with_header=False,
    to_truncate=False
)

# Save to file
with open("ci_results_run_models_gpu/new_failures.txt", "w") as f:
    f.write(complete_failures[-1]["text"]["text"])
```

## Related Pages

(To be populated as wiki structure develops)

---

**Note**: This service is a critical component of the HuggingFace Transformers CI/CD infrastructure, providing visibility into test health, tracking regressions, and enabling rapid response to test failures through detailed Slack notifications.
