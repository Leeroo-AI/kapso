# huggingface_transformers_CIErrorStatistics

## Metadata

| Attribute | Value |
|-----------|-------|
| Source | `/utils/get_ci_error_statistics.py` |
| Repository | huggingface/transformers |
| Commit | f9f6619c2cf7 |
| Domains | continuous-integration, error-analysis, testing, automation |
| Last Updated | 2025-12-18 |

## Overview

The `get_ci_error_statistics.py` script analyzes GitHub Actions workflow runs to extract, aggregate, and report on test failures and error patterns. It downloads test artifacts, parses failure information, and generates statistical summaries organized by error type and affected models.

## Description

This implementation provides automated analysis of Continuous Integration (CI) failures in GitHub Actions workflows. The script connects to the GitHub API to download test artifacts from a specific workflow run, extracts error information from standardized test report files, and produces multiple views of the failure data.

**Core Capabilities**:

1. **Artifact Management**: Downloads all test artifacts from a GitHub Actions workflow run via API
2. **Error Extraction**: Parses standardized test report files to extract error messages, locations, and failed test names
3. **Statistical Analysis**: Aggregates errors across multiple test jobs using Counter-based statistics
4. **Multi-View Reporting**: Generates three complementary views of failure data:
   - Raw error list with complete details
   - Aggregation by error type with test counts
   - Aggregation by model with error breakdowns
5. **GitHub Integration**: Creates formatted GitHub markdown tables for issue/PR comments

**Data Flow**:
1. Query GitHub API for workflow run jobs and artifacts
2. Download all artifacts as ZIP files
3. Extract error information from each artifact's report files
4. Cross-reference with job links for debugging context
5. Aggregate and count errors across dimensions
6. Generate JSON reports and markdown tables

The script handles pagination for large workflow runs, implements polite rate limiting, and provides structured error handling for API failures.

## Usage

### Command Line Interface

```bash
python utils/get_ci_error_statistics.py \
    --workflow_run_id <run_id> \
    --output_dir <output_directory> \
    [--token <github_token>]
```

### Arguments

- `--workflow_run_id` (required): GitHub Actions workflow run ID (numeric)
- `--output_dir` (required): Directory to store downloaded artifacts and generated reports
- `--token` (optional): GitHub personal access token with `actions:read` permission for private repos or higher rate limits

### Example

```bash
# Analyze a public workflow run
python utils/get_ci_error_statistics.py \
    --workflow_run_id 7234567890 \
    --output_dir ./ci_analysis

# With authentication for private repos or higher rate limits
python utils/get_ci_error_statistics.py \
    --workflow_run_id 7234567890 \
    --output_dir ./ci_analysis \
    --token ghp_xxxxxxxxxxxx
```

## Code Reference

### Main Functions

#### `get_jobs(workflow_run_id: str, token: str = None) -> list`

Retrieves all jobs from a GitHub Actions workflow run.

**Parameters**:
- `workflow_run_id`: Numeric workflow run identifier
- `token`: Optional GitHub authentication token

**Returns**: List of job dictionaries with complete job information

**Implementation Details**:
- Uses GitHub API with pagination support (100 jobs per page)
- Calculates total pages from initial response's total_count
- Iterates through remaining pages to collect all jobs
- Returns empty list on error with traceback output

---

#### `get_job_links(workflow_run_id: str, token: str = None) -> dict`

Extracts job names and their web URLs from a workflow run.

**Parameters**:
- `workflow_run_id`: Numeric workflow run identifier
- `token`: Optional GitHub authentication token

**Returns**: Dictionary mapping job names to GitHub job URLs

**Implementation Details**:
- Similar pagination logic to get_jobs()
- Extracts name and html_url fields from each job
- Returns empty dict on error

---

#### `get_artifacts_links(workflow_run_id: str, token: str = None) -> dict`

Retrieves all artifact download URLs from a workflow run.

**Parameters**:
- `workflow_run_id`: Numeric workflow run identifier
- `token`: Optional GitHub authentication token

**Returns**: Dictionary mapping artifact names to download URLs

**Implementation Details**:
- Queries artifacts endpoint with pagination
- Maps artifact names to archive_download_url
- Handles up to 100 artifacts per page

---

#### `download_artifact(artifact_name: str, artifact_url: str, output_dir: str, token: str) -> None`

Downloads a GitHub Actions artifact ZIP file.

**Parameters**:
- `artifact_name`: Name for the downloaded file
- `artifact_url`: API endpoint URL (not direct download)
- `output_dir`: Destination directory
- `token`: GitHub authentication token

**Implementation Details**:
- First request gets redirect URL from Location header
- Second request downloads actual artifact content
- Saves as {artifact_name}.zip in output directory
- API URL format: https://api.github.com/repos/huggingface/transformers/actions/artifacts/{ID}/zip

---

#### `get_errors_from_single_artifact(artifact_zip_path: str, job_links: dict = None) -> list`

Parses error information from a downloaded artifact ZIP file.

**Parameters**:
- `artifact_zip_path`: Path to artifact ZIP file
- `job_links`: Optional dictionary mapping job names to URLs

**Returns**: List of tuples: `[(error_line, error_message, failed_test, job_url), ...]`

**Implementation Details**:
- Reads three specific files from ZIP without extraction:
  - `failures_line.txt`: Error location and message (format: "location: message")
  - `summary_short.txt`: Failed test names (lines starting with "FAILED ")
  - `job_name.txt`: Job identifier for linking
- Validates that number of errors matches number of failed tests
- Enriches each error with job URL when available
- Handles workflow_call events by parsing composite job names

**Expected File Formats**:
```
failures_line.txt:
tests/models/bert/test_modeling.py::TestClass::test_method: AssertionError: Expected X, got Y

summary_short.txt:
FAILED tests/models/bert/test_modeling.py::TestClass::test_method

job_name.txt:
Model tests (models/bert, single-gpu)
```

---

#### `get_all_errors(artifact_dir: str, job_links: dict = None) -> list`

Aggregates errors from all artifact ZIP files in a directory.

**Parameters**:
- `artifact_dir`: Directory containing artifact ZIP files
- `job_links`: Optional job name to URL mapping

**Returns**: Combined list of all error tuples from all artifacts

**Implementation Details**:
- Finds all .zip files in directory
- Calls get_errors_from_single_artifact for each
- Concatenates results

---

#### `reduce_by_error(logs: list, error_filter: set = None) -> dict`

Aggregates errors by unique error message.

**Parameters**:
- `logs`: List of error tuples from get_all_errors()
- `error_filter`: Optional set of error messages to exclude

**Returns**: Dictionary structured as:
```python
{
    "error_message": {
        "count": int,
        "failed_tests": [(test_name, error_location), ...]
    }
}
```

**Implementation Details**:
- Uses Counter to tally occurrences by error message (tuple element [1])
- For each unique error, collects all (test, location) pairs where it occurred
- Sorts by count descending
- Applies optional filtering to exclude known/ignored errors

---

#### `get_model(test: str) -> str | None`

Extracts model name from a test path.

**Parameters**:
- `test`: Full test identifier (e.g., "tests/models/bert/test_modeling.py::TestClass::test_method")

**Returns**: Model name if path matches pattern, else None

**Implementation Details**:
- Splits on "::" to get file path
- Checks if path starts with "tests/models/"
- Returns second path component (model name)
- Returns None for non-model tests

---

#### `reduce_by_model(logs: list, error_filter: set = None) -> dict`

Aggregates errors by affected model.

**Parameters**:
- `logs`: List of error tuples
- `error_filter`: Optional error messages to exclude

**Returns**: Dictionary structured as:
```python
{
    "model_name": {
        "count": total_error_count,
        "errors": {
            "error_message": count,
            ...
        }
    }
}
```

**Implementation Details**:
- Extracts model name from each test using get_model()
- Filters to only model-specific tests
- For each model, counts errors by type using Counter
- Applies optional error filtering
- Sorts models by total error count descending

---

#### `make_github_table(reduced_by_error: dict) -> str`

Generates a GitHub-flavored markdown table of errors.

**Parameters**:
- `reduced_by_error`: Output from reduce_by_error()

**Returns**: Markdown table string

**Table Format**:
```
| no. | error | status |
|-:|:-|:-|
| 42 | AssertionError: Expected X, got Y |  |
| 15 | RuntimeError: CUDA out of memory |  |
```

**Implementation Details**:
- Truncates error messages to 100 characters
- Status column left empty for manual triage
- Right-aligns count, left-aligns error text

---

#### `make_github_table_per_model(reduced_by_model: dict) -> str`

Generates a GitHub-flavored markdown table of model errors.

**Parameters**:
- `reduced_by_model`: Output from reduce_by_model()

**Returns**: Markdown table string

**Table Format**:
```
| model | no. of errors | major error | count |
|-:|-:|-:|-:|
| bert | 42 | AssertionError: Expected X, got Y | 30 |
| gpt2 | 15 | RuntimeError: CUDA out of memory | 10 |
```

**Implementation Details**:
- Shows total error count per model
- Displays most frequent error and its count
- Truncates major error to 60 characters
- All columns right-aligned

---

## I/O Contract

### Inputs

| Input Type | Description | Format | Example |
|------------|-------------|--------|---------|
| Command Line Argument | Workflow run ID | Numeric string | `--workflow_run_id 7234567890` |
| Command Line Argument | Output directory | File path | `--output_dir ./analysis` |
| Command Line Argument | GitHub token (optional) | Token string | `--token ghp_xxxx` |
| GitHub API | Jobs data | JSON API response | 100 jobs per page |
| GitHub API | Artifacts metadata | JSON API response | Artifact names and URLs |
| GitHub API | Artifact ZIP files | Binary ZIP content | Downloaded via redirect |
| Artifact Files | Error locations and messages | failures_line.txt | `path/to/test.py::test: Error message` |
| Artifact Files | Failed test names | summary_short.txt | `FAILED path/to/test.py::test` |
| Artifact Files | Job identifier | job_name.txt | Single line with job name |

### Outputs

| Output Type | Description | Format | Location |
|-------------|-------------|--------|----------|
| JSON File | Job name to URL mapping | JSON object | {output_dir}/job_links.json |
| JSON File | Artifact name to URL mapping | JSON object | {output_dir}/artifacts.json |
| ZIP Files | Downloaded artifacts | Binary ZIP files | {output_dir}/*.zip |
| JSON File | All extracted errors | JSON array of tuples | {output_dir}/errors.json |
| Text File | Errors grouped by type | Markdown table | {output_dir}/reduced_by_error.txt |
| Text File | Errors grouped by model | Markdown table | {output_dir}/reduced_by_model.txt |
| Console Output | Top 30 most common errors | Text listing | stdout |

### Side Effects

- Creates output directory if it doesn't exist
- Downloads potentially large ZIP files to disk
- Makes multiple API requests to GitHub (rate limit considerations)
- 1-second sleep between artifact downloads (rate limiting)

## Usage Examples

### Basic Analysis

```bash
# Analyze a workflow run and save results
python utils/get_ci_error_statistics.py \
    --workflow_run_id 7234567890 \
    --output_dir ./analysis_results

# Output files created:
# ./analysis_results/job_links.json
# ./analysis_results/artifacts.json
# ./analysis_results/*.zip (all test artifacts)
# ./analysis_results/errors.json
# ./analysis_results/reduced_by_error.txt
# ./analysis_results/reduced_by_model.txt
```

### With Authentication

```bash
# Use token for private repos or higher rate limits
export GITHUB_TOKEN=ghp_your_token_here

python utils/get_ci_error_statistics.py \
    --workflow_run_id 7234567890 \
    --output_dir ./analysis_results \
    --token $GITHUB_TOKEN
```

### Programmatic Usage

```python
from get_ci_error_statistics import (
    get_jobs, get_artifacts_links, download_artifact,
    get_all_errors, reduce_by_error, reduce_by_model,
    make_github_table, make_github_table_per_model
)

# Get workflow information
workflow_id = "7234567890"
token = "ghp_xxxx"

jobs = get_jobs(workflow_id, token)
print(f"Found {len(jobs)} jobs")

# Download and analyze artifacts
artifacts = get_artifacts_links(workflow_id, token)
output_dir = "./analysis"

for name, url in artifacts.items():
    download_artifact(name, url, output_dir, token)

# Extract and analyze errors
errors = get_all_errors(output_dir)
print(f"Total errors found: {len(errors)}")

# Aggregate by error type
by_error = reduce_by_error(errors)
for error, data in list(by_error.items())[:5]:
    print(f"{data['count']}x: {error[:80]}")

# Aggregate by model
by_model = reduce_by_model(errors)
for model, data in list(by_model.items())[:5]:
    print(f"{model}: {data['count']} total errors")
    top_error = list(data['errors'].items())[0]
    print(f"  Most common: {top_error[1]}x {top_error[0][:60]}")
```

### Filtering Errors

```python
from get_ci_error_statistics import reduce_by_error, reduce_by_model

# Define known/expected errors to ignore
ignored_errors = {
    "ImportError: cannot import name 'old_module'",
    "DeprecationWarning: This feature is deprecated"
}

# Analyze with filtering
errors = get_all_errors("./artifacts")
by_error = reduce_by_error(errors, error_filter=ignored_errors)
by_model = reduce_by_model(errors, error_filter=ignored_errors)

# Results exclude the filtered errors
print(f"Actionable errors: {sum(d['count'] for d in by_error.values())}")
```

### Generating Reports

```python
from get_ci_error_statistics import (
    reduce_by_error, reduce_by_model,
    make_github_table, make_github_table_per_model
)
import json

errors = get_all_errors("./artifacts")

# Create GitHub-ready tables
by_error = reduce_by_error(errors)
by_model = reduce_by_model(errors)

error_table = make_github_table(by_error)
model_table = make_github_table_per_model(by_model)

# Save for GitHub issue/PR comment
with open("error_report.md", "w") as f:
    f.write("## CI Error Analysis\n\n")
    f.write("### Errors by Type\n\n")
    f.write(error_table)
    f.write("\n\n### Errors by Model\n\n")
    f.write(model_table)
```

### Extracting Model-Specific Errors

```python
from get_ci_error_statistics import get_all_errors, get_model

errors = get_all_errors("./artifacts")

# Filter for specific model
target_model = "bert"
bert_errors = [
    (location, error, test)
    for location, error, test, job_url in errors
    if get_model(test) == target_model
]

print(f"BERT-specific errors: {len(bert_errors)}")
for location, error, test in bert_errors[:5]:
    print(f"  {test}")
    print(f"    {error}")
```

### Working with Job Links

```python
from get_ci_error_statistics import get_job_links, get_errors_from_single_artifact

workflow_id = "7234567890"
job_links = get_job_links(workflow_id, token="ghp_xxxx")

# Handle workflow_call composite job names
processed_links = {}
for job_name, url in job_links.items():
    if " / " in job_name:
        # Extract actual job name from composite
        actual_name = job_name.split(" / ", 1)[1]
        processed_links[actual_name] = url
    else:
        processed_links[job_name] = url

# Use when extracting errors
errors = get_errors_from_single_artifact(
    "artifact.zip",
    job_links=processed_links
)

# Each error now includes job URL for easy debugging
for location, error, test, job_url in errors:
    print(f"Error in: {job_url}")
```

## Related Pages

- [Related implementation pages will be listed here]
