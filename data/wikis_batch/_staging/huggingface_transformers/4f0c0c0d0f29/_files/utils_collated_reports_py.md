# File: `utils/collated_reports.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 217 |
| Classes | `Args` |
| Functions | `simplify_gpu_name`, `parse_short_summary_line`, `validate_path`, `get_gpu_name`, `get_commit_hash`, `get_arguments`, `upload_collated_report` |
| Imports | argparse, dataclasses, json, pathlib, subprocess |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Aggregates test results from multiple model test runs into collated JSON reports for CI analysis.

**Mechanism:** Parses `summary_short.txt` files from test report directories, extracting test statuses (passed, failed, skipped, error) using regex patterns. Collects GPU information via torch or command-line arguments, retrieves git commit hash, and assembles per-model statistics. Generates JSON reports with total status counts and per-model breakdowns, optionally uploading to HuggingFace Hub datasets via `upload_collated_report()`.

**Significance:** Essential CI reporting infrastructure that consolidates test results across different GPU types and model directories, enabling centralized test failure analysis and historical tracking via Hub datasets.
