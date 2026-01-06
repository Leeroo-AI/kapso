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

**Purpose:** Aggregates model test results from pytest reports into structured JSON summaries for CI/CD monitoring and reporting.

**Mechanism:** Parses pytest short summary files from model test report directories, extracting passed/failed/skipped/error counts per test. Groups results by model and machine type (single-gpu/multi-gpu), adds metadata (GPU name, commit hash), and generates a collated JSON report. Optionally uploads reports to HuggingFace Hub for historical tracking and analysis.

**Significance:** Provides centralized test result tracking across the extensive model test suite. Enables monitoring test health over time, comparing results across different hardware configurations, and integrating test data with notification systems for CI/CD observability.
