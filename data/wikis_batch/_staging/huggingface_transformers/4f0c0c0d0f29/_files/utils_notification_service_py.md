# File: `utils/notification_service.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1622 |
| Classes | `Message`, `Artifact` |
| Functions | `handle_test_results`, `handle_stacktraces`, `dicts_to_sum`, `retrieve_artifact`, `retrieve_available_artifacts`, `prepare_reports`, `pop_default` |
| Imports | ast, collections, compare_test_runs, functools, get_ci_error_statistics, get_previous_daily_ci, huggingface_hub, json, operator, os, ... +6 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates and posts comprehensive CI test failure reports to Slack with detailed breakdowns and historical comparisons.

**Mechanism:** `Message` class aggregates test results from artifacts (parsing stats, stacktraces, failures by category), constructs Slack Block Kit payloads with header/summary/failures/warnings sections, compares against previous runs to identify new failures, uploads detailed reports to HuggingFace Hub datasets, and posts main messages with threaded replies containing full error traces. Supports single-gpu/multi-gpu breakdown, job-to-test mapping (`job_to_test_map`), and diff generation for AMD CI workflows.

**Significance:** Mission-critical CI observability tool providing maintainers with instant, actionable failure notifications on Slack. Enables rapid response to test regressions, tracks failure trends over time, and surfaces new issues through historical comparison. Central to maintaining the library's test health and developer productivity.
