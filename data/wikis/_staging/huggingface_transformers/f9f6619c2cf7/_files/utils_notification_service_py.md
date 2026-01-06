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

**Purpose:** Aggregates CI test results and sends comprehensive failure reports to Slack channels with historical comparisons.

**Mechanism:** Collects test artifacts from GitHub Actions, parses pytest output for failures/successes/errors, categorizes failures by test type (PyTorch, Tokenizers, Pipelines, etc.) and device (single/multi-GPU), compares against previous runs to identify new failures, generates formatted Slack messages with summary statistics and detailed failure traces, uploads results to Hugging Face Hub dataset for permanent storage, and handles both scheduled daily CI and PR-triggered runs.

**Significance:** Central CI observability and communication hub that provides immediate visibility into test health across the team. Enables quick identification of regressions by comparing with previous runs, tracks failure trends over time through Hub dataset storage, and ensures critical test failures are promptly communicated to relevant Slack channels. Essential for maintaining code quality in a large, rapidly-evolving codebase with extensive test coverage.
