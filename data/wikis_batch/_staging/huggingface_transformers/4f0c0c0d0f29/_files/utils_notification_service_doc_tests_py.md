# File: `utils/notification_service_doc_tests.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 384 |
| Classes | `Message`, `Artifact` |
| Functions | `handle_test_results`, `extract_first_line_failure`, `retrieve_artifact`, `retrieve_available_artifacts` |
| Imports | get_ci_error_statistics, json, os, re, slack_sdk, time |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sends Slack notifications specifically for documentation doctest failures with Python/Markdown example categorization.

**Mechanism:** Parallel implementation to `notification_service.py` but specialized for doctests. `Message` class parses doctest artifacts from `doc_tests_gpu_test_reports_*`, extracts failures using regex patterns for doctest syntax (`_ [doctest]`), categorizes by "Python Examples" (src/) vs "MD Examples" (docs/), and posts summaries with failure counts. Uses `extract_first_line_failure()` to parse error context from doctest output.

**Significance:** Dedicated monitoring for documentation quality, ensuring code examples in docstrings and markdown remain executable and correct. Separate from main CI notifications due to different failure patterns and stakeholder concerns (docs team vs. model developers). Critical for maintaining user-facing documentation accuracy.
