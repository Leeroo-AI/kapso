# File: `tests/utils/__init__.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 33 |
| Functions | `timer`, `header_footer_context` |
| Imports | contextlib, time |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test utility module providing context managers for timing code execution and formatting test output with headers/footers.

**Mechanism:** Implements two context managers using @contextmanager decorator: (1) timer(name) captures start time on entry, yields control to wrapped code block, then prints elapsed time in seconds on exit with 2 decimal precision, useful for benchmarking test sections, and (2) header_footer_context(title, char="-") prints decorative header line before wrapped code (50 chars + title + 50 chars using specified character), yields control, then prints matching footer line after, creating visually distinct test sections in output.

**Significance:** Reusable utility for test organization and performance monitoring. The timer context manager enables consistent timing measurements across test suites without boilerplate timing code. The header_footer_context improves test output readability by creating clear visual boundaries between test sections, making logs easier to parse when debugging complex multi-stage tests like vision model benchmarks.
