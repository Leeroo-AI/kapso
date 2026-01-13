# File: `tests/utils/__init__.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 33 |
| Functions | `timer`, `header_footer_context` |
| Imports | contextlib, time |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides utility context managers for test instrumentation, including timing measurements and formatted output sections.

**Mechanism:** The module exports two context managers: (1) timer(name) measures and prints the elapsed time for a code block using time.time() for start/end timestamps, outputting in the format "{name} took {seconds:.2f} seconds", (2) header_footer_context(title, char) prints a formatted header line before the block (50 chars + title + 50 chars of the specified character) and a matching footer line after, useful for visually separating test output sections. Both use the @contextmanager decorator from contextlib for clean generator-based implementation with yield.

**Significance:** These utilities enhance test output readability and help identify performance regressions during development. The timer context manager is particularly useful for benchmarking model loading, training steps, and inference operations. The header_footer_context provides visual structure to long test outputs, making it easier to locate specific test phases in console logs or CI output.
