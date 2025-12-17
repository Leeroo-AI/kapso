# File: `packages/@n8n/task-runner-python/tests/integration/test_rpc.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 116 |
| Functions | `test_print_basic_types`, `test_print_complex_types`, `test_print_edge_cases` |
| Imports | pytest, src, tests, textwrap |

## Understanding

**Status:** ✅ Explored

**Purpose:** RPC communication testing for print/console output

**Mechanism:** Tests that print() statements in user Python code are properly captured and transmitted via RPC messages (logNodeOutput method) to the broker. Validates various data types (strings, numbers, booleans, None, multiple args), complex structures (dicts, lists, nested objects), and edge cases (Unicode/emoji, escape sequences, empty values, very long strings 1000 chars).

**Significance:** Ensures debugging and logging functionality works correctly - users rely on print() output to troubleshoot workflows. Validates the RPC message protocol for transmitting console output from isolated Python execution environments back to n8n for display.
