# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/compare_workflows.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 334 |
| Functions | `parse_args`, `load_workflow`, `format_output_json`, `format_output_summary`, `main` |
| Imports | argparse, json, src, sys, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Command-line workflow comparison tool

**Mechanism:** Parses CLI arguments for workflow files and comparison options, loads workflows from JSON files, compares them using graph edit distance calculations, and outputs results in JSON or human-readable summary format.

**Significance:** Provides a user-friendly CLI interface for comparing n8n workflow JSON files. Core entry point for programmatic workflow evaluation used in AI workflow builder testing.
