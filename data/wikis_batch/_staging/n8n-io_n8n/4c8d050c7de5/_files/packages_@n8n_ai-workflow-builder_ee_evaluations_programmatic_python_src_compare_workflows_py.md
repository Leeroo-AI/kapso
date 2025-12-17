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

**Purpose:** CLI tool for comparing workflows using graph edit distance

**Mechanism:** Implements a command-line interface that loads two n8n workflow JSON files, builds graphs with configurable filtering, calculates similarity using graph edit distance, and outputs results in JSON or human-readable summary format. Supports three presets (strict/standard/lenient) and custom config files. Includes detailed parameter diff formatting and priority-based edit operation ranking. The tool calculates similarity scores, edit costs, and provides actionable recommendations for workflow differences.

**Significance:** Core evaluation tool for the AI workflow builder that measures how closely AI-generated workflows match ground truth workflows. Essential for quality assessment and automated testing of the AI workflow generation system. Provides both machine-readable JSON output for automation and human-readable summaries for manual review.
