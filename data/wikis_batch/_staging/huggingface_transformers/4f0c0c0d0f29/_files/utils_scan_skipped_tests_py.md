# File: `utils/scan_skipped_tests.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 199 |
| Functions | `get_common_tests`, `get_models_and_test_files`, `extract_test_info`, `build_model_overrides`, `save_json`, `summarize_single_test`, `summarize_all_tests`, `main` |
| Imports | argparse, json, pathlib, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Analyzes test coverage by scanning which common tests are skipped across model implementations.

**Mechanism:** Extracts common test function names from test_modeling_common.py and generation test_utils.py, parses all model test files to identify skip decorators and their reasons, aggregates data to show which models run vs skip each test, and outputs JSON reports with skip proportions and rationales. Supports single-test mode for focused analysis or bulk mode for complete coverage scan.

**Significance:** Essential testing infrastructure tool that provides visibility into test coverage gaps across 100+ models. Helps maintainers track which common tests (forward pass, generation, etc.) are systematically skipped, identify patterns in skip reasons, and prioritize test enablement efforts to improve overall codebase test quality.
