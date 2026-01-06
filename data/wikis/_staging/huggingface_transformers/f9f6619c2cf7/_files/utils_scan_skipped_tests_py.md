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

**Purpose:** Scans all model test files to identify which common tests are skipped across models, generating comprehensive reports showing skip patterns, reasons, and coverage statistics for test infrastructure analysis.

**Mechanism:** The script extracts common test function names from test_modeling_common.py and generation test_utils.py, recursively scans all model-specific test files in tests/models/ using regex to detect test functions with skip decorators, parses decorator blocks to extract skip reasons, aggregates data to show which models run or skip each test, calculates skip proportions across all models, normalizes test names by removing parameterization details, and outputs detailed JSON reports with per-test coverage analysis or single-test deep dives depending on command-line arguments.

**Significance:** This tool is essential for maintaining test infrastructure health and identifying gaps in model test coverage. It helps maintainers understand which common tests have broad vs. narrow applicability, find models with excessive skips that may need attention, validate that test skips have documented reasons, and guide decisions about refactoring or deprecating tests based on actual usage patterns across the 100+ model implementations.
