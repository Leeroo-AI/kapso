# File: `utils/add_pipeline_model_mapping_to_test.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 308 |
| Functions | `get_mapping_for_task`, `get_model_for_pipeline_test`, `get_pipeline_model_mapping`, `get_pipeline_model_mapping_string`, `is_valid_test_class`, `find_test_class`, `find_block_ending`, `add_pipeline_model_mapping`, `... +1 more` |
| Imports | argparse, get_test_info, glob, inspect, os, re, tests, unittest |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically generates and updates the `pipeline_model_mapping` attribute in model test files to enable pipeline testing.

**Mechanism:** The script scans model test files, identifies test classes that inherit from `ModelTesterMixin`, and generates a mapping between pipeline tasks and model classes based on the centralized `pipeline_test_mapping`. It uses introspection to extract configuration classes from test classes, matches them against pipeline mappings, and programmatically inserts or updates the `pipeline_model_mapping` attribute in the appropriate location within the test class. The script also ensures test classes inherit from `PipelineTesterMixin` and handles different backend availability conditions (e.g., `is_torch_available()`, `is_timm_available()`).

**Significance:** This automation ensures that pipeline tests are consistently configured across all model implementations without manual maintenance. It integrates model tests with the pipeline testing framework, enabling systematic validation that models work correctly with their associated pipelines. This is critical for catching pipeline-related regressions and is run as part of CI/CD workflows.
