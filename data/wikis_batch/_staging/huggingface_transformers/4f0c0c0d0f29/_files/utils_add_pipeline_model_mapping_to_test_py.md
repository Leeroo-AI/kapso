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

**Purpose:** Automatically generates and injects `pipeline_model_mapping` attributes into model test classes to specify which pipeline tasks each model supports.

**Mechanism:** Uses Python introspection and AST manipulation to find model test classes (inheriting from ModelTesterMixin), determines which pipeline tasks the model supports by cross-referencing with PIPELINE_TEST_MAPPING, generates a dictionary mapping task names to model classes, and rewrites the test file source code to include the mapping. Also ensures test classes inherit from PipelineTesterMixin when needed.

**Significance:** Testing infrastructure tool that maintains consistency between pipeline support declarations and test coverage, enabling automated CI checks to verify models work correctly with their supported pipelines.
