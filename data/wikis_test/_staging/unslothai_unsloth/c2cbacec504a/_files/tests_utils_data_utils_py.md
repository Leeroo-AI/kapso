# File: `tests/utils/data_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 153 |
| Functions | `create_instruction_dataset`, `create_dataset`, `describe_param`, `format_summary`, `get_peft_weights`, `describe_peft_weights`, `check_responses` |
| Imports | datasets, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides dataset creation and manipulation utilities for tests, including instruction dataset generation, model weight inspection, and response validation functions.

**Mechanism:** Creates synthetic instruction-following datasets with proper formatting, generates training datasets with varied examples, extracts and describes PEFT adapter weights from models, formats weight statistics, and validates model output quality.

**Significance:** Essential test infrastructure for generating consistent training data across tests, enabling weight analysis for debugging, and providing utilities for verifying model behavior without requiring large external datasets.
