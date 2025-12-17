# File: `utils/check_config_attributes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 548 |
| Functions | `check_attribute_being_used`, `check_config_attributes_being_used`, `check_config_attributes` |
| Imports | inspect, os, re, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates that configuration class attributes defined in __init__ are actually used in corresponding modeling files.

**Mechanism:** Iterates through all configuration classes in CONFIG_MAPPING, extracts parameter names from __init__ signatures, searches modeling files for usage patterns (config.attribute, getattr(config, "attribute")), and reports unused attributes. Includes a large SPECIAL_CASES_TO_ALLOW dictionary for legitimate exceptions like internal-only parameters or attributes used in generation/training.

**Significance:** Code quality enforcement tool that prevents dead configuration parameters from accumulating, ensuring the codebase stays clean and configuration classes only expose parameters that are actually used by models.
