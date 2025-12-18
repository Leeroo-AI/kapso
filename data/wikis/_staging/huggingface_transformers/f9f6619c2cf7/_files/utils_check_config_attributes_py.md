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

**Purpose:** Validates that all configuration class attributes defined in `__init__` are actually used in their corresponding modeling files to prevent dead code.

**Mechanism:** The script uses introspection to extract all parameters from each configuration class's `__init__` method, then searches through all modeling files in the same directory for usage patterns like `config.attribute`, `getattr(config, "attribute")`, or `config.get_text_config().attribute`. It maintains an extensive allowlist (`SPECIAL_CASES_TO_ALLOW`) for attributes that are legitimately unused in modeling code but serve other purposes (training parameters, generation config, internal calculations, etc.). The script also handles attribute mapping where configuration attributes may have different names in usage.

**Significance:** This check ensures code quality by preventing configuration bloat where attributes are defined but never used, making the codebase easier to maintain and understand. It's part of the CI quality checks (`make repo-consistency`) and helps catch cases where attributes are removed from modeling code but left in configuration classes, or where new attributes are added to configs without being used.
