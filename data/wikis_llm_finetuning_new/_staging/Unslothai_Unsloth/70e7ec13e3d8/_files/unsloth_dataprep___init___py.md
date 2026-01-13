# File: `unsloth/dataprep/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 16 |
| Imports | raw_text, synthetic |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initializer that exposes the dataprep module's public API by importing and re-exporting components from the `synthetic` and `raw_text` submodules.

**Mechanism:** Uses wildcard imports (`from .synthetic import *` and `from .raw_text import *`) to bring all publicly exported names from the submodules into the dataprep namespace, making `SyntheticDataKit`, `RawTextDataLoader`, and `TextPreprocessor` accessible directly from `unsloth.dataprep`.

**Significance:** Provides a clean entry point for users to access data preparation utilities without needing to know the internal module structure. Essential for making the dataprep subpackage functional as a cohesive unit.
