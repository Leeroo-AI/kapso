# File: `unsloth/dataprep/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 15 |
| Imports | synthetic |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file that exposes the synthetic data generation functionality from the dataprep submodule.

**Mechanism:** Uses wildcard import (`from .synthetic import *`) to re-export all public members from the synthetic module, making SyntheticDataKit and related functions available at the dataprep package level.

**Significance:** Provides clean package-level access to Unsloth's synthetic data generation capabilities, allowing users to import directly from unsloth.dataprep rather than navigating to submodules. Part of Unsloth's data preparation toolkit for generating training data.
