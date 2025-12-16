# File: `unsloth/dataprep/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 15 |
| Imports | synthetic |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization that exposes synthetic data generation capabilities through wildcard import.

**Mechanism:** Single wildcard import (`from .synthetic import *`) makes all public exports from synthetic module available at package level. Includes Apache License 2.0 header indicating more permissive licensing than the LGPL v3 used in other Unsloth components.

**Significance:** Simplifies access to SyntheticDataKit by allowing users to import directly from `unsloth.dataprep` rather than needing to specify the synthetic submodule. The wildcard import pattern delegates public API control to the synthetic module's `__all__` definition. The Apache 2.0 license may indicate this component was developed separately or intended for broader reuse.
