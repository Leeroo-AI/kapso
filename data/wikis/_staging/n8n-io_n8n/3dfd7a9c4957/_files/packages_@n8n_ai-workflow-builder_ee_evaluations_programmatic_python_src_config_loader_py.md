# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/config_loader.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 389 |
| Classes | `NodeIgnoreRule`, `ParameterComparisonRule`, `ExemptionRule`, `WorkflowComparisonConfig` |
| Functions | `load_config` |
| Imports | dataclasses, json, pathlib, re, typing, yaml |

## Understanding

**Status:** ✅ Explored

**Purpose:** Workflow comparison configuration management

**Mechanism:** Defines dataclass configuration structures for workflow comparison rules including NodeIgnoreRule (for ignoring specific nodes), ParameterComparisonRule (for comparing node parameters), ExemptionRule (for penalizing workflow differences), and WorkflowComparisonConfig (comprehensive configuration). Supports loading from JSON/YAML files or defaults.

**Significance:** Provides flexible, extensible configuration for workflow similarity comparisons. Enables customization of what differences matter and how they should be scored.
