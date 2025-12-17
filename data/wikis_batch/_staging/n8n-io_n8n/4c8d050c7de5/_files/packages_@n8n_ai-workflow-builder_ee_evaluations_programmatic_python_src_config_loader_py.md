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

**Purpose:** Configuration management for workflow comparison

**Mechanism:** Defines dataclasses for comprehensive comparison configuration including cost weights for node/edge operations (insertion, deletion, substitution), ignore rules for nodes and parameters, similarity groups for node types, parameter comparison rules (semantic, normalized, exact, numeric), exemption rules for optional nodes, and connection type handling. Supports loading from YAML, JSON, or built-in presets. Uses pattern matching with wildcards (**/*.patterns) for flexible parameter path filtering. The WorkflowComparisonConfig class provides methods like should_ignore_node(), should_ignore_parameter(), get_parameter_rule(), and get_exemption_penalty().

**Significance:** Central configuration system that makes the workflow comparison tool highly customizable. Enables different comparison strictness levels through presets, allows fine-grained control over what gets compared, and supports domain-specific comparison rules. Critical for balancing false positives vs false negatives in AI-generated workflow evaluation.
