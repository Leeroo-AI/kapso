# File: `libs/langchain_v1/langchain/agents/structured_output.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 443 |
| Classes | `StructuredOutputError`, `MultipleStructuredOutputsError`, `StructuredOutputValidationError`, `_SchemaSpec`, `ToolStrategy`, `ProviderStrategy`, `OutputToolBinding`, `ProviderStrategyBinding`, `AutoStrategy` |
| Imports | __future__, dataclasses, langchain_core, pydantic, types, typing, typing_extensions, uuid |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines types, strategies, and error handling for getting structured output from LLM agents.

**Mechanism:** Provides three strategy classes for structured output: ToolStrategy (uses tool calling with artificial tools), ProviderStrategy (uses native model JSON schema support), and AutoStrategy (auto-detects best approach). Includes _SchemaSpec for schema normalization across Pydantic models, dataclasses, TypedDict, and JSON schemas. Implements parsing and validation with custom error types (MultipleStructuredOutputsError, StructuredOutputValidationError) for better error handling. Uses Pydantic TypeAdapter for flexible schema validation.

**Significance:** This is a critical abstraction layer that allows agents to produce structured output in a model-agnostic way. It enables the agent system to work with different LLM providers (some supporting native JSON mode, others requiring tool-based approaches) while presenting a unified interface. The strategy pattern allows runtime selection of the best approach based on model capabilities.
