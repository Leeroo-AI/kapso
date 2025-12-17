# File: `libs/langchain_v1/langchain/agents/structured_output.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 443 |
| Classes | `StructuredOutputError`, `MultipleStructuredOutputsError`, `StructuredOutputValidationError`, `_SchemaSpec`, `ToolStrategy`, `ProviderStrategy`, `OutputToolBinding`, `ProviderStrategyBinding`, `AutoStrategy` |
| Imports | __future__, dataclasses, langchain_core, pydantic, types, typing, typing_extensions, uuid |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines the type system and strategies for enforcing structured responses from language models, supporting multiple schema formats and output strategies.

**Mechanism:** Implements three main structured output strategies:

1. **ToolStrategy**: Uses artificial tool calls to enforce structure
   - Converts schemas to LangChain tools
   - Supports Union types and JSON schema oneOf
   - Includes error handling with retry mechanisms
   - Allows customization of tool message content

2. **ProviderStrategy**: Leverages native provider structured output
   - Uses OpenAI-style JSON schema format
   - Supports strict mode enforcement
   - Converts schema to model binding kwargs
   - Parses JSON content from AIMessage responses

3. **AutoStrategy**: Automatically selects best strategy
   - Wrapper for raw schemas
   - Detection logic lives in factory.py

Supporting components:
- **_SchemaSpec**: Normalizes schemas (Pydantic, dataclass, TypedDict, JSON schema) into a common format with name, description, and JSON schema representation
- **OutputToolBinding/ProviderStrategyBinding**: Handle parsing and validation for each strategy
- **Error classes**: Specialized exceptions for structured output failures

**Significance:** This module is essential for type-safe agent interactions, enabling:
- Schema-driven responses from language models
- Support for complex types (unions, nested structures)
- Automatic strategy selection based on model capabilities
- Robust error handling and validation
- Provider-agnostic structured output abstraction

The multi-strategy approach allows LangChain to work with various model providers while providing a consistent interface for structured outputs, making it crucial for building reliable agent applications.
