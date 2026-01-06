# File: `libs/langchain/langchain_classic/output_parsers/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 82 |
| Imports | langchain_classic, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Public API entry point for output parsers that convert LLM text outputs into structured formats.

**Mechanism:** Imports and re-exports parsers from both langchain_core (core parsers like PydanticOutputParser, XMLOutputParser) and langchain_classic (specialized parsers like BooleanOutputParser, RetryOutputParser). Uses deprecated import mechanism via create_importer to handle backward compatibility for moved components like GuardrailsOutputParser (now in langchain_community).

**Significance:** Primary entry point for the output_parsers module, providing a unified API surface. Critical for backward compatibility as it manages deprecated exports and centralizes access to all parser implementations across different packages.
