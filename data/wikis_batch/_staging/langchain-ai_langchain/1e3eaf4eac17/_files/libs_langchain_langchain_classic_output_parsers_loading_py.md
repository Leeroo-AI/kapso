# File: `libs/langchain/langchain_classic/output_parsers/loading.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 22 |
| Functions | `load_output_parser` |
| Imports | langchain_classic |

## Understanding

**Status:** ✅ Explored

**Purpose:** Dynamically load output parser instances from configuration dictionaries for deserialization and configuration-based parser construction.

**Mechanism:** Takes a config dict, checks for "output_parsers" key and "_type" field, instantiates the appropriate parser class (currently only supports "regex_parser"), and raises ValueError for unsupported parser types. Modifies the config dict in-place to replace the parser config with the instantiated parser object.

**Significance:** Enables configuration-driven parser instantiation, supporting use cases where parser configurations are stored as JSON/YAML and need to be rehydrated into parser objects, which is essential for saving and loading LangChain workflows.
