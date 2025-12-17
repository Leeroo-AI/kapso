# File: `libs/langchain/langchain_classic/output_parsers/ernie_functions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 45 |
| Imports | langchain_classic, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provide deprecated import redirects for Ernie (Baidu) function calling output parsers now located in langchain_community.

**Mechanism:** Uses dynamic import mechanism via create_importer to handle deprecation warnings and redirect imports of JsonKeyOutputFunctionsParser, JsonOutputFunctionsParser, OutputFunctionsParser, PydanticAttrOutputFunctionsParser, and PydanticOutputFunctionsParser to langchain_community.output_parsers.ernie_functions.

**Significance:** Maintains backward compatibility for code using Ernie-specific function calling parsers while guiding developers to the new location in langchain_community, supporting the modular architecture where third-party integrations are separated from core functionality.
