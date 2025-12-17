# Environment: Python Runtime

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|LangChain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|pyproject.toml|libs/langchain_v1/pyproject.toml]]
|-
! Domains
| [[domain::Infrastructure]], [[domain::Python]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

## Overview

Python 3.10+ runtime environment with core dependencies for LangChain agent and chat model functionality.

### Description

This environment provides the base Python runtime required for all LangChain functionality. It is built on Python 3.10 or higher (up to Python 3.13) and includes the core dependencies: `langchain-core` for base abstractions, `langgraph` for agent graph execution, and `pydantic` for data validation.

### Usage

Use this environment for **any** LangChain workflow including agent creation, chat model initialization, middleware composition, and text splitting. This is the mandatory base environment for all LangChain operations.

## System Requirements

{| class="wikitable"
! Category !! Requirement !! Notes
|-
| OS || Any (Linux, macOS, Windows) || Cross-platform support
|-
| Python || 3.10.0 to <4.0.0 || Python 3.10-3.13 supported
|-
| Disk || Minimal || ~50MB for core packages
|}

## Dependencies

### System Packages

No system-level packages required for core functionality.

### Python Packages

**Core Dependencies:**
* `langchain-core` >= 1.2.1, < 2.0.0
* `langgraph` >= 1.0.2, < 1.1.0
* `pydantic` >= 2.7.4, < 3.0.0

**Development/Testing:**
* `pytest` >= 8.0.0, < 9.0.0
* `pytest-asyncio` >= 0.23.2, < 2.0.0
* `mypy` >= 1.18.1, < 1.19.0
* `ruff` >= 0.14.2, < 0.15.0

## Credentials

No credentials required for the base environment. Provider-specific integrations require their own API keys (see `langchain-ai_langchain_Provider_Integrations`).

## Quick Install

```bash
# Install core LangChain package
pip install langchain>=1.2.0

# Or with uv (recommended for development)
uv pip install langchain>=1.2.0
```

## Code Evidence

Version constraints from `libs/langchain_v1/pyproject.toml:13-18`:
```python
requires-python = ">=3.10.0,<4.0.0"
dependencies = [
    "langchain-core>=1.2.1,<2.0.0",
    "langgraph>=1.0.2,<1.1.0",
    "pydantic>=2.7.4,<3.0.0",
]
```

Package installation check from `libs/langchain_v1/langchain/chat_models/base.py:533-537`:
```python
def _check_pkg(pkg: str, pkg_kebab: str | None = None) -> None:
    """Check if a package is installed and raise ImportError if not."""
    if not util.find_spec(pkg):
        pkg_kebab = pkg_kebab or pkg.replace("_", "-")
        raise ImportError(f"Unable to import {pkg}. Please install with: pip install {pkg_kebab}")
```

## Common Errors

{| class="wikitable"
|-
! Error Message !! Cause !! Solution
|-
|| `ImportError: Unable to import langchain_openai` || Provider package not installed || `pip install langchain-openai`
|-
|| `ModuleNotFoundError: No module named 'langgraph'` || Missing core dependency || `pip install langgraph>=1.0.2`
|-
|| `pydantic.errors.PydanticImportError` || Pydantic version mismatch || `pip install pydantic>=2.7.4`
|}

## Compatibility Notes

* **Python 3.10-3.13:** Fully supported; Python 3.14+ not yet tested
* **Pydantic V1 vs V2:** LangChain v1.2+ requires Pydantic V2 (>=2.7.4)
* **Type Checking:** Uses strict mypy mode; enable `plugins = ["pydantic.mypy"]`

## Related Pages

* [[required_by::Implementation:langchain-ai_langchain_init_chat_model]]
* [[required_by::Implementation:langchain-ai_langchain_create_agent]]
* [[required_by::Implementation:langchain-ai_langchain_text_splitter_types]]
* [[required_by::Implementation:langchain-ai_langchain_AgentMiddleware_class]]
