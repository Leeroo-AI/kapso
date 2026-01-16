# File: `WebAgent/WebDancer/demos/tools/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 7 |
| Imports | private |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization that exports the `Visit` and `Search` tools for use by agents.

**Mechanism:** Imports `Visit` and `Search` classes from the `private` submodule and exports them via `__all__`. This provides a clean public interface to the tool implementations.

**Significance:** Entry point for tool access in WebDancer. Allows other modules to import tools simply as `from demos.tools import Visit, Search` rather than accessing the private submodule directly.
