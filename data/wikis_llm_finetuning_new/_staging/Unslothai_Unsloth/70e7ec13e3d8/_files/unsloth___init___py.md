# File: `unsloth/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 295 |
| Imports | chat_templates, dataprep, functools, import_fixes, importlib, inspect, models, numpy, os, packaging, ... +9 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package entry point that exports all public APIs, manages version compatibility, and orchestrates lazy loading of core components (FastLanguageModel, UnslothTrainer, chat templates, etc.).

**Mechanism:** Uses `__getattr__` for lazy module loading to defer heavy imports until needed. Exports main classes like `FastLanguageModel`, `FastVisionModel`, `UnslothTrainer`, and `UnslothTrainingArguments`. Handles version checking against PyPI, applies import fixes for library compatibility, and manages numpy version constraints for TRL compatibility.

**Significance:** Core component - the main entry point for all Unsloth functionality. Users import directly from this module to access fine-tuning capabilities.
