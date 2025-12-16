# File: `unsloth/import_fixes.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 426 |
| Classes | `HideLoggingMessage`, `MessageFactory` |
| Functions | `Version`, `fix_message_factory_issue`, `fix_xformers_performance_issue`, `fix_vllm_aimv2_issue`, `fix_vllm_guided_decoding_params`, `ignore_logger_messages`, `patch_ipykernel_hf_xet`, `patch_trackio`, `... +5 more` |
| Imports | importlib, logging, os, packaging, pathlib, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Runtime compatibility patches for dependencies

**Mechanism:** Monkey-patches to fix upstream library issues at import time

**Significance:** Prevents crashes from known upstream bugs
