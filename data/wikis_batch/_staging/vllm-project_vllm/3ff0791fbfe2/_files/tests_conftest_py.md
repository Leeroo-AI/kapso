# File: `tests/conftest.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1517 |
| Classes | `ImageAssetPrompts`, `ImageTestAssets`, `VideoAssetPrompts`, `VideoTestAssets`, `AudioAssetPrompts`, `AudioTestAssets`, `DecoderPromptType`, `HfRunner`, `VllmRunner`, `LogHolder`, `AssetHandler`, `LocalAssetServer` |
| Functions | `init_test_http_connection`, `dist_init`, `should_do_global_cleanup_after_test`, `cleanup_fixture`, `workspace_init`, `dynamo_reset`, `example_prompts`, `example_system_message`, `... +24 more` |
| Imports | PIL, collections, contextlib, copy, enum, http, huggingface_hub, json, math, mimetypes, ... +13 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Pytest configuration and shared test fixtures

**Mechanism:** Defines pytest fixtures and test utilities including HfRunner (Hugging Face model runner), VllmRunner (vLLM model runner), multimodal asset management (images, videos, audio), logging capture fixtures, distributed environment setup, and helper classes for model testing. Provides fixtures for example prompts, asset servers, GPU cleanup, and custom options.

**Significance:** Central testing infrastructure that enables consistent model comparison between vLLM and Hugging Face implementations, manages test resources, and provides reusable test utilities across the entire test suite.
