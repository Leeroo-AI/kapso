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

**Mechanism:** Defines pytest fixtures for test infrastructure including: multimodal assets (images/videos/audio), model runners (HfRunner for HuggingFace, VllmRunner for vLLM), test environment setup (distributed init, workspace init, cleanup), logging capture fixtures, dummy model paths, and HTTP asset server. Provides test infrastructure for comparing vLLM against HuggingFace baseline.

**Significance:** Core test infrastructure file that centralizes fixture definitions, model runner implementations, and common test utilities. Essential for test reproducibility and standardized testing patterns across the test suite. Enables comprehensive integration testing between vLLM and HuggingFace implementations.
