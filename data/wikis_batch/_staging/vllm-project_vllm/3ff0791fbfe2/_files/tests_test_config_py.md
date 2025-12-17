# File: `tests/test_config.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1052 |
| Classes | `_TestConfigFields`, `_TestNestedConfig`, `MockConfig` |
| Functions | `test_compile_config_repr_succeeds`, `test_get_field`, `test_update_config`, `test_auto_runner`, `test_pooling_runner`, `test_draft_runner`, `test_disable_sliding_window`, `test_get_pooling_config`, `... +22 more` |
| Imports | dataclasses, logging, os, pydantic, pytest, unittest, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration system validation tests

**Mechanism:** Tests ModelConfig, PoolerConfig, CompilationConfig, SchedulerConfig, and VllmConfig classes. Validates model runner detection, pooling type inference, MoE model detection, quantization detection, RoPE customization, nested hf_overrides, chunked prefill support, prefix caching support, optimization level defaults, S3 model loading, and custom op enabling.

**Significance:** Ensures configuration system correctly handles model metadata, optimization settings, hardware-specific features, and provides proper defaults for different model types and deployment scenarios.
