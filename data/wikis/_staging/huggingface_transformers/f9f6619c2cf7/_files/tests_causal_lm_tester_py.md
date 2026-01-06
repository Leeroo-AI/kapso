# File: `tests/causal_lm_tester.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 667 |
| Classes | `CausalLMModelTester`, `CausalLMModelTest` |
| Imports | inspect, parameterized, pytest, tempfile, test_configuration_common, test_modeling_common, test_pipeline_mixin, test_training_mixin, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides reusable testing infrastructure for causal language models by offering base classes that standardize model testing across different architectures.

**Mechanism:** Defines two main components: `CausalLMModelTester` that prepares test configurations and inputs with automatic model class inference from naming conventions, and `CausalLMModelTest` that inherits from multiple test mixins to run comprehensive tests including generation, pipelines, and training. The tester automatically infers related model classes (config, sequence classification, token classification, etc.) from the base model class name, reducing boilerplate code. Includes specialized tests for RoPE scaling (linear, dynamic, yarn), Flash Attention 2 equivalence, and various downstream tasks (classification, QA, token tagging).

**Significance:** Core testing utility that enables consistent, comprehensive testing of new causal LM models. By providing a standardized test interface, it ensures all causal language models meet quality standards and behave correctly across different configurations, attention mechanisms, and task heads. Reduces duplication and ensures uniform test coverage across the model zoo.
