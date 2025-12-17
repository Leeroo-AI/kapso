# File: `tests/causal_lm_tester.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 667 |
| Classes | `CausalLMModelTester`, `CausalLMModelTest` |
| Imports | inspect, parameterized, pytest, tempfile, test_configuration_common, test_modeling_common, test_pipeline_mixin, test_training_mixin, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unified test framework for causal language models with automatic model class inference and standardized testing.

**Mechanism:** Provides CausalLMModelTester for test data preparation and CausalLMModelTest class that combines ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, and TrainingTesterMixin. Automatically infers model class names from base_model_class using standard naming conventions, validates configurations, and tests sequence classification, token classification, question answering, and RoPE scaling features including linear, dynamic, and yarn scaling types.

**Significance:** Critical test infrastructure that enables model developers to inherit comprehensive test coverage by setting only base_model_class, dramatically reducing boilerplate test code while ensuring consistency across all causal LM model implementations in the transformers library.
