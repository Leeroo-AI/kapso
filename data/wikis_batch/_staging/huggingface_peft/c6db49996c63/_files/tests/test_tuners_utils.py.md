# tests/test_tuners_utils.py

## Understanding

### Purpose
Comprehensive testing of BaseTuner utilities including regex matching for target_modules, layers_to_transform/layers_pattern indexing, model/layer status reporting, rank/alpha patterns, and target module optimization.

### Mechanism
- REGEX_TEST_CASES (131 cases) tests check_target_module_exists with combinations of key, target_modules, layers_to_transform, layers_pattern, validates realistic patterns like "transformer.h.1.attn.attention.q_proj" with [1] matches [1] not [0,2]
- TestPeftCustomKwargs tests _maybe_include_all_linear_layers expands INCLUDE_LINEAR_LAYERS_SHORTHAND to ["k_proj", "v_proj", "q_proj", "o_proj", "down_proj", "up_proj", "gate_proj"] for Llama, ["c_attn", "c_proj", "c_fc"] for gpt2 with Conv1D, excludes lm_head/embed_out classification heads
- Tests IA3 feedforward_modules matching, validates is_feedforward attribute set correctly
- Tests BNB quantization (4bit/8bit) with all-linear expansion
- TestTargetedModuleNames/TestTargetedParameterNames validates targeted_module_names/targeted_parameter_names attributes correctly set
- TestExcludedModuleNames tests exclude_modules regex/list matching, validates ValueError when all modules excluded or no modules matched
- TestModelAndLayerStatus validates get_layer_status/get_model_status return LayerStatus/ModelStatus with name, module_type, enabled, active_adapters, merged_adapters, requires_grad, available_adapters, devices for LoRA/LoHa/VeRA/modules_to_save/trainable_tokens
- TestBaseTuner tests get_model_config with to_dict, dict, dataclass configs, tests warn_for_tied_embeddings for lm_head targeting
- TestFindMinimalTargetModules tests _find_minimal_target_modules optimization: ["model.decoder.layers.{i}.self_attn.k_proj" for i in range(12)] compressed to {"k_proj"}, validates MIN_TARGET_MODULES_FOR_OPTIMIZATION threshold
- TestRankAndAlphaPattern validates rank_pattern/alpha_pattern regex matching: {"^foo": 16, "^module.foo": 24, "^module.module.foo": 32} applies different ranks to outer/middle/inner layers

### Significance
Core infrastructure tests ensuring target module matching works correctly across all BaseTuner subclasses, critical for preventing bugs like #2027 (all-linear targeting classification heads), #2390 (nested adapter layers), #2429 (IA3 subset check failure), enabling users to debug configurations via inspect_matched_modules.
