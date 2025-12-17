# tests/test_target_parameters.py

## Understanding

### Purpose
Tests LoRA targeting of nn.Parameters directly (not just modules) for expert mixture models like Llama4 and GptOss where individual parameters like gate_up_proj and down_proj need separate adaptation.

### Mechanism
- ALL_CONFIGS tests 8 configurations on tiny-Llama4ForCausalLM and tiny-GptOssForCausalLM: targeting down_proj alone, gate_up_proj+down_proj on different modules, both on same module, mixing target_modules (q_proj, v_proj) with target_parameters (down_proj)
- TestDecoderModelsTargetParameters extends PeftCommonTester but skips tests requiring multiple adapters (add_adapter, delete_adapter, weighted_combination) as target_parameters doesn't support multiple adapters yet
- TestTargetParameters.test_targeting_module_and_targeting_param_equivalent validates target_modules=["gate_proj"] produces identical results to target_parameters=["gate_proj.weight"]
- test_target_multiple_parameters_on_same_module uses monkeypatch to track _LoraParameterProxy.forward calls ensuring both gate_up_proj and down_proj are used
- test_target_parameters_works_with_existing_parametrization validates LoRA parametrization composes with nn.utils.parametrize
- test_target_parameter_result_caching_works verifies results are cached (only 2 forward calls per step, not 25)

### Significance
Enables LoRA to target individual parameters in complex architectures where modules contain multiple parameters requiring different adaptation strategies, critical for expert mixture models where gate_up_proj and down_proj need independent control.
