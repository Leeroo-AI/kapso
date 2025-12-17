# tests/testing_common.py

## Understanding

### Purpose
Provides PeftCommonTester base class with ~40 shared test methods (_test_model_attr, _test_save_pretrained, _test_merge_layers, _test_training, etc.) that all PEFT model test classes inherit from to ensure consistent behavior across adapters.

### Mechanism
- PeftCommonTester defines transformers_class (override in subclass), torch_device via infer_device(), prepare_inputs_for_testing() (override in subclass)
- Core test methods: _test_model_attr validates model.peft_type/base_model_torch_dtype/active_adapters, _test_adapter_name validates adapter renaming, _test_prepare_for_training validates requires_grad only on adapter params
- Save/load methods: _test_save_pretrained validates adapter_model.safetensors/adapter_config.json, _test_save_pretrained_selected_adapters validates selected_adapters parameter, _test_inference_safetensors validates safetensors loading
- Merge methods: _test_merge_layers validates merged outputs == peft outputs, _test_merge_layers_multi validates merging multiple adapters, _test_merge_layers_nan validates merging with NaN base weights
- Training methods: _test_training validates loss decreases, adapter params update but base params don't, _test_training_gradient_checkpointing validates gradient_checkpointing_enable
- Advanced methods: _test_mixed_adapter_batches validates mixed batch inference with adapter_names per sample, _test_weighted_combination_of_adapters validates add_weighted_adapter, _test_delete_adapter validates adapter deletion
- Generation methods: _test_generate validates generate() produces valid outputs, _test_generate_pos_args validates positional args deprecated warning
- Device methods: _test_peft_model_device_map validates device_map="auto" for multi-GPU, _test_modules_devices_mapping validates correct device assignment

### Significance
Centralizes test logic ensuring all PEFT adapters implement consistent behavior across save/load/merge/train/generate operations, preventing regressions and reducing code duplication across 10+ adapter test files.
