# tests/test_vblora.py

## Understanding

### Purpose
Tests VBLoRA (Vector Bank LoRA) adapter which uses shared vector banks across layers and optional topk sparse weights to reduce parameter count while maintaining performance.

### Mechanism
- TestVBLoRA tests MLP with VBLoRAConfig(target_modules=["lin0", "lin1", "lin3"], vector_length=2, num_vectors=10)
- test_vblora_parameters validates vblora_logits_B shape is (out_features//vector_length, r, num_vectors), vblora_logits_A shape is (r, in_features//vector_length, num_vectors), vblora_vector_bank shape is (num_vectors, vector_length)
- Validates vector bank sharing: mlp_vblora.lin0.vblora_vector_bank["default"].data_ptr() == mlp_vblora.lin3.vblora_vector_bank["default"].data_ptr()
- test_save_with_topk_weights validates save_only_topk_weights=True saves vblora_logits_A_topk_indices (shape: r, in_features//vector_length, topk), vblora_logits_A_topk_weights (shape: r, in_features//vector_length, topk-1), but not vblora_logits_A
- test_save_load validates save/load with save_only_topk_weights True/False produces identical outputs
- test_resume_training_model_with_topk_weights validates RuntimeError "Found infinity values in VB-LoRA logits" when resuming training from save_only_topk_weights model
- test_vblora_nb_savable_params_only_topk_weights validates topk_indices_parameter uses factor=0.25 (uint8 dtype), topk_weights_parameter uses (topk-1)

### Significance
VBLoRA reduces parameters by sharing vector banks across layers and using sparse topk weights, enabling efficient fine-tuning at scale while maintaining ability to resume training when save_only_topk_weights=False.
