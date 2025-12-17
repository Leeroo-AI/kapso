# tests/test_xlora.py

## Understanding

### Purpose
Tests X-LoRA (Mixture of LoRA Experts) which dynamically mixes multiple LoRA adapters using learned scaling weights per token, supporting layerwise_scalings, topk_lora, and scalings logging.

### Mechanism
- TestXlora uses saved_lora_adapters fixture: 4 LoRA adapters with target_modules ["q_proj", "v_proj"] (last one adds "k_proj"), saved to tmp_path
- XLoraConfig(hidden_size, xlora_depth=8, adapters={str(i): file_name}) creates X-LoRA classifier with internal_xlora_classifier
- test_functional validates model.generate produces finite outputs, test_forward_hooks_are_cleaned_up ensures forward hooks don't accumulate (issue #1472)
- test_scalings_logging_methods validates enable_scalings_logging(), get_latest_scalings(), get_scalings_log(), get_bucketed_scalings_log() return scalings per token
- test_save_load_functional validates save/load produces identical outputs after_logits == before_logits
- test_topk_lora validates set_topk_lora(2) limits active adapters, test_softmax_topk validates enable_softmax=False + enable_softmax_topk=True
- test_set_override_scaling_pass_value validates scaling_pass_value defaults to 0, can be set to 2 or None (1/n)
- test_disable_adapter validates disable_adapter() produces different outputs than enabled
- test_xlora_loading_valid validates loading from hub "peft-internal-testing/opt-125m-dummy-lora"
- test_per_token_normalization_with_softmax_topk validates scaling weights sum to 1.0 per token
- test_xlora_embed_scale_is_applied validates X-LoRA handles Gemma3 embed_scale correctly

### Significance
X-LoRA enables dynamic mixture-of-experts by learning optimal scaling weights per token for each adapter, critical for combining multiple specialized adapters adaptively rather than using fixed weights, with scalings logging for interpretability.
