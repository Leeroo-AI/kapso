# tests/test_stablediffusion.py

## Understanding

### Purpose
Validates PEFT adapters work correctly with diffusers StableDiffusion pipelines by testing both text_encoder and unet components with separate adapter configurations.

### Mechanism
- TestStableDiffusionModel tests DIFFUSERS_CONFIGS (LoRA, LoHa, LoKr, OFT, BOFT, HRA) against "hf-internal-testing/tiny-sd-pipe"
- instantiate_sd_peft creates two separate PEFT configs: text_encoder_kwargs targets ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"], unet_kwargs targets ["proj_in", "proj_out", "to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
- Tests merge_layers and merge_layers_safe_merge by comparing peft_output vs merged_output with atol=1.0 (uint8 image tolerance)
- Tests add_weighted_adapter_base_unchanged ensures adapter config unchanged after creating weighted adapter
- Tests disable_adapter, load_model_low_cpu_mem_usage with inject_adapter_in_model and set_peft_model_state_dict on meta device

### Significance
Critical for validating PEFT works with diffusion models which have complex architectures (separate text_encoder and unet) and require special handling for image generation tasks where numerical precision is less critical (atol=1.0 for uint8 images).
