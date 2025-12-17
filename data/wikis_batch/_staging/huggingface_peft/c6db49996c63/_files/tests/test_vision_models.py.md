# tests/test_vision_models.py

## Understanding

### Purpose
Tests PEFT adapters work correctly with vision models (ResNet, ViT, Swin) including handling of BatchNorm layers and Llava multimodal models with past_key_values.

### Mechanism
- TestResNet tests CONFIGS (lora, loha, lokr, oft, hra, boft) against "peft-internal-testing/resnet-18" with prepare_inputs_for_testing returning {"pixel_values": torch.randn((1,3,224,224))}
- test_merge_layers_with_batchnorm validates merging works with BatchNorm layers which were previously problematic, compares model_before_merging outputs vs merged_model outputs with atol=1e-4
- test_batchnorm_layers_are_trainable validates BatchNorm.weight.requires_grad=True, BatchNorm.bias.requires_grad=True when modules_to_save=["bn1", "bn2"]
- test_batchnorm_with_multiple_adapters validates adding second adapter with different BatchNorm configuration works
- TestViT/TestSwin test same CONFIGS against "peft-internal-testing/vit" and "peft-internal-testing/swin-tiny-patch4-window7-224"
- TestLlavaNext tests LoraConfig against "llava-hf/llava-v1.6-mistral-7b-hf" with test_past_key_values_are_not_set_to_none validating past_key_values preserved during forward, preventing KeyError: 'cache_position' after PR #2188

### Significance
Vision models have unique requirements like BatchNorm layers needing special handling during merge (eval mode, no running stats update) and multimodal models like Llava need past_key_values preservation for efficient generation, critical for preventing regression bugs.
