# Adapter Loading & Inference

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Model_Serving]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

End-to-end process for loading pretrained PEFT adapters and running inference, enabling deployment of specialized models from small adapter checkpoints.

=== Description ===

This workflow covers loading saved PEFT adapters onto base models for inference. Adapters can be loaded from local paths or the HuggingFace Hub. The workflow supports loading multiple adapters, switching between them at runtime, and optionally merging adapters into the base model for production deployment with zero inference overhead.

=== Usage ===

Execute this workflow when:
* You have a trained PEFT adapter and need to run inference
* You want to load adapters from HuggingFace Hub
* You need to switch between multiple task-specific adapters
* You're deploying a PEFT model in production

== Execution Steps ==

=== Step 1: Load Base Model ===
[[step::Principle:huggingface_peft_Base_Model_Loading]]

Load the same base model that was used during adapter training. The base model ID is stored in the adapter config, but you can also specify it explicitly if using a different checkpoint.

'''Loading options:'''
* Standard precision: Load in float16/bfloat16 for inference
* Quantized: Load in 4-bit/8-bit for memory-constrained deployment
* Device placement: Use `device_map="auto"` for multi-GPU inference

=== Step 2: Load PEFT Adapter ===
[[step::Principle:huggingface_peft_Adapter_Loading]]

Load the pretrained adapter using `PeftModel.from_pretrained()`. This reads the adapter configuration and weights, then injects the adapter layers into the base model.

'''Loading sources:'''
* Local path: `PeftModel.from_pretrained(model, "./path/to/adapter")`
* HuggingFace Hub: `PeftModel.from_pretrained(model, "username/adapter-name")`
* Specific revision: Use `revision` parameter for version control

'''Options:'''
* `is_trainable=False`: Load in inference mode (default)
* `adapter_name`: Custom name for multi-adapter scenarios

=== Step 3: Configure Inference Mode ===
[[step::Principle:huggingface_peft_Inference_Configuration]]

Set up the model for efficient inference. This includes disabling dropout, setting eval mode, and optionally enabling optimizations like torch.compile or Flash Attention.

'''Configuration:'''
* Call `model.eval()` to disable dropout and batch norm updates
* Disable gradient computation with `torch.no_grad()` context
* Enable inference optimizations (e.g., `torch.compile`)

=== Step 4: Run Inference ===
[[step::Principle:huggingface_peft_Inference_Execution]]

Execute forward passes through the PEFT model. The adapter modifies the base model's behavior according to its learned weights.

'''Inference patterns:'''
* Text generation: Use `model.generate()` with appropriate parameters
* Classification: Forward pass and extract logits
* Embeddings: Extract hidden states from intermediate layers

=== Step 5: (Optional) Merge Adapter ===
[[step::Principle:huggingface_peft_Adapter_Merging_Into_Base]]

For production deployment where inference latency is critical, merge the adapter weights into the base model. This eliminates the adapter computation overhead but makes the model permanent (can't switch adapters).

'''Merging:'''
* `model.merge_and_unload()`: Merge adapter into base and remove PEFT wrapper
* Results in standard transformers model with adapter effects baked in
* No runtime overhead from adapter computation

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load Base Model] --> B[Load PEFT Adapter]
    B --> C[Configure Inference]
    C --> D[Run Inference]
    D --> E{Production Deploy?}
    E -->|Yes| F[Merge Adapter]
    E -->|No| G[Keep Adapter Separate]
    F --> H[Standard Model Inference]
    G --> D
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Base_Model_Loading]]
* [[step::Principle:huggingface_peft_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Inference_Configuration]]
* [[step::Principle:huggingface_peft_Inference_Execution]]
* [[step::Principle:huggingface_peft_Adapter_Merging_Into_Base]]
