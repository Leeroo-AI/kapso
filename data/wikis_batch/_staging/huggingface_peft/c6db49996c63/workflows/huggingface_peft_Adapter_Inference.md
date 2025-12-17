# huggingface_peft_Adapter_Inference

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|PEFT|https://github.com/huggingface/peft]]
* [[source::Doc|PEFT Documentation|https://huggingface.co/docs/peft]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Deployment]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

Load and deploy a trained PEFT adapter for inference with a pretrained base model.

=== Description ===

This workflow covers the deployment path for trained PEFT adapters. After fine-tuning a model with LoRA or other PEFT methods, the adapter weights must be loaded alongside the base model for inference. The workflow supports multiple deployment scenarios: serving with adapter attached, merging adapter into base for simplified deployment, or loading from HuggingFace Hub.

**Deployment Options:**
* Keep adapter separate: Easy to swap, minimal storage
* Merge into base: Single model file, no runtime overhead
* Quantized inference: Combine with quantization for memory efficiency

=== Usage ===

Execute this workflow when you need to:
* Deploy a fine-tuned PEFT model for inference
* Load adapters trained on different tasks
* Switch between multiple specialized adapters
* Convert PEFT checkpoints for production serving

**Prerequisites:**
* Trained PEFT adapter checkpoint (local or on Hub)
* Access to the same base model used during training
* Inference environment with PyTorch and PEFT installed

== Execution Steps ==

=== Step 1: Load Base Model ===
[[step::Principle:huggingface_peft_Model_Loading]]

Initialize the same base model that was used during adapter training. The model ID must match exactly as the adapter configuration references the base model.

'''Key considerations:'''
* Use same model version/revision as training
* Configure dtype to match deployment requirements
* Set device_map for multi-GPU deployment
* Consider quantization for memory-constrained serving

=== Step 2: Load Trained Adapter ===
[[step::Principle:huggingface_peft_Adapter_Loading]]

Load the trained PEFT adapter onto the base model using PeftModel.from_pretrained(). The adapter weights are loaded and injected into the appropriate model layers.

'''Loading options:'''
* Local path: Load from saved checkpoint directory
* HuggingFace Hub: Load directly from Hub model ID
* Multiple adapters: Load several adapters with different names
* Trainable flag: Set is_trainable=False for inference-only

=== Step 3: Configure Inference Mode ===

Prepare the model for inference by ensuring proper mode settings and optimizations. This step configures the model for production use.

'''Inference setup:'''
* Call model.eval() to disable dropout
* Disable gradient computation with torch.no_grad()
* Optional: Apply torch.compile() for faster inference
* Optional: Enable static KV cache for transformer decoding

=== Step 4: Run Inference ===

Execute model inference using the combined base model + adapter. The forward pass automatically computes base model output plus adapter contributions.

'''Inference flow:'''
* Tokenize input text
* Forward pass through adapted model
* Base model computation + LoRA delta applied
* Generate or compute outputs as needed

=== Step 5: (Optional) Merge and Unload ===
[[step::Principle:huggingface_peft_Adapter_Merging]]

For simplified deployment, merge the adapter weights directly into the base model and remove the PEFT wrapper. This eliminates runtime overhead but loses the ability to swap adapters.

'''Merge process:'''
* Adapter weights merged into base: W' = W + BA * scaling
* PEFT wrapper removed, returns standard model
* Can be saved as regular HuggingFace checkpoint
* Single model file for deployment

== Execution Diagram ==

{{#mermaid:graph TD
    A[Load Base Model] --> B[Load Adapter]
    B --> C[Configure Inference]
    C --> D[Run Inference]
    B --> E{Merge Option}
    E -->|Keep Separate| D
    E -->|Merge| F[Merge and Unload]
    F --> G[Save Merged Model]
    G --> H[Deploy Merged]
}}

== Related Pages ==

* [[step::Principle:huggingface_peft_Model_Loading]]
* [[step::Principle:huggingface_peft_Adapter_Loading]]
* [[step::Principle:huggingface_peft_Adapter_Merging]]
