{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Context Windows]], [[domain::RoPE]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates how to extend the context length of a model using the YARN method with RoPE parameters.

=== Description ===
The context extension example shows how to dynamically extend a model's maximum context window beyond its training length using YARN (Yet Another RoPE Scaling) method. By configuring rope_parameters in hf_overrides, the example extends Qwen-3-0.6B from its original 32,768 token limit to 131,072 tokens (4x extension) without requiring model retraining. This technique is essential for processing longer documents or conversations while maintaining model performance.

=== Usage ===
Use this pattern when you need to process inputs longer than a model's default context window, such as long documents, extended conversations, or large codebases. This approach works well with models that support RoPE-based position embeddings.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/context_extension.py examples/offline_inference/context_extension.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/context_extension.py
</syntaxhighlight>

== Key Concepts ==

=== RoPE YARN Scaling ===
The example uses YARN (Yet Another RoPE Scaling) to extend context windows:
* '''rope_theta''': Base frequency for rotary position embeddings (1,000,000)
* '''rope_type''': "yarn" - interpolation method for position embeddings
* '''factor''': 4.0 - multiplier for context extension (4x original length)
* '''original_max_position_embeddings''': 32,768 - model's training context length

=== HF Overrides Configuration ===
The hf_overrides parameter allows customizing model configuration without modifying checkpoint files:
* Overrides Hugging Face config values at model load time
* Enables max_model_len to be set to extended length (131,072 tokens)
* No retraining or fine-tuning required

== Usage Examples ==

<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

# Configure YARN context extension
rope_theta = 1000000
original_max_position_embeddings = 32768
factor = 4.0

hf_overrides = {
    "rope_parameters": {
        "rope_theta": rope_theta,
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings,
    },
    "max_model_len": int(original_max_position_embeddings * factor),
}

# Create LLM with extended context
llm = LLM(model="Qwen/Qwen3-0.6B", hf_overrides=hf_overrides)

# Generate with chat interface
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128,
)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
]

outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related_to::Implementation:vllm-project_vllm_Qwen_1M_Example]]
