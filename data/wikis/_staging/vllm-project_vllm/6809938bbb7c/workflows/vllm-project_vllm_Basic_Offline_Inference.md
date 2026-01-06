{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Quickstart|https://docs.vllm.ai/en/latest/getting_started/quickstart.html]]
|-
! Domains
| [[domain::LLM_Inference]], [[domain::Batch_Processing]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
End-to-end process for running batch text generation with vLLM's offline `LLM` class on local GPU hardware.

=== Description ===
This workflow outlines the standard procedure for running batch inference using vLLM's high-level `LLM` API. The workflow initializes a model with memory-efficient PagedAttention, prepares prompts with configurable sampling parameters, and generates text outputs in a single batch operation. This approach maximizes GPU utilization through continuous batching and is ideal for processing large numbers of prompts without requiring a persistent server.

=== Usage ===
Execute this workflow when you have a collection of prompts to process offline and need high-throughput text generation. This is the simplest vLLM usage pattern, suitable for batch jobs, data processing pipelines, and experimentation without the overhead of running an HTTP server.

== Execution Steps ==

=== Step 1: Model Initialization ===
[[step::Principle:vllm-project_vllm_LLM_Class_Initialization]]

Initialize the vLLM inference engine with model configuration. The `LLM` class loads model weights, allocates KV cache memory using PagedAttention, and prepares the GPU for efficient batch inference. Key configuration options include model path, tensor parallel size, GPU memory utilization, and context length limits.

'''Key considerations:'''
* Model weights are loaded and distributed across available GPUs if using tensor parallelism
* KV cache is pre-allocated based on `gpu_memory_utilization` setting (default 0.9)
* `max_model_len` can be set to limit context window and memory usage
* `enforce_eager` disables CUDA graph capture for debugging

=== Step 2: Sampling Configuration ===
[[step::Principle:vllm-project_vllm_Sampling_Parameters]]

Configure generation parameters using `SamplingParams` to control output behavior. Parameters include temperature for randomness, top-p/top-k for token selection, repetition penalties, and maximum token counts. The sampling configuration determines the quality-diversity tradeoff in generated text.

'''Key parameters:'''
* `temperature` - Controls randomness (0 = greedy, higher = more random)
* `top_p` - Nucleus sampling probability threshold
* `max_tokens` - Maximum number of tokens to generate
* `presence_penalty` / `frequency_penalty` - Discourage repetition
* `stop` - Stop sequences to terminate generation

=== Step 3: Prompt Preparation ===
[[step::Principle:vllm-project_vllm_Prompt_Formatting]]

Prepare prompts in the format expected by the model. For chat models, apply the appropriate chat template. For base models, use raw text prompts. Prompts can be provided as strings, token IDs, or structured message lists for chat completion.

'''Input formats supported:'''
* Plain text strings for completion models
* List of message dictionaries for chat models (using `llm.chat()`)
* Pre-tokenized `TokensPrompt` for advanced use cases
* Multi-modal inputs (images, audio) for VLM models

=== Step 4: Batch Generation ===
[[step::Principle:vllm-project_vllm_Batch_Generation]]

Execute text generation across all prompts using the `generate()` method. vLLM uses continuous batching to dynamically schedule requests, maximizing GPU utilization. The engine handles memory management, attention computation, and token sampling efficiently across the batch.

'''What happens:'''
* Prompts are added to the scheduler queue
* Engine processes prefill (prompt encoding) and decode (token generation) phases
* PagedAttention manages KV cache blocks dynamically
* Results are returned as `RequestOutput` objects containing generated text

=== Step 5: Output Processing ===
[[step::Principle:vllm-project_vllm_Output_Processing]]

Process the generation results from `RequestOutput` objects. Each output contains the original prompt, generated completions (potentially multiple if `n > 1`), token IDs, and optionally log probabilities. Extract the generated text and any metadata needed for downstream processing.

'''Output structure:'''
* `output.prompt` - The input prompt text
* `output.outputs` - List of `CompletionOutput` objects
* `output.outputs[i].text` - Generated text for completion i
* `output.outputs[i].token_ids` - Token IDs of generated sequence
* `output.outputs[i].logprobs` - Per-token log probabilities (if requested)

== Execution Diagram ==
{{#mermaid:graph TD
    A[Model Initialization] --> B[Sampling Configuration]
    B --> C[Prompt Preparation]
    C --> D[Batch Generation]
    D --> E[Output Processing]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_LLM_Class_Initialization]]
* [[step::Principle:vllm-project_vllm_Sampling_Parameters]]
* [[step::Principle:vllm-project_vllm_Prompt_Formatting]]
* [[step::Principle:vllm-project_vllm_Batch_Generation]]
* [[step::Principle:vllm-project_vllm_Output_Processing]]
