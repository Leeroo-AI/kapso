{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM Documentation|https://docs.vllm.ai]]
* [[source::Paper|PagedAttention|https://arxiv.org/abs/2309.06180]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2025-12-17 19:00 GMT]]
|}

== Overview ==

End-to-end process for high-throughput text generation using vLLM's offline batch inference capabilities with PagedAttention memory management.

=== Description ===

This workflow demonstrates the standard procedure for running offline batch inference with vLLM. It leverages the `LLM` class to efficiently process multiple prompts in parallel, utilizing PagedAttention for optimal GPU memory management. The process covers model initialization with configuration, sampling parameter definition, and batch generation with streaming support.

Key capabilities:
* **High throughput**: Continuous batching maximizes GPU utilization
* **Memory efficiency**: PagedAttention manages KV cache dynamically
* **Flexible sampling**: Temperature, top-p, top-k, repetition penalties
* **Structured outputs**: JSON schema, regex, and grammar constraints

=== Usage ===

Execute this workflow when you need to process a batch of text prompts for generation tasks such as:
* Text completion and summarization
* Question answering
* Code generation
* Translation

The workflow is ideal for scenarios where you have a collection of prompts to process offline and want to maximize throughput rather than optimize for individual request latency.

== Execution Steps ==

=== Step 1: Initialize Engine Configuration ===
[[step::Principle:vllm-project_vllm_Engine_Configuration]]

Configure the inference engine with model selection and resource allocation parameters. This involves specifying the model path (HuggingFace ID or local), setting memory utilization limits, and defining parallelism strategy (tensor parallel, data parallel).

'''Key considerations:'''
* Select appropriate `gpu_memory_utilization` (default 0.9) based on available VRAM
* Set `max_model_len` to limit context window and reduce memory
* Configure `max_num_seqs` to control concurrent request batching
* Enable `trust_remote_code` for custom model architectures

=== Step 2: Define Sampling Parameters ===
[[step::Principle:vllm-project_vllm_Sampling_Configuration]]

Create sampling parameter objects that control the generation behavior. Parameters include temperature for randomness, nucleus sampling (top-p), and various penalty factors to control repetition and diversity.

'''Key parameters:'''
* `temperature`: Controls randomness (0 = greedy, higher = more random)
* `top_p`: Nucleus sampling threshold (cumulative probability cutoff)
* `top_k`: Limits token selection to top-k candidates
* `max_tokens`: Maximum number of tokens to generate
* `stop`: Stop sequences to terminate generation
* `presence_penalty` / `frequency_penalty`: Control token repetition

=== Step 3: Instantiate LLM Engine ===
[[step::Principle:vllm-project_vllm_Model_Loading]]

Load the model and tokenizer into GPU memory using the configured engine arguments. The LLM class handles model downloading (if needed), weight loading, and initialization of the PagedAttention KV cache.

'''What happens:'''
* Model weights are loaded from HuggingFace Hub or local path
* Tokenizer is initialized with appropriate chat template
* KV cache blocks are pre-allocated based on memory configuration
* CUDA graphs are captured for optimized execution (unless enforce_eager=True)

=== Step 4: Prepare Input Prompts ===
[[step::Principle:vllm-project_vllm_Input_Formatting]]

Format input prompts for the model. This can include raw text strings, tokenized inputs, or chat-formatted messages. For chat models, apply the appropriate chat template to format multi-turn conversations.

'''Input formats supported:'''
* Raw text strings
* Pre-tokenized token IDs (`TokensPrompt`)
* Chat messages with roles (system, user, assistant)
* Multimodal inputs (images, audio) for VLMs

=== Step 5: Execute Batch Generation ===
[[step::Principle:vllm-project_vllm_Batch_Generation]]

Submit prompts for parallel generation using continuous batching. The engine schedules requests dynamically, processes them in batches, and returns `RequestOutput` objects containing the generated text and metadata.

'''Generation process:'''
1. Prompts are tokenized and added to the request queue
2. Scheduler batches requests based on available KV cache
3. Forward passes generate tokens iteratively
4. Outputs are returned when stop conditions are met

=== Step 6: Process Generation Results ===
[[step::Principle:vllm-project_vllm_Output_Processing]]

Extract and process the generated outputs from the `RequestOutput` objects. Each output contains the generated text, token IDs, log probabilities (if requested), and finish reason.

'''Output structure:'''
* `outputs[i].text`: Generated text string
* `outputs[i].token_ids`: Token ID sequence
* `outputs[i].logprobs`: Per-token log probabilities
* `outputs[i].finish_reason`: Why generation stopped (length, stop, etc.)

== Execution Diagram ==
{{#mermaid:graph TD
    A[Engine Configuration] --> B[Define Sampling Parameters]
    B --> C[Instantiate LLM Engine]
    C --> D[Prepare Input Prompts]
    D --> E[Execute Batch Generation]
    E --> F[Process Generation Results]
}}

== Related Pages ==
* [[step::Principle:vllm-project_vllm_Engine_Configuration]]
* [[step::Principle:vllm-project_vllm_Sampling_Configuration]]
* [[step::Principle:vllm-project_vllm_Model_Loading]]
* [[step::Principle:vllm-project_vllm_Input_Formatting]]
* [[step::Principle:vllm-project_vllm_Batch_Generation]]
* [[step::Principle:vllm-project_vllm_Output_Processing]]
