# Implementation: LLMEngine

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vLLM|https://github.com/vllm-project/vllm]]
* [[source::Doc|vLLM documentation|https://docs.vllm.ai]]
|-
! Domains
| [[domain::LLMs]], [[domain::Inference]], [[domain::Serving]]
|-
! Last Updated
| [[last_updated::2026-02-10 12:00 GMT]]
|}

== Overview ==

LLMEngine is the core class of vLLM: it accepts generation requests, schedules them into iterations, runs the model and returns outputs. It owns the scheduler, the block manager that allocates cache memory, and the workers that execute the network on the accelerators. Higher-level entry points such as the LLM class for offline use and the OpenAI-compatible server are thin layers over it. Its behaviour is set by engine arguments: max_num_seqs bounds how many sequences are scheduled together, block_size sets the number of tokens per cache block, max_model_len limits the context length, and gpu_memory_utilization sets the share of device memory claimed at start-up, with the remainder of that share after weights and activations given to cache blocks.

=== Description ===

LLMEngine is the core class of vLLM, the engine that accepts generation requests, schedules them and runs the model on the accelerators. Higher-level entry points such as the LLM class and the OpenAI-compatible server are thin layers over it. Each call to step schedules a set of sequences, executes one model iteration and returns the request outputs produced by it.

=== Usage ===

Create it from engine arguments, add requests, and loop on step; most users go through the LLM class instead.

== Code Reference ==

<syntaxhighlight lang="python">
from vllm import EngineArgs, LLMEngine
engine = LLMEngine.from_engine_args(EngineArgs(model=name, max_num_seqs=256,
                                                gpu_memory_utilization=0.9))
</syntaxhighlight>

== Related Pages ==

* [[requires_env::Environment:vllm-project_vllm_CUDA_Environment]]
* [[uses_heuristic::Heuristic:vllm-project_vllm_GPU_Memory_Utilization_Setting]]
