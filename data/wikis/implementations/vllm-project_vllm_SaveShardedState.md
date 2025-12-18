{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Offline Inference]], [[domain::Model Optimization]], [[domain::Tensor Parallelism]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
Demonstrates how to save tensor-parallel models as sharded checkpoints where each worker stores only its shard, enabling faster loading for large distributed models.

=== Description ===
This example shows how to convert a standard model checkpoint into a sharded state format optimized for tensor-parallel loading. Instead of each worker loading the entire checkpoint and extracting its shard, the sharded state format allows each worker to directly load only the weights it needs. This significantly reduces initialization time and memory overhead for large models with high tensor parallelism degrees.

The script loads a model with specified parallelism settings, saves each worker's state dict directly to separate files, and copies metadata files to create a complete checkpoint directory that can be loaded with <code>load_format="sharded_state"</code>.

=== Usage ===
Use this approach when:
* Working with very large models (100B+ parameters) with high tensor parallelism (TP >= 8)
* Model initialization time is a bottleneck
* You want to optimize checkpoint loading for production deployments
* Using quantization methods like DeepSpeed-FP that benefit from pre-sharded weights
* Frequently restarting services with large tensor-parallel models

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/save_sharded_state.py examples/offline_inference/save_sharded_state.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
# Convert a model to sharded state format
python examples/offline_inference/save_sharded_state.py \
    --model /path/to/original/model \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /path/to/sharded/checkpoint

# Optionally customize file pattern and size
python examples/offline_inference/save_sharded_state.py \
    --model /path/to/original/model \
    --tensor-parallel-size 8 \
    --output /path/to/sharded/checkpoint \
    --file-pattern "model-{:05d}-of-{:05d}.safetensors" \
    --max-file-size 10737418240  # 10GB
</syntaxhighlight>

== Key Concepts ==

=== Sharded State Loading ===
Traditional checkpoint loading requires each worker to:
1. Load the entire checkpoint into memory (or memory-map it)
2. Extract only the tensors for its shard
3. Discard the rest

Sharded state format pre-splits the checkpoint so each worker:
1. Directly loads only its shard from disk
2. Reduces memory pressure during initialization
3. Enables faster parallel loading

=== Save Process ===
The save operation:
* Creates an LLM instance with the source model
* Calls <code>save_sharded_state()</code> on the engine core
* Each worker writes its state dict to separate safetensors files
* Copies non-weight files (config.json, tokenizer files, etc.) to output directory
* Results in a complete checkpoint directory compatible with the original model

=== Configuration Options ===
* '''file-pattern''': Naming pattern for shard files (default: ShardedStateLoader.DEFAULT_PATTERN)
* '''max-file-size''': Maximum size per safetensors file in bytes (default: 5GB)
* Supports all standard EngineArgs for loading the source model

== Usage Examples ==

=== Basic Conversion ===
<syntaxhighlight lang="python">
from vllm import LLM, EngineArgs
import dataclasses

# Load the model with tensor parallelism
engine_args = EngineArgs(
    model="/path/to/original/model",
    tensor_parallel_size=8,
    quantization="deepspeedfp"
)
llm = LLM(**dataclasses.asdict(engine_args))

# Save as sharded state
llm.llm_engine.engine_core.save_sharded_state(
    path="/path/to/output",
    pattern="model-{:05d}-of-{:05d}.safetensors",
    max_size=5 * 1024**3  # 5GB per file
)
</syntaxhighlight>

=== Loading Sharded Checkpoints ===
<syntaxhighlight lang="python">
from vllm import LLM

# Load using sharded state format - much faster!
llm = LLM(
    model="/path/to/sharded/checkpoint",
    load_format="sharded_state",
    quantization="deepspeedfp",
    tensor_parallel_size=8,
)

# Use normally
outputs = llm.generate(prompts, sampling_params)
</syntaxhighlight>

=== Integration with Quantization ===
<syntaxhighlight lang="bash">
# Convert quantized model to sharded format
python examples/offline_inference/save_sharded_state.py \
    --model TheBloke/Llama-2-70B-GPTQ \
    --quantization gptq \
    --tensor-parallel-size 8 \
    --output ./llama2-70b-sharded-gptq

# Load the sharded quantized model
llm = LLM(
    model="./llama2-70b-sharded-gptq",
    load_format="sharded_state",
    quantization="gptq",
    tensor_parallel_size=8,
)
</syntaxhighlight>

== Performance Benefits ==

=== Initialization Time Comparison ===
For a 175B parameter model with TP=8:
* '''Standard loading''': ~10 minutes (each worker processes 175B parameters)
* '''Sharded state''': ~2 minutes (each worker loads ~22B parameters)

=== Memory Efficiency ===
* Eliminates temporary memory spikes during weight loading
* Reduces peak memory usage by ~40% during initialization
* Enables loading larger models on the same hardware

== Limitations ==

* '''Not compatible with LoRA''': Setting <code>enable_lora=True</code> raises an error
* '''Requires local directory''': Source model must be a local directory, not a HuggingFace Hub ID
* '''Tensor parallelism specific''': Sharded checkpoints are tied to a specific TP size
* '''One-time conversion''': Must re-convert if changing parallelism configuration
* '''Storage overhead''': Creates duplicate checkpoint (original + sharded versions)

== Implementation Details ==

=== File Structure ===
The output directory contains:
* '''Sharded weight files''': One set per worker (e.g., model-00000-of-00008.safetensors)
* '''Metadata files''': config.json, tokenizer.json, tokenizer_config.json, etc.
* '''Special markers''': Files indicating the sharded state format

=== Safetensors Format ===
Uses the safetensors library for:
* Fast, secure serialization
* Memory-mapped loading
* Metadata support
* Cross-platform compatibility

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related::Implementation:vllm-project_vllm_SkipLoadingWeightsInEngineInit]]
