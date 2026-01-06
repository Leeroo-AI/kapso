{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Model Loading]], [[domain::Sharded State]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates how to load models saved in the sharded_state format for fast initialization.

=== Description ===
The load sharded state example shows how to load models that were previously saved using vLLM's sharded_state format. This format pre-shards model weights according to the tensor parallel configuration, enabling significantly faster model loading compared to loading from standard checkpoints. When using tensor parallelism, each GPU can directly load its shard without requiring full model loading and redistribution. This is particularly beneficial for large models and repeated initialization scenarios, reducing startup time from minutes to seconds.

=== Usage ===
Use sharded_state loading when you need fast model initialization with tensor parallelism, frequently restart vLLM instances with the same model, or want to minimize startup latency in production deployments. The model must first be saved in sharded_state format using save_sharded_state.py.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/load_sharded_state.py examples/offline_inference/load_sharded_state.py]

=== CLI Usage ===

==== Step 1: Save Model in Sharded Format ====
<syntaxhighlight lang="bash">
python save_sharded_state.py \
    --model /path/to/load \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --output /path/to/save/sharded/model
</syntaxhighlight>

==== Step 2: Load Sharded Model ====
<syntaxhighlight lang="bash">
python load_sharded_state.py \
    --model /path/to/saved/sharded/model \
    --load-format sharded_state \
    --quantization deepspeedfp \
    --tensor-parallel-size 8 \
    --prompt "Hello, my name is" \
    --max-tokens 50
</syntaxhighlight>

== Key Concepts ==

=== Sharded State Format ===
The sharded_state format provides optimized loading:
* Weights are pre-sharded by tensor parallel rank
* Each GPU loads only its portion of the model
* Eliminates redistribution overhead during initialization
* Preserves quantization format (e.g., deepspeedfp)
* Directory structure contains per-rank checkpoint files

=== Load Format Configuration ===
The load_format parameter controls how weights are loaded:
* '''sharded_state''': Load pre-sharded weights directly
* '''auto''': Automatically detect format (default)
* Must match tensor_parallel_size used during saving
* Incompatible configurations will fail at load time

=== Engine Args Integration ===
The example uses EngineArgs for configuration:
* Accepts all standard vLLM engine arguments
* load_format defaults to "sharded_state" in this example
* Supports quantization, tensor parallelism, and other options
* Arguments passed via CLI and converted to LLM kwargs

== Usage Examples ==

=== Loading Sharded Model ===
<syntaxhighlight lang="python">
import dataclasses
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    # Default to sharded_state format
    parser.set_defaults(load_format="sharded_state")

    # Validation arguments
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)

    return parser.parse_args()

args = parse_args()
engine_args = EngineArgs.from_cli_args(args)

print(f"Loading model from {engine_args.model}")
print(f"Using format {engine_args.load_format}")
print(f"Tensor parallel size: {engine_args.tensor_parallel_size}")

# Load the sharded model
llm = LLM(**dataclasses.asdict(engine_args))
</syntaxhighlight>

=== Running Inference ===
<syntaxhighlight lang="python">
# Prepare sampling parameters
sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
)

print(f"\nRunning inference:")
print(f"Prompt: {args.prompt}")

# Generate completion
outputs = llm.generate(args.prompt, sampling_params)

# Display results
print("\nGenerated outputs:")
for output in outputs:
    generated_text = output.outputs[0].text
    print("-" * 50)
    print(f"Full output: {args.prompt}{generated_text}")
    print("-" * 50)
</syntaxhighlight>

=== Performance Comparison ===
<syntaxhighlight lang="python">
import time

# Time standard loading
start = time.time()
llm_standard = LLM(
    model="meta-llama/Meta-Llama-3-70B",
    tensor_parallel_size=8,
    load_format="auto"
)
standard_time = time.time() - start

# Time sharded loading
start = time.time()
llm_sharded = LLM(
    model="/path/to/sharded/Llama-3-70B",
    tensor_parallel_size=8,
    load_format="sharded_state"
)
sharded_time = time.time() - start

print(f"Standard loading: {standard_time:.2f}s")
print(f"Sharded loading: {sharded_time:.2f}s")
print(f"Speedup: {standard_time / sharded_time:.2f}x")
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
