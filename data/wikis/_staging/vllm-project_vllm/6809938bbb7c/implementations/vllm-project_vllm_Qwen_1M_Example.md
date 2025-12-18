{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Long Context]], [[domain::Chunked Prefill]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates processing prompts with up to 1 million tokens using Qwen2.5 models with chunked prefill.

=== Description ===
The Qwen 1M example showcases vLLM's ability to handle extremely long context windows of up to 1 million tokens using the Qwen2.5-7B-Instruct-1M model. The example downloads test prompts of varying lengths (64K, 200K, 600K, or 1M tokens) and processes them using chunked prefill with tensor parallelism. Chunked prefill breaks the long prompt into smaller chunks for memory-efficient processing, while tensor parallelism distributes the model across multiple GPUs. This capability is crucial for processing entire books, large codebases, or extensive documentation in a single context.

=== Usage ===
Use this pattern when processing documents or contexts exceeding 100K tokens, analyzing entire codebases or books, performing long-range reasoning tasks, or building applications requiring comprehensive document understanding. Requires multiple high-memory GPUs and the VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 environment variable.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/qwen_1m.py examples/offline_inference/qwen_1m.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/qwen_1m.py
</syntaxhighlight>

== Key Concepts ==

=== Million-Token Context Configuration ===
The example configures vLLM for extreme long-context processing:
* '''max_model_len''': 1,048,576 (1M tokens)
* '''tensor_parallel_size''': 4 - distribute model across GPUs
* '''enforce_eager''': True - disable CUDA graphs for memory savings
* '''enable_chunked_prefill''': True - process prompt in chunks
* '''max_num_batched_tokens''': 131,072 - chunk size for prefill

=== Environment Variable ===
Million-token contexts require explicit opt-in:
* '''VLLM_ALLOW_LONG_MAX_MODEL_LEN=1''': Enable extremely long contexts
* Safety flag to prevent accidental OOM errors
* Must be set before importing vLLM

=== Chunked Prefill Strategy ===
Chunked prefill enables memory-efficient long-context processing:
* Breaks prompt into max_num_batched_tokens chunks (131K tokens)
* Processes chunks sequentially to stay within memory limits
* KV cache built incrementally across chunks
* Trading latency for memory efficiency

=== Test Prompt Datasets ===
Qwen provides test prompts of various lengths:
* 64K tokens: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/64k.txt
* 200K tokens: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/200k.txt
* 600K tokens: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt
* 1M tokens: https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/1m.txt

== Usage Examples ==

=== Environment Setup ===
<syntaxhighlight lang="python">
import os
from urllib.request import urlopen

# Enable million-token contexts (must be set before importing vLLM)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from vllm import LLM, SamplingParams
</syntaxhighlight>

=== Loading Long Prompts ===
<syntaxhighlight lang="python">
def load_prompt() -> str:
    """Download a long-context test prompt."""
    # Options: 64k.txt, 200k.txt, 600k.txt, 1m.txt
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt"

    with urlopen(url, timeout=5) as response:
        prompt = response.read().decode("utf-8")

    return prompt

prompt = load_prompt()
print(f"Loaded prompt with ~{len(prompt.split())} words")
</syntaxhighlight>

=== Initializing LLM for Million-Token Context ===
<syntaxhighlight lang="python">
def initialize_engine() -> LLM:
    """Create LLM configured for million-token contexts."""
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-1M",
        max_model_len=1048576,           # 1M tokens
        tensor_parallel_size=4,          # 4 GPUs
        enforce_eager=True,              # Disable CUDA graphs
        enable_chunked_prefill=True,     # Enable chunked processing
        max_num_batched_tokens=131072,   # 131K token chunks
    )
    return llm

llm = initialize_engine()
</syntaxhighlight>

=== Processing Long Context ===
<syntaxhighlight lang="python">
def process_requests(llm: LLM, prompts: list[str]) -> None:
    """Process prompts with long-context generation."""
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        detokenize=True,
        max_tokens=256,
    )

    # Generate with long context
    outputs = llm.generate(prompts, sampling_params)

    # Display results
    for output in outputs:
        prompt_token_ids = output.prompt_token_ids
        generated_text = output.outputs[0].text
        print(
            f"Prompt length: {len(prompt_token_ids)} tokens, "
            f"Generated text: {generated_text!r}"
        )

# Run inference
prompt = load_prompt()
process_requests(llm, [prompt])
</syntaxhighlight>

=== Complete Example ===
<syntaxhighlight lang="python">
import os
from urllib.request import urlopen
from vllm import LLM, SamplingParams

# Enable million-token contexts
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

def main():
    # Initialize engine
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-1M",
        max_model_len=1048576,
        tensor_parallel_size=4,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=131072,
    )

    # Load 600K token test case
    with urlopen(
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt",
        timeout=5,
    ) as response:
        prompt = response.read().decode("utf-8")

    # Configure generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=256,
    )

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        print(f"Prompt: {len(output.prompt_token_ids)} tokens")
        print(f"Generated: {output.outputs[0].text}")

if __name__ == "__main__":
    main()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
* [[related_to::Implementation:vllm-project_vllm_Context_Extension_Example]]
