= LLM.generate() Method =

== Metadata ==
{| class="wikitable"
|-
! Field !! Value
|-
| Knowledge Sources || vllm/entrypoints/llm.py
|-
| Domains || API Design, Text Generation, Inference
|-
| Last Updated || 2025-12-17
|}

== Overview ==
The <code>generate()</code> method is the primary interface for text generation in vLLM's LLM class. It accepts prompts and sampling parameters, executes inference through the underlying engine, and returns generated text completions. The method supports both single-worker and data parallel execution modes.

== Description ==
<code>generate()</code> implements a synchronous, blocking API for batch text generation. It:

* Validates and prepares input prompts
* Applies sampling parameters to control generation
* Submits requests to the engine's scheduler
* Runs the engine until all requests complete
* Returns structured output objects with generated text

In data parallel mode, each worker independently calls <code>generate()</code> on its partition of prompts, with no inter-worker communication during execution.

== Code Reference ==

=== Source Location ===
* '''File''': <code>/tmp/praxium_repo_583nq7ea/vllm/entrypoints/llm.py</code>
* '''Lines''': 344-434 (method implementation)

=== Method Signature ===
<syntaxhighlight lang="python">
def generate(
    self,
    prompts: PromptType | Sequence[PromptType] | None = None,
    /,
    *,
    sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    use_tqdm: bool = True,
    lora_request: LoRARequest | Sequence[LoRARequest] | None = None,
    priority: int | Sequence[int] = 0,
    prompt_token_ids: Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> list[RequestOutput]:
    """Generates texts from given prompts and sampling parameters.

    This method is synchronous and will not return until all
    completions are finished.

    Args:
        prompts: Prompts to generate completions for. Can be:
            - A single text string
            - A list of text strings
            - A single prompt dict with "prompt" and optional "multi_modal_data"
            - A list of prompt dicts
        sampling_params: Sampling parameters for text generation. Can be:
            - A single SamplingParams object (used for all prompts)
            - A list of SamplingParams objects (one per prompt)
            If None, uses default parameters.
        use_tqdm: Whether to display a progress bar.
        lora_request: LoRA request for the generation.
        priority: Priority level(s) for the requests.
        prompt_token_ids: Token IDs for prompts (legacy parameter).

    Returns:
        A list of RequestOutput objects containing the generated
        completions in the same order as the input prompts.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
</syntaxhighlight>

== I/O Contract ==

=== Input Parameters ===
{| class="wikitable"
|-
! Parameter !! Type !! Default !! Description
|-
| prompts || str / list[str] / dict / list[dict] || required || Input prompt(s) to generate from
|-
| sampling_params || SamplingParams / list[SamplingParams] || None || Controls generation behavior (temperature, top_p, max_tokens, etc.)
|-
| use_tqdm || bool || True || Show progress bar during generation
|-
| lora_request || LoRARequest / list[LoRARequest] || None || LoRA adapters to apply
|-
| priority || int / list[int] || 0 || Priority level for request scheduling
|-
| prompt_token_ids || list[int] / list[list[int]] || None || Pre-tokenized prompts (legacy)
|}

=== Output ===
{| class="wikitable"
|-
! Return !! Type !! Description
|-
| outputs || list[RequestOutput] || List of output objects, one per input prompt
|}

=== RequestOutput Structure ===
Each <code>RequestOutput</code> contains:
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| request_id || str || Unique identifier for the request
|-
| prompt || str || Original input prompt
|-
| prompt_token_ids || list[int] || Tokenized prompt
|-
| outputs || list[CompletionOutput] || Generated completion(s)
|-
| finished || bool || Whether generation is complete
|}

=== CompletionOutput Structure ===
Each <code>CompletionOutput</code> in <code>outputs</code> contains:
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| index || int || Index of this completion (for n > 1)
|-
| text || str || Generated text
|-
| token_ids || list[int] || Generated token IDs
|-
| cumulative_logprob || float || Sum of log probabilities
|-
| logprobs || dict || Token-level log probabilities (if requested)
|-
| finish_reason || str || Reason for completion ("stop", "length", etc.)
|}

== Usage Examples ==

=== Basic Generation ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50
)

outputs = llm.generate(prompts, sampling_params=sampling_params)

# Process outputs
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated!r}\n")
</syntaxhighlight>

=== Single Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Single string prompt
prompt = "Explain quantum computing in simple terms:"

outputs = llm.generate(
    prompt,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=200)
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Different Sampling Per Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = [
    "Write a haiku about spring:",
    "Explain machine learning:"
]

# Different sampling params for each prompt
sampling_params = [
    SamplingParams(temperature=1.0, max_tokens=30),   # Creative for haiku
    SamplingParams(temperature=0.3, max_tokens=150),  # Factual for explanation
]

outputs = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}\n")
</syntaxhighlight>

=== With Progress Bar ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = ["Prompt {}".format(i) for i in range(100)]

# use_tqdm=True shows progress bar (default)
outputs = llm.generate(
    prompts,
    sampling_params=SamplingParams(max_tokens=50),
    use_tqdm=True
)

print(f"Generated {len(outputs)} completions")
</syntaxhighlight>

=== Accessing Detailed Output ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

outputs = llm.generate(
    "Tell me a joke",
    sampling_params=SamplingParams(
        max_tokens=100,
        logprobs=1  # Request log probabilities
    )
)

output = outputs[0]
completion = output.outputs[0]

print(f"Generated text: {completion.text}")
print(f"Token IDs: {completion.token_ids}")
print(f"Finish reason: {completion.finish_reason}")
print(f"Cumulative logprob: {completion.cumulative_logprob}")
if completion.logprobs:
    print(f"First token logprobs: {completion.logprobs[0]}")
</syntaxhighlight>

=== Data Parallel Execution ===
<syntaxhighlight lang="python">
import os
from vllm import LLM, SamplingParams

def worker_main(rank, size, all_prompts):
    """Worker process for data parallel inference."""
    # Setup DP environment
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(size)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = "29500"

    # Partition prompts
    my_prompts = partition_data(all_prompts, rank, size)

    # Initialize LLM
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Generate on this worker's partition
    # No communication with other workers during generation
    outputs = llm.generate(
        my_prompts,
        sampling_params=SamplingParams(max_tokens=100)
    )

    print(f"Worker {rank}: Generated {len(outputs)} completions")
    return outputs
</syntaxhighlight>

=== With Multi-Modal Input ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Multi-modal prompt with image
prompt = {
    "prompt": "Describe this image:",
    "multi_modal_data": {
        "image": "path/to/image.jpg"
    }
}

outputs = llm.generate(
    prompt,
    sampling_params=SamplingParams(max_tokens=200)
)

print(outputs[0].outputs[0].text)
</syntaxhighlight>

=== Multiple Completions Per Prompt ===
<syntaxhighlight lang="python">
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Request multiple completions using beam search
sampling_params = SamplingParams(
    n=3,  # Generate 3 completions
    best_of=5,  # Sample 5 and return best 3
    use_beam_search=True,
    temperature=0
)

outputs = llm.generate("Once upon a time", sampling_params=sampling_params)

# Access multiple completions
for i, completion in enumerate(outputs[0].outputs):
    print(f"Completion {i}: {completion.text}")
</syntaxhighlight>

== Implementation Details ==

=== Execution Flow ===
<syntaxhighlight lang="python">
def generate(self, prompts, sampling_params=None, ...):
    # 1. Validate runner type
    if runner_type != "generate":
        raise ValueError("Only for generative models")

    # 2. Apply default sampling params if needed
    if sampling_params is None:
        sampling_params = self.get_default_sampling_params()

    # 3. Validate and add requests to engine
    self._validate_and_add_requests(
        prompts=prompts,
        params=sampling_params,
        use_tqdm=use_tqdm,
        lora_request=lora_request,
        priority=priority,
    )

    # 4. Run engine until all requests complete
    outputs = self._run_engine(use_tqdm=use_tqdm)

    # 5. Validate and return outputs
    return self.engine_class.validate_outputs(outputs, RequestOutput)
</syntaxhighlight>

=== Blocking Behavior ===
<code>generate()</code> is synchronous and blocks until completion:
* Submits all prompts to engine's scheduler
* Runs engine step-by-step
* Waits for all requests to finish
* Returns complete results

This is suitable for offline batch inference but not for online serving (use AsyncLLMEngine for that).

=== Progress Tracking ===
When <code>use_tqdm=True</code>:
* Displays progress bar showing completion percentage
* Updates as requests finish
* Useful for long-running batch jobs

=== Error Handling ===
The method validates:
* Model runner type matches "generate"
* Prompt formats are valid
* Sampling parameters are valid
* Raises appropriate exceptions for invalid inputs

== Performance Characteristics ==

=== Throughput ===
* Processes multiple prompts in parallel through batching
* Throughput scales with batch size up to memory limits
* Optimal batch size depends on prompt/output lengths

=== Latency ===
* Synchronous call blocks until all prompts complete
* Total time determined by longest prompt in batch
* First token latency + generation time

=== Memory ===
* Memory usage proportional to:
  * Batch size (number of prompts)
  * Sequence lengths (prompt + generation)
  * KV cache size

== Best Practices ==

# '''Batch Similar Prompts''': Group prompts with similar lengths for better efficiency
# '''Set Reasonable max_tokens''': Avoid unnecessarily long generations
# '''Use Appropriate Temperature''': Lower for factual, higher for creative
# '''Monitor Progress''': Enable tqdm for long jobs
# '''Handle Outputs''': Check finish_reason to detect truncation

== Common Issues ==

=== OOM During Generation ===
* Reduce batch size (fewer prompts at once)
* Reduce max_tokens
* Reduce max_model_len in LLM initialization
* Increase gpu_memory_utilization carefully

=== Slow Generation ===
* Check if CUDA graphs are enabled (disable enforce_eager)
* Verify GPU utilization is high
* Reduce max_num_seqs if scheduling overhead is high
* Check for stragglers in DP mode

=== Unexpected Outputs ===
* Verify sampling parameters (temperature, top_p)
* Check finish_reason for early termination
* Validate prompt format
* Ensure model supports the task

== Related Pages ==
* [[implements::Principle:vllm-project_vllm_LLM_generate_dp]] - Parallel execution principle
* [[related_to::vllm-project_vllm_LLM_class]] - LLM class
* [[related_to::SamplingParams]] - Sampling configuration
* [[related_to::RequestOutput]] - Output structure
* [[related_to::LLMEngine]] - Underlying engine

== See Also ==
* vllm/entrypoints/llm.py - Full implementation
* vllm/sampling_params.py - SamplingParams documentation
* vllm/outputs.py - Output structures
