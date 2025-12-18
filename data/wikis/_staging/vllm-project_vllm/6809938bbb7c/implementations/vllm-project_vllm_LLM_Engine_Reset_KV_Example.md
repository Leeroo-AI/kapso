{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::KV Cache]], [[domain::Engine API]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates how to reset the prefix cache during request processing using the LLMEngine API.

=== Description ===
The LLM Engine Reset KV example shows how to use the lower-level LLMEngine API for fine-grained control over request processing, including the ability to reset the prefix cache mid-stream. The example processes requests in a step-by-step fashion, demonstrating how to manually control the execution loop. At step 10, it resets the prefix cache while maintaining running requests, which can be useful for memory management or changing system behavior during long-running inference sessions. This pattern provides maximum control over vLLM's execution compared to the high-level LLM class.

=== Usage ===
Use this pattern when you need fine-grained control over request processing, want to implement custom scheduling logic, need to reset or manage KV cache during execution, or want to build a custom serving system on top of vLLM's engine.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/llm_engine_reset_kv.py examples/offline_inference/llm_engine_reset_kv.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/llm_engine_reset_kv.py [ENGINE_ARGS]
</syntaxhighlight>

== Key Concepts ==

=== LLMEngine Step-by-Step Execution ===
The LLMEngine provides manual control over inference:
* '''add_request()''': Queue new requests for processing
* '''step()''': Execute one scheduler step and return completed outputs
* '''has_unfinished_requests()''': Check if work remains
* Each step processes a batch of requests based on available resources

=== Prefix Cache Reset ===
The reset_prefix_cache() method provides cache management:
* '''reset_running_requests''': True - reset even for active requests
* Clears cached prompt prefixes to free memory
* Can be called mid-execution without stopping the engine
* Useful for memory pressure situations or behavioral changes

=== Request Output Handling ===
The step() method returns a list of RequestOutput objects:
* '''finished''': Boolean indicating completion
* '''prompt''': Original input prompt
* '''outputs''': List of generated completion(s)
* Only finished outputs are typically printed/stored

== Usage Examples ==

=== Engine Initialization ===
<syntaxhighlight lang="python">
from vllm import EngineArgs, LLMEngine
from vllm.utils.argparse_utils import FlexibleArgumentParser

def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()

def initialize_engine(args):
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

args = parse_args()
engine = initialize_engine(args)
</syntaxhighlight>

=== Step-by-Step Processing ===
<syntaxhighlight lang="python">
from vllm import SamplingParams, RequestOutput

test_prompts = [
    (
        "A robot may not injure a human being " * 50,
        SamplingParams(
            temperature=0.0,
            logprobs=1,
            prompt_logprobs=1,
            max_tokens=16
        ),
    ),
    (
        "What is the meaning of life?",
        SamplingParams(
            n=2,
            temperature=0.8,
            top_p=0.95,
            max_tokens=128
        ),
    ),
]

request_id = 0
step_id = 0

while test_prompts or engine.has_unfinished_requests():
    print(f"Step {step_id}")

    # Add new request if available
    if test_prompts:
        prompt, sampling_params = test_prompts.pop(0)
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1

    # Reset prefix cache at specific step
    if step_id == 10:
        print(f"Resetting prefix cache at step {step_id}")
        engine.reset_prefix_cache(reset_running_requests=True)

    # Execute one step
    request_outputs: list[RequestOutput] = engine.step()

    # Process completed requests
    for request_output in request_outputs:
        if request_output.finished:
            print("-" * 50)
            print(request_output)
            print("-" * 50)

    step_id += 1
</syntaxhighlight>

=== Custom Scheduling Logic ===
<syntaxhighlight lang="python">
# Example: Rate limiting
import time

while engine.has_unfinished_requests():
    start = time.time()

    request_outputs = engine.step()

    # Process outputs
    for output in request_outputs:
        if output.finished:
            handle_completed_request(output)

    # Rate limiting: max 10 steps per second
    elapsed = time.time() - start
    if elapsed < 0.1:
        time.sleep(0.1 - elapsed)
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
