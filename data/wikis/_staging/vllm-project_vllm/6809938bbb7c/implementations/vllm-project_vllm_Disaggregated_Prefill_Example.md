{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Examples]], [[domain::Disaggregated Architecture]], [[domain::KV Cache]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
This example demonstrates disaggregated prefill architecture where KV cache is transferred between separate prefill and decode instances.

=== Description ===
The disaggregated prefill example shows how to split prefill and decode phases across two separate vLLM instances running on different GPUs. The prefill node processes the initial prompt tokens and generates KV cache, which is then transferred to a decode node via P2P NCCL connector. This architecture optimizes resource utilization by allowing specialized hardware allocation for compute-intensive prefill (prompt processing) versus memory-intensive decode (token generation). The decode node can handle more prompts than it prefills by receiving KV cache from the prefill node.

=== Usage ===
Use disaggregated prefill when you want to optimize hardware utilization by separating prefill and decode workloads, need to scale prefill and decode independently, or want to use different GPU configurations for different phases (e.g., high-compute GPUs for prefill, high-memory GPUs for decode).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/disaggregated_prefill.py examples/offline_inference/disaggregated_prefill.py]

=== CLI Usage ===
<syntaxhighlight lang="bash">
python examples/offline_inference/disaggregated_prefill.py
</syntaxhighlight>

== Key Concepts ==

=== KV Transfer Configuration ===
Both nodes are configured with KVTransferConfig for communication:

'''Prefill Node (Producer):'''
* '''kv_connector''': "P2pNcclConnector" - NCCL-based P2P transfer
* '''kv_role''': "kv_producer" - generates and sends KV cache
* '''kv_rank''': 0 - rank in KV transfer group
* '''kv_parallel_size''': 2 - total number of instances

'''Decode Node (Consumer):'''
* '''kv_connector''': "P2pNcclConnector"
* '''kv_role''': "kv_consumer" - receives and uses KV cache
* '''kv_rank''': 1
* '''kv_parallel_size''': 2

=== KV Cache Transfer Flow ===
The example demonstrates the complete transfer pipeline:
1. Prefill node processes prompts with max_tokens=1
2. KV cache is generated and transmitted via NCCL
3. Decode node waits for prefill completion
4. Decode node receives KV cache via P2P transfer
5. Decode node performs full generation using transferred cache
6. Decode node can also prefill prompts not sent by prefill node

=== Multiprocessing Coordination ===
The example uses multiprocessing.Event for synchronization:
* Prefill process signals completion via Event
* Decode process waits for Event before starting generation
* Prefill process kept alive to maintain KV transfer connection

== Usage Examples ==

=== Prefill Node Setup ===
<syntaxhighlight lang="python">
import os
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def run_prefill(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]

    # Generate only 1 token to create KV cache
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
    )

    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
    )

    llm.generate(prompts, sampling_params)
    print("Prefill node is finished.")
    prefill_done.set()

    # Keep running to maintain connection
    while True:
        time.sleep(1)
</syntaxhighlight>

=== Decode Node Setup ===
<syntaxhighlight lang="python">
def run_decode(prefill_done):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",  # Decode will prefill this one
    ]

    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_parallel_size=2,
    )

    llm = LLM(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
    )

    # Wait for KV cache transfer
    prefill_done.wait()

    # Generate with transferred KV cache
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
</syntaxhighlight>

=== Main Process Coordination ===
<syntaxhighlight lang="python">
from multiprocessing import Event, Process

def main():
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done,))
    decode_process = Process(target=run_decode, args=(prefill_done,))

    prefill_process.start()
    decode_process.start()

    decode_process.join()
    prefill_process.terminate()
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:vllm-project_vllm_GPU_Environment]]
