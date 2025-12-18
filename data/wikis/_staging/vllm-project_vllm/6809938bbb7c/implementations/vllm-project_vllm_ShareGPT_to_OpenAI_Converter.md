{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|vllm-project_vllm|https://github.com/vllm-project/vllm]]
|-
! Domains
| [[domain::Data Processing]], [[domain::Dataset Conversion]], [[domain::Multi-turn Conversations]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==
A data conversion tool that transforms ShareGPT conversation datasets into OpenAI API-compatible format for multi-turn benchmarking.

=== Description ===
This script converts ShareGPT conversation datasets (raw multi-turn human-AI dialogs) into OpenAI API message format suitable for vLLM benchmarking. It merges conversation fragments with the same ID, validates turn alternation between user and assistant, filters based on content length and turn count, and generates comprehensive statistics including token counts when a tokenizer is provided. The tool supports random sampling, handles non-English filtering, and can limit dataset size. It's essential for preparing multi-turn conversation benchmarks from ShareGPT sources.

=== Usage ===
Use this tool when preparing ShareGPT datasets for multi-turn inference benchmarks, creating filtered conversation datasets with specific characteristics, or converting between conversation formats for compatibility with vLLM's multi-turn APIs.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/vllm-project/vllm vllm-project_vllm]
* '''File:''' [https://github.com/vllm-project/vllm/blob/main/benchmarks/multi_turn/convert_sharegpt_to_openai.py benchmarks/multi_turn/convert_sharegpt_to_openai.py]

=== Signature ===
<syntaxhighlight lang="python">
def convert_sharegpt_to_openai(
    seed: int,
    input_file: str,
    output_file: str,
    max_items: int | None,
    min_content_len: int | None = None,
    max_content_len: int | None = None,
    min_turns: int | None = None,
    max_turns: int | None = None,
    model: str | None = None
) -> None

def print_stats(conversations: list[dict], tokenizer: AutoTokenizer | None = None) -> None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Download ShareGPT dataset first
# From: https://huggingface.co/datasets/philschmid/sharegpt-raw

export INPUT_FILE=sharegpt_20230401_clean_lang_split.json
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    $INPUT_FILE sharegpt_conv_128.json --max-items=128
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| input_file || str || Path to ShareGPT JSON file
|-
| output_file || str || Path for output OpenAI format JSON
|-
| --seed || int || Random seed for shuffling/sampling (default 0)
|-
| --max-items || int || Maximum conversations in output (None = all)
|-
| --min-content-len || int || Minimum message character length
|-
| --max-content-len || int || Maximum message character length
|-
| --min-turns || int || Minimum conversation turns
|-
| --max-turns || int || Maximum conversation turns
|-
| --model || str || HF model for tokenization statistics (optional)
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| output_json || File || OpenAI-format conversations with "id" and "messages" keys
|-
| statistics || stdout || Pandas DataFrame with turn/word/token counts and percentiles
|-
| conversation_count || int || Number of conversations after filtering
|}

== Usage Examples ==

<syntaxhighlight lang="python">
# Basic conversion with sampling
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    sharegpt_raw.json \
    sharegpt_openai.json \
    --max-items=1000 \
    --seed=42

# Filter by conversation characteristics
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    sharegpt_raw.json \
    filtered_conversations.json \
    --min-turns=3 \
    --max-turns=10 \
    --min-content-len=50 \
    --max-content-len=1000 \
    --max-items=500

# Include tokenization statistics
python benchmarks/multi_turn/convert_sharegpt_to_openai.py \
    sharegpt_raw.json \
    sharegpt_with_stats.json \
    --model=meta-llama/Llama-2-7b-hf \
    --max-items=200

# Output format
# [
#   {
#     "id": "conversation_id_123",
#     "messages": [
#       {"role": "user", "content": "Hello, how are you?"},
#       {"role": "assistant", "content": "I'm doing well, thank you!"},
#       {"role": "user", "content": "What can you help me with?"}
#     ]
#   },
#   ...
# ]
</syntaxhighlight>

== Related Pages ==
* [[Benchmark:Multi_Turn_Inference]]
* [[Dataset:ShareGPT]]
* [[Tool:Dataset_Conversion]]
* [[Concept:OpenAI_API_Format]]
