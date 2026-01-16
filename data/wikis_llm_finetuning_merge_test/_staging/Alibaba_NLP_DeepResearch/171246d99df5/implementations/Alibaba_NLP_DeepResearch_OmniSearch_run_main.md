# Implementation: OmniSearch_run_main

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba-NLP/DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|agent_eval.py|WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py]]
|-
! Domains
| [[domain::Multimodal]], [[domain::Agent_Systems]], [[domain::Vision_Language]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Main inference method that implements the multi-turn agent loop for vision-language question answering in the OmniSearch class.

=== Description ===

The `run_main()` method orchestrates the complete multimodal agent pipeline. It takes a sample containing an image path and question, processes the image, constructs the conversation, and iteratively calls the vision-language model while routing tool calls until a final answer is produced.

The method handles:
- Image preprocessing and base64 encoding
- System prompt construction with tool definitions
- Multi-turn conversation management
- Tool call parsing and execution routing
- Step limit enforcement (12 steps max)
- Answer extraction from model responses

=== Usage ===

Use `run_main()` when:
- Running single-sample multimodal inference
- Implementing custom evaluation pipelines
- Debugging agent behavior on specific samples

For batch processing with parallelization, use `eval()` instead.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' WebAgent/WebWatcher/infer/scripts_eval/agent_eval.py
* '''Lines:''' 182-401

=== Signature ===
<syntaxhighlight lang="python">
def run_main(self, sample: Dict) -> Tuple[str, List[Dict], str]:
    """
    Run the multimodal agent loop on a single sample.

    Args:
        sample: Dict - Input sample with keys:
            - "file_path": str - Path to input image
            - "prompt": str - Question about the image

    Returns:
        Tuple[str, List[Dict], str]:
            - status: str - "success", "max_steps", or "error"
            - messages: List[Dict] - Full conversation history
            - answer_content: str - Extracted final answer

    Raises:
        Exception: Propagates errors from tool execution or API calls
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

agent = OmniSearch(model_name="qwen-vl-plus")

sample = {"file_path": "/path/to/image.jpg", "prompt": "What is this?"}
status, messages, answer = agent.run_main(sample)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| sample || Dict || Yes || Sample dictionary with "file_path" and "prompt" keys
|-
| sample["file_path"] || str || Yes || Path to input image file
|-
| sample["prompt"] || str || Yes || Question about the image
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| status || str || Termination status: "success" (answer found), "max_steps" (limit reached), or "error"
|-
| messages || List[Dict] || Complete conversation history with all turns
|-
| answer_content || str || Extracted answer text from model response
|}

== Usage Examples ==

=== Basic Single-Sample Inference ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

agent = OmniSearch(
    model_name="qwen-vl-plus",
    max_steps=12
)

sample = {
    "file_path": "/data/images/bird.jpg",
    "prompt": "What species of bird is shown in this image?"
}

status, messages, answer = agent.run_main(sample)
print(f"Status: {status}")
print(f"Answer: {answer}")
print(f"Total turns: {len(messages)}")
</syntaxhighlight>

=== Analyzing Agent Behavior ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

agent = OmniSearch(model_name="qwen-vl-plus")

sample = {
    "file_path": "/data/images/landmark.jpg",
    "prompt": "Where was this photo taken and what is the historical significance?"
}

status, messages, answer = agent.run_main(sample)

# Analyze tool usage
for msg in messages:
    if msg["role"] == "assistant":
        content = msg["content"]
        if "<tool_call>" in content:
            print("Tool called:", content.split("name=")[1].split('"')[1])
        if "<think>" in content:
            print("Reasoning:", content.split("<think>")[1].split("</think>")[0][:100])
</syntaxhighlight>

=== With Error Handling ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch

agent = OmniSearch(model_name="qwen-vl-plus")

try:
    status, messages, answer = agent.run_main(sample)
    if status == "success":
        print(f"Answer: {answer}")
    elif status == "max_steps":
        print(f"Reached step limit. Best answer: {answer}")
    else:
        print(f"Error occurred: {status}")
except Exception as e:
    print(f"Inference failed: {e}")
</syntaxhighlight>

=== Integration with Evaluation Pipeline ===
<syntaxhighlight lang="python">
from scripts_eval.agent_eval import OmniSearch
import json

agent = OmniSearch(model_name="qwen-vl-plus")

# Load benchmark dataset
with open("/data/hle/test.json") as f:
    dataset = json.load(f)

results = []
for sample in dataset:
    status, messages, answer = agent.run_main(sample)
    results.append({
        "id": sample.get("id"),
        "status": status,
        "answer": answer,
        "num_steps": len([m for m in messages if m["role"] == "assistant"])
    })
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Multi_Turn_Agent_Loop]]
