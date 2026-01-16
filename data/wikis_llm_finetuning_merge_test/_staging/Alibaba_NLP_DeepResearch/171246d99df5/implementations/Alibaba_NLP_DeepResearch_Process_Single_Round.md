# Implementation: Process_Single_Round

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|evaluate_deepsearch_official.py|evaluation/evaluate_deepsearch_official.py]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Data_Processing]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for parsing JSONL inference result files into structured prediction items.

=== Description ===

The `process_single_round()` function loads and parses a JSONL file containing agent inference results for one evaluation round. Each line in the file represents a complete inference trajectory for one question.

This is a simple but critical data loading function that:
1. Opens the specified JSONL file with UTF-8 encoding
2. Parses each line as a JSON object
3. Returns a list of all parsed items

The function preserves all fields from the original JSONL records, enabling downstream processing for both evaluation (judgement) and analysis (behavioral statistics).

=== Usage ===

Use `process_single_round()` when:
- Loading inference results for evaluation
- Preparing data for LLM judge scoring
- Extracting trajectories for behavioral analysis
- Processing any single rollout result file

This function is called 3 times per evaluation run (once per round: iter1.jsonl, iter2.jsonl, iter3.jsonl).

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' evaluation/evaluate_deepsearch_official.py
* '''Lines:''' 147-151

=== Signature ===
<syntaxhighlight lang="python">
def process_single_round(input_file: str) -> List[Dict]:
    """
    Load and parse a JSONL file containing inference results.

    Args:
        input_file: str - Path to JSONL file (e.g., "results/iter1.jsonl")

    Returns:
        List[Dict] - List of parsed prediction items, one per line
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import process_single_round
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_file || str || Yes || Path to JSONL file containing inference results
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| items || List[Dict] || List of parsed inference items
|-
| items[i]["question"] || str || Original question posed to agent
|-
| items[i]["answer"] || str || Ground truth answer
|-
| items[i]["prediction"] || str || Agent's final response
|-
| items[i]["messages"] || List[Dict] || Complete message trajectory
|-
| items[i]["termination"] || str (optional) || How inference ended
|}

=== File Format ===
{| class="wikitable"
|-
! Field !! Type !! Description
|-
| question || str || The evaluation question
|-
| answer || str || Ground truth answer
|-
| prediction || str || Agent's extracted answer
|-
| messages || List[Dict] || Full conversation history
|-
| termination || str || "answered", "max_turns_reached", etc.
|}

== Usage Examples ==

=== Basic File Loading ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import process_single_round

# Load single round results
items = process_single_round("results/iter1.jsonl")
print(f"Loaded {len(items)} items")

# Access first item
first = items[0]
print(f"Question: {first['question']}")
print(f"Answer: {first['answer']}")
print(f"Prediction: {first['prediction']}")
</syntaxhighlight>

=== Loading All Three Rounds ===
<syntaxhighlight lang="python">
import os

input_folder = "results/browsecomp_en"

round_items = {
    "round1": process_single_round(os.path.join(input_folder, "iter1.jsonl")),
    "round2": process_single_round(os.path.join(input_folder, "iter2.jsonl")),
    "round3": process_single_round(os.path.join(input_folder, "iter3.jsonl"))
}

for round_name, items in round_items.items():
    print(f"{round_name}: {len(items)} items")
</syntaxhighlight>

=== Extracting Predictions for Evaluation ===
<syntaxhighlight lang="python">
items = process_single_round("results/iter1.jsonl")

# Prepare for LLM judge
for item in items:
    eval_item = {
        "question": item["question"],
        "answer": item["answer"],
        "prediction": item["prediction"]
    }
    # Pass to call_llm_judge()
</syntaxhighlight>

=== Analyzing Message Trajectories ===
<syntaxhighlight lang="python">
items = process_single_round("results/iter1.jsonl")

for item in items:
    messages = item.get("messages", [])
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    print(f"Question: {item['question'][:50]}...")
    print(f"  Total messages: {len(messages)}")
    print(f"  Assistant turns: {len(assistant_msgs)}")
    print(f"  Termination: {item.get('termination', 'unknown')}")
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Result_Collection]]
