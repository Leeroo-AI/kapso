# Principle: Result_Collection

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

JSONL file parsing for collecting multi-rollout inference results from agent evaluation runs.

=== Description ===

Result Collection is the foundational data ingestion step in the DeepResearch evaluation pipeline. It handles the parsing of JSONL (JSON Lines) formatted files that contain the complete inference trajectories from agent runs.

Each JSONL file represents one "round" or "rollout" of inference across all evaluation questions. The multi-rollout evaluation design requires collecting results from multiple independent runs (typically 3 rounds) to compute robust metrics like Pass@k.

The collection process extracts key fields from each inference record:
- **question** - The original query posed to the agent
- **answer** - The ground truth answer for evaluation
- **prediction** - The agent's final response
- **messages** - Complete conversation trajectory for behavioral analysis
- **termination** - How the agent concluded (answered, max_turns, max_tokens)

=== Usage ===

Use Result Collection when:
- Loading agent inference outputs for evaluation
- Preparing data for LLM judge scoring
- Gathering trajectories for behavioral statistics computation
- Aggregating results from multiple evaluation rounds

This principle is the entry point for all downstream evaluation processing.

== Theoretical Basis ==

The Result Collection step transforms raw JSONL files into structured data suitable for evaluation.

'''JSONL File Format:'''
<syntaxhighlight lang="json">
{"question": "...", "answer": "...", "prediction": "...", "messages": [...], "termination": "..."}
{"question": "...", "answer": "...", "prediction": "...", "messages": [...], "termination": "..."}
...
</syntaxhighlight>

'''Processing Flow:'''
<syntaxhighlight lang="text">
Input: JSONL file path (e.g., iter1.jsonl)

For each line in file:
  1. Parse JSON object
  2. Extract prediction item with all fields
  3. Append to results list

Output: List[Dict] - All parsed items for one round
</syntaxhighlight>

'''Multi-Round Structure:'''
{| class="wikitable"
|-
! File !! Round !! Purpose
|-
| iter1.jsonl || Round 1 || First independent agent run
|-
| iter2.jsonl || Round 2 || Second independent agent run
|-
| iter3.jsonl || Round 3 || Third independent agent run
|}

The separation into independent rounds enables:
- Pass@k metric calculation (success with k attempts)
- Variance analysis across runs
- Consensus-based evaluation

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Process_Single_Round]]
