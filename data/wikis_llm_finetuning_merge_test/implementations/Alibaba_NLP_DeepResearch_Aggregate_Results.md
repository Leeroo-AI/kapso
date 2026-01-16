# Implementation: Aggregate_Results

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|evaluate_deepsearch_official.py|evaluation/evaluate_deepsearch_official.py]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Metrics]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for combining multi-round evaluation results into a query-centric aggregated structure.

=== Description ===

The `aggregate_results()` function combines LLM judge results from three independent evaluation rounds into a unified dictionary keyed by question. This aggregated structure enables efficient computation of Pass@k metrics.

The function:
1. Iterates through results from all three rounds
2. Groups judgements by question text
3. Normalizes judgement values using `is_correct_judgement()`
4. Preserves ground truth answers for reference

The resulting structure provides O(1) access to per-question, per-round correctness for metric calculation.

=== Usage ===

Use `aggregate_results()` when:
- Preparing data for Pass@k metric computation
- Analyzing per-question consistency across rounds
- Generating detailed evaluation reports
- Identifying questions with variable success rates

This function bridges individual round judgements to aggregate benchmark metrics.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' evaluation/evaluate_deepsearch_official.py
* '''Lines:''' 382-402

=== Signature ===
<syntaxhighlight lang="python">
def aggregate_results(
    round1_results: List[Dict],
    round2_results: List[Dict],
    round3_results: List[Dict]
) -> Dict:
    """
    Aggregate judgement results from multiple rounds.

    Args:
        round1_results: List of judgement dicts from round 1
        round2_results: List of judgement dicts from round 2
        round3_results: List of judgement dicts from round 3

    Returns:
        Dict mapping questions to per-round correctness:
        {
            "question_text": {
                "round1": "Correct" | "Incorrect" | "Error",
                "round2": "Correct" | "Incorrect" | "Error",
                "round3": "Correct" | "Incorrect" | "Error",
                "answer": "ground_truth"
            },
            ...
        }
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import aggregate_results
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| round1_results || List[Dict] || Yes || Judgements from round 1 evaluation
|-
| round2_results || List[Dict] || Yes || Judgements from round 2 evaluation
|-
| round3_results || List[Dict] || Yes || Judgements from round 3 evaluation
|-
| results[i]["question"] || str || Yes || Question text used as aggregation key
|-
| results[i]["answer"] || str || Yes || Ground truth answer
|-
| results[i]["judgement"] || str || Yes || LLM judge verdict
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| query_results || Dict || Aggregated results keyed by question
|-
| query_results[q]["round1"] || str or None || Round 1 judgement ("Correct", "Incorrect", etc.)
|-
| query_results[q]["round2"] || str or None || Round 2 judgement
|-
| query_results[q]["round3"] || str or None || Round 3 judgement
|-
| query_results[q]["answer"] || str || Ground truth answer
|}

== Usage Examples ==

=== Basic Aggregation ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import aggregate_results

# After running LLM judge on all rounds
round1_results = [{"question": "Q1", "answer": "A1", "judgement": "Correct"}, ...]
round2_results = [{"question": "Q1", "answer": "A1", "judgement": "Incorrect"}, ...]
round3_results = [{"question": "Q1", "answer": "A1", "judgement": "Correct"}, ...]

query_results = aggregate_results(round1_results, round2_results, round3_results)

# Access per-question results
for question, results in query_results.items():
    print(f"Q: {question[:50]}...")
    print(f"  Round 1: {results['round1']}")
    print(f"  Round 2: {results['round2']}")
    print(f"  Round 3: {results['round3']}")
</syntaxhighlight>

=== Computing Pass@3 from Aggregated Results ===
<syntaxhighlight lang="python">
query_results = aggregate_results(round1_results, round2_results, round3_results)

# Count questions with at least one correct answer
pass_at_3_count = 0
for question, results in query_results.items():
    rounds = [results["round1"], results["round2"], results["round3"]]
    if "Correct" in rounds:
        pass_at_3_count += 1

pass_at_3 = pass_at_3_count / len(query_results) * 100
print(f"Pass@3: {pass_at_3:.2f}%")
</syntaxhighlight>

=== Analyzing Consistency ===
<syntaxhighlight lang="python">
query_results = aggregate_results(round1_results, round2_results, round3_results)

# Categorize questions by consistency
always_correct = []
sometimes_correct = []
never_correct = []

for question, results in query_results.items():
    rounds = [results["round1"], results["round2"], results["round3"]]
    correct_count = sum(1 for r in rounds if r == "Correct")

    if correct_count == 3:
        always_correct.append(question)
    elif correct_count > 0:
        sometimes_correct.append(question)
    else:
        never_correct.append(question)

print(f"Always correct: {len(always_correct)}")
print(f"Sometimes correct: {len(sometimes_correct)}")
print(f"Never correct: {len(never_correct)}")
</syntaxhighlight>

=== Integration with Full Pipeline ===
<syntaxhighlight lang="python">
# Full evaluation pipeline
round_results = {}
for round_name in ["round1", "round2", "round3"]:
    # Run LLM judge in parallel...
    round_results[round_name] = results

# Aggregate
query_results = aggregate_results(
    round_results["round1"],
    round_results["round2"],
    round_results["round3"]
)

# Compute all metrics
pass_at_3 = calculate_pass_at_k(query_results, k=3)
best_pass_at_1 = calculate_best_pass_at_1(query_results)
avg_pass_at_3 = calculate_avg_pass_at_3(query_results)
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_Result_Aggregation]]
