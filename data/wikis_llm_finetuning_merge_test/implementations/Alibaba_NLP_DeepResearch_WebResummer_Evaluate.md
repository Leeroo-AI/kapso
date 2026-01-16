# Implementation: WebResummer_Evaluate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Metrics]], [[domain::NLP_Benchmarks]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Evaluation pipeline for WebResummer agent that computes accuracy metrics across multiple reasoning trajectories using LLM-based judging with support for pass@k scoring.

=== Description ===
The `evaluate.py` module provides a comprehensive evaluation framework for the WebResummer agent on web research benchmarks. It processes multiple rollout trajectories, extracts predicted answers, and uses LLM-based judges (CoT-QA evaluator from LangChain) to score predictions against reference answers. The system supports:

- Multi-rollout evaluation with pass@k metric computation
- Parallel evaluation using ThreadPoolExecutor for efficiency
- Difficulty stratification (easy/medium/hard) for granular analysis
- Single-source vs multi-source question categorization
- Automatic report generation with aggregated statistics

=== Usage ===
Run this evaluation script when benchmarking WebResummer agent performance on datasets like GAIA, BrowseComp, or WebWalkerQA. Use for comparing model variants or validating ReSum improvements.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebResummer/src/evaluate.py WebAgent/WebResummer/src/evaluate.py]
* '''Lines:''' 1-309

=== Signature ===
<syntaxhighlight lang="python">
def eval_result(
    input_path: str,
    output_path: str,
    evaluator_type: str = "cot_qa"
) -> None:
    """
    Evaluate agent predictions against reference answers.

    Args:
        input_path: Path to JSONL file with predictions
        output_path: Path to save evaluation results
        evaluator_type: Type of LangChain evaluator (default "cot_qa")
    """
    ...

def calculate_pass_at_k(
    scores: List[float],
    k: int
) -> float:
    """
    Calculate pass@k metric from scores.

    Args:
        scores: List of binary scores (0 or 1)
        k: Number of attempts

    Returns:
        Pass@k probability
    """
    ...

def generate_report(
    output_path: str,
    info_dict: Dict
) -> Dict:
    """
    Generate stratified evaluation report.

    Args:
        output_path: Path to evaluation results
        info_dict: Question metadata dictionary

    Returns:
        Report dict with metrics by difficulty/source type
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.evaluate import (
    eval_result,
    calculate_pass_at_k,
    generate_report
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_path || str || Yes || Path to predictions JSONL file
|-
| output_path || str || Yes || Path for evaluation output
|-
| evaluator_type || str || No || LangChain evaluator type (default "cot_qa")
|-
| info_dict || Dict || No || Question metadata for stratification
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| scores || List[float] || Per-question evaluation scores
|-
| report || Dict || Aggregated metrics by category
|-
| pass_at_k || float || Pass@k probability metric
|}

== Usage Examples ==

=== Basic Evaluation ===
<syntaxhighlight lang="python">
from WebAgent.WebResummer.src.evaluate import eval_result

# Evaluate predictions from a single rollout
eval_result(
    input_path="results/gaia_rollout1.jsonl",
    output_path="results/gaia_eval.jsonl"
)
</syntaxhighlight>

=== Multi-Rollout Evaluation with Report ===
<syntaxhighlight lang="python">
import json
from WebAgent.WebResummer.src.evaluate import (
    eval_result,
    generate_report,
    calculate_pass_at_k
)

# Evaluate multiple rollouts
for i in range(1, 4):
    eval_result(
        input_path=f"results/iter{i}.jsonl",
        output_path=f"results/eval_iter{i}.jsonl"
    )

# Load info dictionary for stratification
info_dict = {}
with open("data/gaia_metadata.json") as f:
    for item in json.load(f):
        info_dict[item["question"]] = item

# Generate stratified report
report = generate_report(
    output_path="results/eval_iter1.jsonl",
    info_dict=info_dict
)

print(f"Overall Accuracy: {report['overall']:.2%}")
print(f"Single-source Easy: {report['single_source_easy']:.2%}")
print(f"Multi-source Hard: {report['multi_source_hard']:.2%}")

# Calculate pass@3 across rollouts
all_scores = []
for i in range(1, 4):
    with open(f"results/eval_iter{i}.jsonl") as f:
        for line in f:
            all_scores.append(json.loads(line)["score"])

pass_at_3 = calculate_pass_at_k(all_scores, k=3)
print(f"Pass@3: {pass_at_3:.2%}")
</syntaxhighlight>

=== Command-Line Evaluation ===
<syntaxhighlight lang="bash">
# Run evaluation from command line
python WebAgent/WebResummer/src/evaluate.py \
    --input_path results/predictions.jsonl \
    --output_path results/evaluation.jsonl
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_LLM_Judge_Scoring]]
* [[implements::Principle:Alibaba_NLP_DeepResearch_Pass_At_K_Metrics]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
