# Implementation: WebSailor_Evaluate

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
Evaluation module for WebSailor agent that scores predictions using LLM-based CoT-QA judges with stratified reporting by question difficulty and source type.

=== Description ===
The `evaluate.py` module in WebSailor provides evaluation infrastructure for benchmarking the vLLM-powered ReAct agent. It implements:

- LangChain CoT-QA evaluator integration for semantic answer matching
- Parallel evaluation with ThreadPoolExecutor and exponential backoff retry
- Stratified metrics by difficulty level (easy/medium/hard) and source type (single/multi)
- JSON report generation with comprehensive accuracy breakdowns
- Resume capability via processed question tracking

The evaluation handles JSONL prediction files and produces detailed reports suitable for academic benchmarking.

=== Usage ===
Use this evaluation script after running WebSailor agent inference on benchmarks like GAIA, SimpleQA, or BrowseComp. Essential for model comparison and ablation studies.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebSailor/src/evaluate.py WebAgent/WebSailor/src/evaluate.py]
* '''Lines:''' 1-329

=== Signature ===
<syntaxhighlight lang="python">
def eval_result(
    input_path: str,
    output_path: str,
    max_workers: int = 16
) -> None:
    """
    Evaluate predictions against reference answers.

    Args:
        input_path: Path to JSONL predictions file
        output_path: Path to save evaluation results
        max_workers: Number of parallel evaluation threads
    """
    ...

def safe_average(scores: List[float]) -> Optional[float]:
    """
    Compute average avoiding division by zero.

    Args:
        scores: List of numeric scores

    Returns:
        Average or None if empty
    """
    ...

def call_evaluator(
    data: Dict,
    evaluator: Any,
    max_retries: int = 10
) -> Dict:
    """
    Call LangChain evaluator with retry logic.

    Args:
        data: Dict with 'prediction', 'question', 'answer'
        evaluator: LangChain evaluator instance
        max_retries: Max retry attempts

    Returns:
        Evaluation result dict with 'score' key
    """
    ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.evaluate import (
    eval_result,
    safe_average,
    call_evaluator
)
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_path || str || Yes || Path to JSONL file with predictions
|-
| output_path || str || Yes || Path for evaluation output JSONL
|-
| max_workers || int || No || Parallel threads (default 16)
|-
| info_adic || Dict || No || Question metadata for stratification
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| evaluation.jsonl || File || Per-question scores and metadata
|-
| report.json || File || Aggregated metrics by category
|-
| current_accuracy || float || Running accuracy printed during eval
|}

== Usage Examples ==

=== Basic Evaluation ===
<syntaxhighlight lang="python">
from WebAgent.WebSailor.src.evaluate import eval_result

# Run evaluation on predictions
eval_result(
    input_path="output/websailor_predictions.jsonl",
    output_path="output/websailor_eval.jsonl",
    max_workers=16
)

# Results are written incrementally to output_path
# Report is auto-generated at output_path.replace('.jsonl', '_report.json')
</syntaxhighlight>

=== Command-Line Usage ===
<syntaxhighlight lang="bash">
# Run evaluation from terminal
python WebAgent/WebSailor/src/evaluate.py \
    --input_path output/gaia_predictions.jsonl \
    --output_path output/gaia_eval.jsonl

# Check generated report
cat output/gaia_eval_report.json
</syntaxhighlight>

=== Accessing Stratified Results ===
<syntaxhighlight lang="python">
import json

# After running eval_result()
with open("output/gaia_eval_report.json") as f:
    report = json.load(f)

print(f"Overall: {report['overall']:.2%}")
print(f"Single-source Easy: {report['single_source_easy']:.2%}")
print(f"Single-source Medium: {report['single_source_medium']:.2%}")
print(f"Single-source Hard: {report['single_source_hard']:.2%}")
print(f"Multi-source Easy: {report['multi_source_easy']:.2%}")
print(f"Multi-source Medium: {report['multi_source_medium']:.2%}")
print(f"Multi-source Hard: {report['multi_source_hard']:.2%}")
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_LLM_Judge_Scoring]]
* [[implements::Principle:Alibaba_NLP_DeepResearch_Behavioral_Statistics]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
