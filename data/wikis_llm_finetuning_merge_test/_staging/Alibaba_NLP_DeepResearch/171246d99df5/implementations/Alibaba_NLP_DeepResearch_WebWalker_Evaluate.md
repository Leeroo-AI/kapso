# Implementation: WebWalker_Evaluate

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Doc|WebWalkerQA Dataset|https://huggingface.co/datasets/callanwu/WebWalkerQA]]
|-
! Domains
| [[domain::Evaluation]], [[domain::Benchmark]], [[domain::LangChain]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Evaluation script for WebWalker on the WebWalkerQA benchmark with CoT-QA scoring and stratified reporting.

=== Description ===
The `evaluate.py` module evaluates WebWalker predictions on WebWalkerQA:

- Loads WebWalkerQA dataset from HuggingFace
- Uses LangChain `cot_qa` evaluator for semantic scoring
- Parallel evaluation with ThreadPoolExecutor (16 workers)
- Exponential backoff retry on failures
- Stratified reporting by difficulty and source type

Generates detailed JSON reports with per-category accuracy.

=== Usage ===
Run after WebWalker inference to evaluate predictions against reference answers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebWalker/src/evaluate.py WebAgent/WebWalker/src/evaluate.py]
* '''Lines:''' 1-156

=== Signature ===
<syntaxhighlight lang="python">
def eval_result(input_path: str, output_path: str) -> None:
    """
    Evaluate predictions against WebWalkerQA.

    Args:
        input_path: Path to predictions JSONL
        output_path: Path for evaluation output

    Generates:
        - output_path: Per-question scores
        - output_path.replace('.jsonl', '_report.json'): Stratified report
    """
    ...

def call(data: Dict) -> Dict:
    """Evaluate single prediction with retry."""
    ...

def safe_average(scores: List[float]) -> Optional[float]:
    """Average with zero-division handling."""
    ...

# Dataset loading
ds = load_dataset("callanwu/WebWalkerQA", split="main")
info_adic = {q: [a, i] for q, a, i in zip(ds["question"], ds["answer"], ds["info"])}
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from WebAgent.WebWalker.src.evaluate import eval_result
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| input_path || str || Yes || Predictions JSONL path
|-
| output_path || str || Yes || Evaluation output path
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| scores || JSONL || Per-question evaluation
|-
| report || JSON || Stratified accuracy report
|}

== Usage Examples ==

=== Command-Line Evaluation ===
<syntaxhighlight lang="bash">
python WebAgent/WebWalker/src/evaluate.py \
    --input_path results/predictions.jsonl \
    --output_path results/evaluation.jsonl
</syntaxhighlight>

=== Report Structure ===
<syntaxhighlight lang="json">
{
    "single_source_easy": 0.85,
    "single_source_medium": 0.72,
    "single_source_hard": 0.58,
    "multi_source_easy": 0.78,
    "multi_source_medium": 0.65,
    "multi_source_hard": 0.45,
    "overall": 0.67
}
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_LLM_Judge_Scoring]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_API_Keys_Configuration]]
