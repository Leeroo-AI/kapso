# Implementation: Call_Llm_Judge

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::File|evaluate_deepsearch_official.py|evaluation/evaluate_deepsearch_official.py]]
|-
! Domains
| [[domain::Evaluation]], [[domain::LLM]], [[domain::Agent_Systems]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

Concrete tool for invoking LLM judge models to score agent predictions against ground truth answers.

=== Description ===

The `call_llm_judge()` function implements the LLM-as-Judge evaluation pattern for the DeepResearch benchmark. It takes a prediction item containing a question, ground truth answer, and agent prediction, then invokes an appropriate judge model to determine correctness.

Key features:
- **Multi-model support**: Routes to GPT-4o, Qwen2.5-72B, or Gemini based on dataset
- **Structured outputs**: Uses JSON schema enforcement for consistent parsing
- **Retry logic**: Implements up to 100 retries with exponential backoff
- **Error handling**: Returns error status on persistent failures

The function uses global variables (`judge_prompt`, `dataset`, `judge_model`) configured by the main evaluation script based on the target benchmark.

=== Usage ===

Use `call_llm_judge()` when:
- Evaluating a single prediction item
- Running parallel evaluation via ThreadPoolExecutor
- Implementing custom evaluation pipelines

This function is called once per prediction item, typically in parallel across many items.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba-NLP/DeepResearch]
* '''File:''' evaluation/evaluate_deepsearch_official.py
* '''Lines:''' 76-144

=== Signature ===
<syntaxhighlight lang="python">
def call_llm_judge(item: Dict) -> Dict:
    """
    Invoke LLM judge to evaluate a single prediction.

    Args:
        item: Dict containing:
            - question: str - The original query
            - answer: str - Ground truth answer
            - prediction: str - Agent's response

    Returns:
        Dict containing:
            - question: str - Original question
            - answer: str - Ground truth answer
            - judgement: str - 'Correct', 'Incorrect', or 'Error'
            - error: str (optional) - Error message if failed
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import call_llm_judge
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| item || Dict || Yes || Prediction item to evaluate
|-
| item["question"] || str || Yes || Original question posed to agent
|-
| item["answer"] || str || Yes || Ground truth correct answer
|-
| item["prediction"] || str || Yes || Agent's generated response
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| result || Dict || Judgement result dictionary
|-
| result["question"] || str || Echo of input question
|-
| result["answer"] || str || Echo of ground truth answer
|-
| result["judgement"] || str || "Correct", "Incorrect", or "Error"
|-
| result["error"] || str (optional) || Error message if judgement failed
|}

=== Global Dependencies ===
{| class="wikitable"
|-
! Variable !! Type !! Description
|-
| judge_prompt || str || Dataset-specific prompt template with {question}, {correct_answer}, {response} placeholders
|-
| dataset || str || Dataset name (e.g., "browsecomp_en", "gaia", "xbench-deepsearch")
|-
| judge_model || str || Model identifier (e.g., "gpt-4o-2024-08-06", "openai/qwen2.5-72b-instruct")
|}

== Usage Examples ==

=== Basic Single Item Evaluation ===
<syntaxhighlight lang="python">
from evaluate_deepsearch_official import call_llm_judge

# Configure globals (normally done by main())
global judge_prompt, dataset, judge_model
judge_prompt = "Question: {question}\nCorrect Answer: {correct_answer}\nResponse: {response}\n\nIs the response correct? Answer 'Correct' or 'Incorrect'."
dataset = "browsecomp_en"
judge_model = "gpt-4o-2024-08-06"

# Evaluate single item
item = {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "prediction": "The capital of France is Paris."
}

result = call_llm_judge(item)
print(f"Judgement: {result['judgement']}")  # "Correct"
</syntaxhighlight>

=== Parallel Evaluation with ThreadPoolExecutor ===
<syntaxhighlight lang="python">
import concurrent.futures
from tqdm import tqdm

items = [...]  # List of prediction items

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    futures = {executor.submit(call_llm_judge, item): item for item in items}
    results = []

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        results.append(future.result())

# Count correct
correct = sum(1 for r in results if r["judgement"].lower() == "correct")
print(f"Accuracy: {correct/len(results)*100:.2f}%")
</syntaxhighlight>

=== Handling Different Datasets ===
<syntaxhighlight lang="python">
# Dataset-specific judge model selection
if dataset in ["gaia", "webwalker"]:
    judge_model = "openai/qwen2.5-72b-instruct"
elif dataset == "xbench-deepsearch":
    judge_model = "google/gemini-2.0-flash-001"
elif dataset.startswith("browsecomp"):
    judge_model = "gpt-4o-2024-08-06"

# BrowseComp uses structured JSON output with confidence
# XBench uses Chinese JSON format
# Others use plain text judgement
</syntaxhighlight>

== Related Pages ==

=== Implements Principle ===
* [[implements::Principle:Alibaba_NLP_DeepResearch_LLM_Judge_Scoring]]
