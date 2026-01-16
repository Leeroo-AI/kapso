# Implementation: WebResummer_Main

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|Alibaba_NLP_DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
|-
! Domains
| [[domain::Entry_Point]], [[domain::Parallel_Execution]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2026-01-15 20:00 GMT]]
|}

== Overview ==
Multi-rollout parallel execution entry point for WebResummer agent with configurable model, dataset, and summary iteration parameters.

=== Description ===
The `main.py` module serves as the command-line entry point for running WebResummer agent evaluations. It implements:

- Command-line argument parsing for model, output, dataset, and parameters
- Multi-rollout execution (configurable number of parallel trajectories)
- ProcessPoolExecutor for parallel question processing
- Resume capability via processed question tracking
- Thread-safe output writing with locks per rollout

The script orchestrates the full evaluation pipeline from data loading through results writing.

=== Usage ===
Run from command line to execute WebResummer agent on benchmark datasets with multiple rollouts.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebResummer/src/main.py WebAgent/WebResummer/src/main.py]
* '''Lines:''' 1-164

=== Signature ===
<syntaxhighlight lang="python">
# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--output", type=str, default="")
parser.add_argument("--dataset", type=str, default="gaia")
parser.add_argument("--temperature", type=float, default=0.85)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_workers", type=int, default=20)
parser.add_argument("--roll_out_count", type=int, default=3)
parser.add_argument("--summary_iteration", type=int, default=1000)

# Main execution
if __name__ == "__main__":
    # Parse args
    args = parser.parse_args()

    # Load dataset
    items = load_jsonl(f"eval_data/{args.dataset}.jsonl")

    # Initialize agent
    test_agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["search", "visit"],
        system_message=SYSTEM_PROMPT
    )

    # Run parallel execution
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        ...
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python WebAgent/WebResummer/src/main.py --model MODEL --output OUTPUT
</syntaxhighlight>

== I/O Contract ==

=== Inputs (CLI Args) ===
{| class="wikitable"
|-
! Arg !! Type !! Default !! Description
|-
| --model || str || "" || Model path or name
|-
| --output || str || "" || Output directory base
|-
| --dataset || str || "gaia" || Dataset name
|-
| --temperature || float || 0.85 || Sampling temperature
|-
| --top_p || float || 0.95 || Top-p sampling
|-
| --max_workers || int || 20 || Parallel workers
|-
| --roll_out_count || int || 3 || Number of rollouts
|-
| --summary_iteration || int || 1000 || ReSum trigger interval
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| iter{N}.jsonl || File || Results for each rollout iteration
|-
| stdout || Text || Progress and status messages
|}

== Usage Examples ==

=== Basic Execution ===
<syntaxhighlight lang="bash">
# Run WebResummer on GAIA dataset
python WebAgent/WebResummer/src/main.py \
    --model /path/to/qwen-model \
    --output ./results \
    --dataset gaia \
    --roll_out_count 3 \
    --max_workers 10
</syntaxhighlight>

=== With Custom Parameters ===
<syntaxhighlight lang="bash">
# Custom temperature and summary iteration
python WebAgent/WebResummer/src/main.py \
    --model qwen-max \
    --output ./results/custom \
    --dataset browsecomp_en \
    --temperature 0.7 \
    --top_p 0.9 \
    --roll_out_count 5 \
    --summary_iteration 500 \
    --max_workers 30
</syntaxhighlight>

=== Output Structure ===
<syntaxhighlight lang="text">
results/
├── qwen-max_sglang/
│   └── gaia/
│       ├── iter1.jsonl
│       ├── iter2.jsonl
│       └── iter3.jsonl
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Multi_Turn_Agent_Loop]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
