# Implementation: WebSailor_Main

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
Multi-rollout parallel execution runner for WebSailor agent with support for multiple benchmark datasets.

=== Description ===
The `run_multi_react.py` module provides the main entry point for WebSailor evaluations:

- Support for multiple datasets: GAIA, BrowseComp, SimpleQA, WebWalker, etc.
- Multi-rollout execution with configurable parallelism
- ThreadPoolExecutor for concurrent question processing
- Resume capability via processed question tracking
- Per-rollout output files with thread-safe writing

The script handles dataset-specific question extraction and supports diverse benchmark formats.

=== Usage ===
Run from command line to execute WebSailor on benchmarks. Supports various dataset formats.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/Alibaba-NLP/DeepResearch Alibaba_NLP_DeepResearch]
* '''File:''' [https://github.com/Alibaba-NLP/DeepResearch/blob/main/WebAgent/WebSailor/src/run_multi_react.py WebAgent/WebSailor/src/run_multi_react.py]
* '''Lines:''' 1-188

=== Signature ===
<syntaxhighlight lang="python">
# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--output", type=str, default="")
parser.add_argument("--dataset", type=str, default="gaia",
    choices=["gaia", "browsecomp_zh", "browsecomp_en", "webwalker",
             "simple_qa", "time_qa", "xbench-deepsearch", "hle"])
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_workers", type=int, default=20)
parser.add_argument("--sys_prompt", type=str, default="SYSTEM_PROMPT_MULTI")
parser.add_argument("--roll_out_count", type=int, default=3)

if __name__ == "__main__":
    args = parser.parse_args()
    # ... execution logic
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
# Run as script
python WebAgent/WebSailor/src/run_multi_react.py --model MODEL --output OUTPUT
</syntaxhighlight>

== I/O Contract ==

=== Inputs (CLI Args) ===
{| class="wikitable"
|-
! Arg !! Type !! Default !! Description
|-
| --model || str || "" || Model path
|-
| --output || str || "" || Output directory
|-
| --dataset || str || "gaia" || Dataset name
|-
| --temperature || float || 0.6 || Sampling temp
|-
| --max_workers || int || 20 || Parallel workers
|-
| --roll_out_count || int || 3 || Rollout count
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| iter{N}.jsonl || File || Results per rollout
|}

== Usage Examples ==

=== Run on GAIA ===
<syntaxhighlight lang="bash">
python WebAgent/WebSailor/src/run_multi_react.py \
    --model /path/to/model \
    --output ./results \
    --dataset gaia \
    --roll_out_count 3
</syntaxhighlight>

=== Run on BrowseComp ===
<syntaxhighlight lang="bash">
python WebAgent/WebSailor/src/run_multi_react.py \
    --model qwen-max \
    --output ./results \
    --dataset browsecomp_en \
    --temperature 0.7 \
    --max_workers 30
</syntaxhighlight>

== Related Pages ==
* [[implements::Principle:Alibaba_NLP_DeepResearch_Multi_Turn_Agent_Loop]]
* [[requires_env::Environment:Alibaba_NLP_DeepResearch_Python_Dependencies]]
