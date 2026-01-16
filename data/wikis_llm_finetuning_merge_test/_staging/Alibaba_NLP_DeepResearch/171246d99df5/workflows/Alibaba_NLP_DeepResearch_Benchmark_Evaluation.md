# Workflow: Benchmark_Evaluation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|DeepResearch|https://github.com/Alibaba-NLP/DeepResearch]]
* [[source::Paper|Tongyi DeepResearch|https://arxiv.org/pdf/2510.24701]]
|-
! Domains
| [[domain::LLMs]], [[domain::Evaluation]], [[domain::Benchmarking]], [[domain::Web_Agents]]
|-
! Last Updated
| [[last_updated::2026-01-15 19:00 GMT]]
|}

== Overview ==

End-to-end process for evaluating web research agent performance using multi-rollout inference with LLM-as-Judge scoring across standard benchmarks (GAIA, BrowseComp, WebWalkerQA, XBench-DeepSearch).

=== Description ===

This workflow implements the evaluation pipeline for Tongyi DeepResearch and related web agents. It supports multiple benchmark datasets and uses a multi-rollout strategy (typically 3 independent runs per question) to compute Pass@k metrics. The evaluation employs LLM judges (Qwen2.5-72B, GPT-4o, or Gemini) to assess answer correctness against ground truth, providing both overall accuracy metrics and detailed statistics on agent behavior.

Key features:
* **Multi-rollout inference**: Parallel execution of N independent runs per question
* **LLM-as-Judge**: Structured judgment prompts with configurable models
* **Pass@k metrics**: Computing Pass@1, Pass@3, Avg Pass@3, and Best Pass@1
* **Behavioral statistics**: Tool usage frequency, token consumption, termination analysis

=== Usage ===

Execute this workflow when you need to:
* Evaluate a web research agent on standard benchmarks
* Compare model performance across different datasets or configurations
* Generate detailed statistics on agent behavior and resource usage
* Produce reproducible evaluation results with multi-run aggregation

The input is a trained model and evaluation dataset, and the output is comprehensive metrics including accuracy scores and behavioral analysis.

== Execution Steps ==

=== Step 1: Environment Setup ===

Configure the evaluation environment with required API keys and model paths. Set up the judge model configuration based on the target benchmark dataset.

'''Configuration requirements:'''
* OPENAI_API_KEY for judge model access (GPT-4o for BrowseComp)
* LiteLLM configuration for Qwen judge access
* Model path for tokenizer-based statistics
* Output directory for results storage

=== Step 2: Multi_rollout Inference ===

Execute the agent multiple times (typically 3 rollouts) on each question in the dataset. Each rollout runs independently with its own random seed and is saved to a separate output file.

'''Execution details:'''
* ThreadPoolExecutor for parallel question processing
* Round-robin port assignment for load balancing across vLLM instances
* Resume capability via processed query tracking
* Per-rollout JSONL output files (iter1.jsonl, iter2.jsonl, iter3.jsonl)

=== Step 3: Result Collection ===

Load and parse the inference results from all rollout files. Each result contains the question, ground truth answer, model prediction, conversation messages, and termination reason.

'''Data structure:'''
* question: Original input query
* answer: Ground truth reference
* prediction: Agent's final answer (extracted from <answer> tags)
* messages: Full conversation trace
* termination: Stop reason (answer, token_limit, time_limit, etc.)

=== Step 4: LLM Judge Scoring ===

Submit each prediction to the LLM judge for correctness assessment. The judge receives the question, ground truth, and model response, then determines if the answer is correct.

'''Judge configuration by dataset:'''
* GAIA/WebWalker: Qwen2.5-72B-Instruct with standard prompt
* BrowseComp: GPT-4o with structured JSON output (confidence scoring)
* XBench-DeepSearch: Gemini-2.0-Flash with Chinese response format
* HLE: Specialized prompt for Humanity's Last Exam

=== Step 5: Metric Aggregation ===

Aggregate judgment results across all rollouts to compute Pass@k metrics. Track which questions were answered correctly in each rollout and compute overall statistics.

'''Computed metrics:'''
* Pass@1: Percentage correct in single rollout (per-round and best)
* Pass@3: Percentage with at least one correct answer across 3 rollouts
* Avg Pass@3: Average accuracy across all 3 rollouts
* Best Pass@1: Maximum single-rollout accuracy

=== Step 6: Behavioral Statistics ===

Analyze agent behavior patterns from the conversation traces. Compute statistics on tool usage, token consumption, and termination reasons.

'''Statistics gathered:'''
* Average tool calls per question (search, visit, other)
* Average assistant tokens per message and per question
* Termination frequency distribution
* Invalid answer format rate
* Extra-long context occurrences

=== Step 7: Enhanced Statistics ===

Compute additional statistics specifically for correctly-solved questions. This provides insight into the efficiency of successful reasoning paths.

'''Enhanced metrics:'''
* Tool calls per correctly-solved question
* Assistant tokens per correctly-solved question
* Comparison with overall population statistics

=== Step 8: Result Export ===

Export comprehensive evaluation results to output files. Generate scored versions of each rollout file and a summary JSONL with all metrics.

'''Output artifacts:'''
* iter{N}_scored.jsonl: Per-question judgments with is_correct flag
* summary.jsonl: Aggregated metrics and statistics
* Console output: Human-readable performance summary

== Execution Diagram ==

{{#mermaid:graph TD
    A[Environment Setup] --> B[Multi-rollout Inference]
    B --> C[Result Collection]
    C --> D[LLM Judge Scoring]
    D --> E[Metric Aggregation]
    E --> F[Behavioral Statistics]
    F --> G[Enhanced Statistics]
    G --> H[Result Export]
    H --> I[Evaluation Complete]
}}

== GitHub URL ==

[[github_url::https://github.com/leeroo-coder/workflow-alibaba-nlp-deepresearch-benchmark-evaluation]]
