# Principle: LLM_Judge_Scoring

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

LLM-as-Judge evaluation paradigm where large language models score agent responses against ground truth answers using dataset-specific prompts.

=== Description ===

LLM Judge Scoring is an evaluation methodology that leverages large language models as automated judges to assess the correctness of agent-generated responses. Rather than relying on exact string matching or human evaluation, this approach uses the semantic understanding capabilities of LLMs to determine whether an agent's prediction correctly answers a question relative to a ground truth answer.

The DeepResearch evaluation framework supports multiple judge models:

1. **GPT-4o (gpt-4o-2024-08-06)** - Primary judge for BrowseComp English and Chinese benchmarks
2. **Qwen2.5-72B-Instruct** - Judge for GAIA and WebWalker datasets
3. **Gemini 2.0 Flash** - Judge for XBench-DeepSearch (Chinese benchmark)

Each dataset uses a specialized prompt template optimized for its evaluation criteria. The judge receives the original question, ground truth answer, and agent prediction, then outputs a structured judgement.

=== Usage ===

Use LLM Judge Scoring when:
- Evaluating open-ended question answering where exact match is insufficient
- Assessing responses that may be semantically correct but lexically different
- Comparing agent performance across diverse benchmark datasets
- Requiring automated evaluation at scale without human annotators

This principle underlies the entire DeepResearch benchmark evaluation pipeline.

== Theoretical Basis ==

The LLM-as-Judge paradigm is based on the observation that large language models can perform reliable semantic similarity assessment between candidate and reference answers.

'''Evaluation Process:'''

<syntaxhighlight lang="text">
Input:
  - Question: The original query posed to the agent
  - Ground Truth: The known correct answer
  - Prediction: The agent's generated response

Judge Model Processing:
  1. Format inputs using dataset-specific prompt template
  2. Send formatted prompt to judge LLM
  3. Parse structured response (Correct/Incorrect)

Output:
  - Binary judgement: "Correct" or "Incorrect"
  - Optional: Confidence score (for some datasets)
</syntaxhighlight>

'''Dataset-Specific Handling:'''

{| class="wikitable"
|-
! Dataset !! Judge Model !! Output Format
|-
| BrowseComp (EN/ZH) || GPT-4o || JSON with confidence score
|-
| GAIA, WebWalker || Qwen2.5-72B || Plain text judgement
|-
| XBench-DeepSearch || Gemini 2.0 Flash || Chinese JSON format
|}

'''Structured Output Schema (BrowseComp):'''
<syntaxhighlight lang="json">
{
  "extracted_final_answer": "string",
  "reasoning": "string",
  "correct": "yes" | "no",
  "confidence": number,
  "strict": boolean
}
</syntaxhighlight>

The judge model selection balances cost, availability, and accuracy for each benchmark's specific requirements.

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:Alibaba_NLP_DeepResearch_Call_Llm_Judge]]
