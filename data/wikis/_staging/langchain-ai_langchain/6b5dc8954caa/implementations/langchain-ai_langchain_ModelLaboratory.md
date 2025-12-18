{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Experimentation]], [[domain::Model Comparison]], [[domain::Evaluation]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Utility class `ModelLaboratory` for comparing outputs from multiple LLMs or chains on the same input.

=== Description ===

The `ModelLaboratory` class provides an interactive way to experiment with and compare the performance of different models or chains. It runs the same input through multiple chains and displays color-coded outputs for easy comparison. This is useful for evaluating which model performs best for a specific task.

=== Usage ===

Use this class during development to compare model outputs side-by-side. It helps with model selection by showing how different LLMs respond to the same prompt, making it easier to evaluate quality, style, and accuracy differences.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/model_laboratory.py libs/langchain/langchain_classic/model_laboratory.py]
* '''Lines:''' 1-99

=== Signature ===
<syntaxhighlight lang="python">
class ModelLaboratory:
    """A utility to experiment with and compare the performance of different models."""

    def __init__(self, chains: Sequence[Chain], names: list[str] | None = None):
        """Initialize with chains to experiment with.

        Args:
            chains: Sequence of chains (each must have exactly one input/output).
            names: Optional list of names for each chain.
        """

    @classmethod
    def from_llms(
        cls,
        llms: list[BaseLLM],
        prompt: PromptTemplate | None = None,
    ) -> ModelLaboratory:
        """Initialize with LLMs and optional prompt template."""

    def compare(self, text: str) -> None:
        """Compare model outputs on an input text."""
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.model_laboratory import ModelLaboratory
</syntaxhighlight>

== I/O Contract ==

=== Constructor Parameters ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| chains || Sequence[Chain] || Yes || Chains to compare (each must have single input/output)
|-
| names || list[str] | None || No || Display names for each chain
|}

=== from_llms Parameters ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| llms || list[BaseLLM] || Yes || List of LLMs to compare
|-
| prompt || PromptTemplate | None || No || Prompt template (defaults to passthrough)
|}

=== Outputs ===
{| class="wikitable"
|-
! Method !! Output !! Description
|-
| compare() || None || Prints color-coded outputs to stdout
|}

== Usage Examples ==

=== Compare LLMs Directly ===
<syntaxhighlight lang="python">
from langchain_classic.model_laboratory import ModelLaboratory
from langchain_openai import OpenAI
from langchain_anthropic import Anthropic

# Create laboratory from LLMs
llms = [
    OpenAI(model="gpt-3.5-turbo-instruct", temperature=0),
    OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7),
]
lab = ModelLaboratory.from_llms(llms)

# Compare outputs
lab.compare("What is the capital of France?")
# Output shows each model's response with color coding
</syntaxhighlight>

=== Compare with Custom Prompt ===
<syntaxhighlight lang="python">
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer concisely: {question}"
)

lab = ModelLaboratory.from_llms(llms, prompt=prompt)
lab.compare("Explain quantum computing")
</syntaxhighlight>

=== Compare Custom Chains ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain

# Create chains with different configurations
chains = [
    LLMChain(llm=llm1, prompt=prompt1),
    LLMChain(llm=llm2, prompt=prompt2),
]

lab = ModelLaboratory(chains, names=["Creative", "Factual"])
lab.compare("Write a haiku about programming")
</syntaxhighlight>

== Related Pages ==
* [[uses_concept::Concept:langchain-ai_langchain_LLM_Abstraction]]
* [[related_to::Implementation:langchain-ai_langchain_LLMChain]]

