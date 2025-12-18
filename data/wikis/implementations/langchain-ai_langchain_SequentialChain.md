{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Chains]], [[domain::Composition]], [[domain::Pipelines]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Chain classes `SequentialChain` and `SimpleSequentialChain` that compose multiple chains into sequential pipelines where outputs feed into inputs.

=== Description ===

The `SequentialChain` class connects multiple chains in sequence, where the outputs of one chain become available as inputs to subsequent chains. `SimpleSequentialChain` is a simplified version for single-input/output chains. Both validate at construction that the chain dependencies are satisfiable. These classes are largely superseded by LCEL's `|` operator which provides more flexible composition.

=== Usage ===

Use these classes for legacy code that needs to compose multiple Chain objects. For new code, prefer LCEL composition with `chain1 | chain2` which offers better streaming and async support.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/sequential.py libs/langchain/langchain_classic/chains/sequential.py]
* '''Lines:''' 1-208

=== Signature ===
<syntaxhighlight lang="python">
class SequentialChain(Chain):
    """Chain where outputs of one chain feed directly into next.

    Attributes:
        chains: List of chains to execute in sequence.
        input_variables: Required input keys.
        output_variables: Keys to return in output.
        return_all: Whether to return all intermediate outputs.
    """

    chains: list[Chain]
    input_variables: list[str]
    output_variables: list[str]
    return_all: bool = False


class SimpleSequentialChain(Chain):
    """Simple chain for single input/output chains.

    Attributes:
        chains: List of chains (each must have 1 input and 1 output).
        strip_outputs: Whether to strip whitespace from outputs.
        input_key: Key for input (default: "input").
        output_key: Key for output (default: "output").
    """

    chains: list[Chain]
    strip_outputs: bool = False
    input_key: str = "input"
    output_key: str = "output"
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains import SequentialChain, SimpleSequentialChain
</syntaxhighlight>

== I/O Contract ==

=== SequentialChain Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (varies) || Any || Yes || Keys specified in input_variables
|}

=== SequentialChain Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (varies) || Any || Keys specified in output_variables (or all if return_all=True)
|}

== Usage Examples ==

=== SimpleSequentialChain ===
<syntaxhighlight lang="python">
from langchain_classic.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI()

# Chain 1: Generate a title
title_prompt = PromptTemplate.from_template(
    "Generate a title for a story about: {input}"
)
title_chain = LLMChain(llm=llm, prompt=title_prompt)

# Chain 2: Write the story
story_prompt = PromptTemplate.from_template(
    "Write a short story with the title: {input}"
)
story_chain = LLMChain(llm=llm, prompt=story_prompt)

# Compose into sequence
sequential = SimpleSequentialChain(chains=[title_chain, story_chain])

# Run: topic -> title -> story
result = sequential.invoke({"input": "a robot learning to paint"})
print(result["output"])  # The complete story
</syntaxhighlight>

=== SequentialChain with Named Keys ===
<syntaxhighlight lang="python">
from langchain_classic.chains import SequentialChain

# Chain 1: Analyze sentiment
sentiment_prompt = PromptTemplate(
    input_variables=["review"],
    template="What is the sentiment of this review?\n\n{review}\n\nSentiment:",
)
sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt, output_key="sentiment")

# Chain 2: Generate response based on sentiment
response_prompt = PromptTemplate(
    input_variables=["review", "sentiment"],
    template="The review:\n{review}\n\nHas sentiment: {sentiment}\n\nWrite an appropriate response:",
)
response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="response")

# Compose with explicit input/output keys
overall_chain = SequentialChain(
    chains=[sentiment_chain, response_chain],
    input_variables=["review"],
    output_variables=["sentiment", "response"],  # Return both
)

result = overall_chain.invoke({"review": "This product is amazing!"})
print(f"Sentiment: {result['sentiment']}")
print(f"Response: {result['response']}")
</syntaxhighlight>

=== Modern LCEL Alternative ===
<syntaxhighlight lang="python">
from langchain_core.runnables import RunnablePassthrough

# LCEL provides more flexibility
title_chain = title_prompt | llm | StrOutputParser()
story_chain = story_prompt | llm | StrOutputParser()

# Compose with pipe operator
modern_chain = (
    {"topic": RunnablePassthrough()}
    | title_chain
    | {"title": RunnablePassthrough()}
    | story_chain
)

# Or simpler for direct piping
simple_modern = title_chain | story_chain
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

