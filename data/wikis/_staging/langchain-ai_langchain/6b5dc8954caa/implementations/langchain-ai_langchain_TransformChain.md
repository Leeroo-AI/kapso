{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
|-
! Domains
| [[domain::Chains]], [[domain::Data_Transformation]], [[domain::Pipelines]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Chain class `TransformChain` that wraps arbitrary Python functions to integrate custom transformations into chain pipelines.

=== Description ===

The `TransformChain` allows any synchronous or asynchronous Python function to be used as a chain step. It defines explicit input and output variable names, making it composable with other chains in sequential pipelines. This is useful for data preprocessing, postprocessing, or any custom logic that needs to be part of a chain workflow.

=== Usage ===

Use `TransformChain` when you need to insert custom Python logic into a Chain-based pipeline. For LCEL-based code, prefer using `RunnableLambda` which provides similar functionality with better composability.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain/langchain_classic/chains/transform.py libs/langchain/langchain_classic/chains/transform.py]
* '''Lines:''' 1-79

=== Signature ===
<syntaxhighlight lang="python">
class TransformChain(Chain):
    """Chain that transforms the chain output using a custom function.

    Attributes:
        input_variables: Keys expected by the transform function.
        output_variables: Keys returned by the transform function.
        transform_cb: Synchronous transform function (alias: transform).
        atransform_cb: Async transform function (alias: atransform).
    """

    input_variables: list[str]
    output_variables: list[str]
    transform_cb: Callable[[dict[str, str]], dict[str, str]]
    atransform_cb: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain_classic.chains import TransformChain
</syntaxhighlight>

== I/O Contract ==

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| (varies) || Any || Yes || Keys specified in input_variables
|}

=== Outputs ===
{| class="wikitable"
|-
! Name !! Type !! Description
|-
| (varies) || Any || Keys specified in output_variables
|}

== Usage Examples ==

=== Basic Transform ===
<syntaxhighlight lang="python">
from langchain_classic.chains import TransformChain

# Define a transform function
def clean_text(inputs: dict) -> dict:
    """Clean and normalize text."""
    text = inputs["raw_text"]
    cleaned = text.lower().strip()
    cleaned = " ".join(cleaned.split())  # Normalize whitespace
    return {"cleaned_text": cleaned}

# Create chain
clean_chain = TransformChain(
    input_variables=["raw_text"],
    output_variables=["cleaned_text"],
    transform=clean_text,
)

# Use the chain
result = clean_chain.invoke({"raw_text": "  Hello   WORLD  "})
print(result["cleaned_text"])  # "hello world"
</syntaxhighlight>

=== Async Transform ===
<syntaxhighlight lang="python">
import asyncio

async def async_transform(inputs: dict) -> dict:
    """Async transformation (e.g., API call)."""
    await asyncio.sleep(0.1)  # Simulate async work
    return {"result": inputs["data"].upper()}

transform_chain = TransformChain(
    input_variables=["data"],
    output_variables=["result"],
    transform=lambda x: {"result": x["data"].upper()},  # Sync fallback
    atransform=async_transform,
)

# Async invocation
result = await transform_chain.ainvoke({"data": "hello"})
</syntaxhighlight>

=== In Sequential Chain ===
<syntaxhighlight lang="python">
from langchain_classic.chains import SequentialChain, LLMChain

# Preprocess input
def preprocess(inputs: dict) -> dict:
    return {"processed_query": f"Please answer: {inputs['query']}"}

preprocess_chain = TransformChain(
    input_variables=["query"],
    output_variables=["processed_query"],
    transform=preprocess,
)

# LLM chain uses preprocessed input
llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate(
        input_variables=["processed_query"],
        template="{processed_query}",
    ),
    output_key="answer",
)

# Compose
pipeline = SequentialChain(
    chains=[preprocess_chain, llm_chain],
    input_variables=["query"],
    output_variables=["answer"],
)
</syntaxhighlight>

=== Modern LCEL Alternative ===
<syntaxhighlight lang="python">
from langchain_core.runnables import RunnableLambda

# LCEL equivalent
clean_runnable = RunnableLambda(
    lambda x: {"cleaned_text": x["raw_text"].lower().strip()}
)

# Composable with pipe
chain = clean_runnable | some_other_runnable
</syntaxhighlight>

== Related Pages ==
* [[requires_env::Environment:langchain-ai_langchain_LangChain_Runtime_Environment]]
* [[uses_heuristic::Heuristic:langchain-ai_langchain_Warning_Deprecated_langchain_classic]]

